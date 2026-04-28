from typing import Optional
import click
import os
from pathlib import Path
import json
import tempfile
import cv2
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
from collections.abc import Generator
from typing import TypedDict

from ocrd import Processor, OcrdPage, OcrdPageResult, OcrdPageResultImage
from ocrd_models.ocrd_page import (
    CoordsType,
    TextRegionType,
    ImageRegionType,
    LineDrawingRegionType,
    SeparatorRegionType,
    AlternativeImageType,
)
from ocrd_utils import points_from_polygon
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from eynollah.training.inference import sbb_predict as EynollahInference

from PIL import Image


# color coding for Eynollah inference results,
# used in visualize_model_output() method of sbb_predict class
# the below colors are extracted from kimlee87/eynollah.git@new_inference_color
# the original colors from eynollah.git@main are commented for reference
# colors here are the same as in Eynollah visualize_model_output()
eynollah_inference_colors = {
    # (B, G, R): (label, region type, region label, subtype)
    # subtype should comply with the allowed values in PAGE schema (...TypeSimpleType)
    # https://ocr-d.de/en/gt-guidelines/pagexml/Simple_Type.html
    (255, 255, 255): (
        "background",
        None,
        None,
        None,
    ),  # background, same color as original
    (0, 204, 0): (
        "artificial_boundary",
        LineDrawingRegionType,
        "LineDrawingRegion",
        None,  # no type attribute in LineDrawingRegion
    ),
    (60, 76, 231): (
        "text",
        TextRegionType,
        "TextRegion",
        "paragraph",  # here, caption or header/footer is also possibe but we use paragraph for simplicity
    ),  # text, orginally (0, 0, 255)
    (219, 152, 52): (
        "image",
        ImageRegionType,
        "ImageRegion",
        None,  # no type attribute in ImageRegion
    ),  # image, orginally (0, 125, 255)
    (34, 126, 230): (
        "heading",
        TextRegionType,
        "TextRegion",
        "heading",
    ),  # heading, orginally (125, 0, 255)
    (182, 89, 155): (
        "separator",
        SeparatorRegionType,
        "SeparatorRegion",
        None,  # no type attribute in SeparatorRegion
    ),  # separator, orginally (125, 125, 125)
}

eynollah_inference_colors_noheading = {
    (
        34,
        126,
        230,
    ): (  # in case the model does not predict heading, the color for heading will be used for separator
        "separator",
        SeparatorRegionType,
        "SeparatorRegion",
        None,
    ),
    (182, 89, 155): None,
}


class PolygonDict(TypedDict):  # for type hinting
    shell: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]]


# wrap Eynollah inference into an ocrd processor class
class EynollahInferenceProcessor(Processor):
    """OCR-D Processor for Eynollah inference"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        """Override setup of the Processor class to perform any necessary setup before processing pages."""
        # check if the required parameters are provided
        if not self.parameter or "model" not in self.parameter:
            raise ValueError("Parameters 'model' is required for Eynollah inference.")

        model_name = self.parameter["model"]
        self.eynollah_inference_colors = eynollah_inference_colors
        if (
            "noheading" in model_name
        ):  # TODO: find another way in case model name has different pattern
            self.eynollah_inference_colors.update(eynollah_inference_colors_noheading)

        model_dir = self.resolve_resource(model_name)
        device = self.parameter.get(  # will be used for Eynollah 0.7.0
            "device", "cuda"
        )  # default to 'cuda' if not provided

        dummy_img = "dummy_img.png"  # use dummy image file to create an instance of EynollahInference

        # get model task
        with open(os.path.join(model_dir, "config.json")) as f:
            config_params_model = json.load(f)
        task = config_params_model["task"]

        self.detector = EynollahInference(
            image=dummy_img,  # replace the dummy image with the actual image in the process_page_pcgts method
            dir_in=None,  # infer individual image
            model=model_dir,
            task=task,
            config_params_model=config_params_model,
            patches=True,  # process in patches
            save=None,  # do not need the overlayed image
            save_layout=dummy_img,  # replace the dummy image with the actual resulting layout
            ground_truth=None,  # no ground truth for inference
            xml_file=None,  # no xml file for reading order
            out=None,  # no output directory as we infer individual image
            min_area=0,  # min area size for reading order detection
        )

        self.detector.start_new_session_and_model()

    def shutdown(self) -> None:
        # TODO check if we need to close tensorflow session
        # But the session is store locally and not in the class
        if hasattr(self, "detector"):
            del self.detector

    def _build_contours_hierarchy(
        self, contours: list[np.ndarray], hierarchy: np.ndarray
    ) -> list[dict]:
        """Build a hierarchy of contours based on the hierarchy information from OpenCV.

        Args:
            contours (list[np.ndarray]): List of contours as returned by cv2.findContours.
                Each contour is a NumPy array of shape (n_points, 1, 2)
                where n_points is the number of points in the contour.
            hierarchy (np.ndarray): Hierarchy information as returned by cv2.findContours.
                It is a NumPy array of shape (n_contours, 4)
                where each row corresponds to a contour and contains the indices of
                the next, previous, first child, and parent contours.
                `hierarchy` should be `hierarchy[0]` if cv2.findContours is called with cv2.RETR_TREE.

        Returns:
            list[dict]: A list of dictionaries representing the contour hierarchy.
                Each dictionary has the following keys:
                - "contour": The contour points as a NumPy array of shape (n_points, 2).
                - "children": A list of indices of child contours in the returned contour list.
                - "parent": The index of the parent contour in the returned contour list,
                    or -1 if there is no parent.
        """
        nodes = []

        for i, cnt in enumerate(contours):
            nodes.append(
                {
                    "contour": cnt.reshape(
                        -1, 2
                    ),  # reshape to 2 columns for x and y coordinates, infer the number of points with -1 placeholder
                    "children": [],
                    "parent": hierarchy[i][3],  # parent index
                }
            )

        # link children to parents
        for i, h in enumerate(hierarchy):
            parent_idx = h[3]
            if parent_idx != -1:  # if it has a parent
                nodes[parent_idx]["children"].append(i)  # add index of child

        return nodes

    def _extract_polygons_from_hierarchy_contours(
        self, nodes: list[dict], idx: int, is_outer: bool = True
    ) -> Generator[PolygonDict, None, None]:
        """Recursively extract polygons from the contour hierarchy.

        Args:
            nodes (list[dict]): The contour hierarchy as returned by `_build_contours_hierarchy`.
            idx (int): The index of the current contour node to process.
            is_outer (bool): Whether the current contour is an outer contour (True) or a hole (False).
                If there is a child inside a hole, this child is also an outer contour (island in the hole).

        Yields:
            PolygonDict: A dictionary representing a polygon with the following keys:
                - "shell": A list of (x, y) tuples representing the exterior boundary of the polygon.
                - "holes": A list of lists of (x, y) tuples, where each inner list represents a hole in the polygon.
        """
        node = nodes[idx]

        polygon = {"shell": node["contour"], "holes": []}

        for child_idx in node["children"]:
            if is_outer:
                # children of outer = holes
                hole = nodes[child_idx]["contour"]
                polygon["holes"].append(hole)

                # grandchildren of outer = islands in holes
                for grandchild_idx in nodes[child_idx]["children"]:
                    yield from self._extract_polygons_from_hierarchy_contours(
                        nodes, grandchild_idx, is_outer=True
                    )  # islands are also outer
            else:
                # children of inner = islands in holes
                yield from self._extract_polygons_from_hierarchy_contours(
                    nodes, child_idx, is_outer=True
                )  # islands are also outer
        yield polygon

    def _close_ring(self, ring: np.ndarray) -> np.ndarray:
        """Ensure that a ring of coordinates is closed
        by checking the first and last points and appending
        the first point to the end if they are not the same.
        """
        if not np.array_equal(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[0]])
        return ring

    def _is_valid_ring(self, ring: np.ndarray) -> bool:
        """Check if a ring of coordinates is valid for creating a polygon.
        A valid ring must have at least 4 points (including the closing point)
        and must be closed (the first and last points must be the same).
        """
        return len(ring) >= 4 and np.array_equal(ring[0], ring[-1])

    def _create_polygon(
        self, shell: np.ndarray, holes: list[np.ndarray]
    ) -> Polygon | None:
        """Create a Shapely Polygon from a shell and holes, ensuring that the rings are valid.
        If the shell or any hole is not a valid ring, return None.
        """
        shell = self._close_ring(shell)
        if not self._is_valid_ring(shell):
            return None

        valid_holes = []
        for hole in holes:
            hole = self._close_ring(hole)
            if self._is_valid_ring(hole):
                valid_holes.append(hole)

        try:
            polygon = Polygon(shell=shell, holes=valid_holes)
            if (
                not polygon.is_empty
            ):  # some polygons can be self intersecting, aka invalid but we still skip them
                return polygon
            else:
                return None
        except Exception as e:
            self.logger.warning("Failed to create polygon: %s", e)
            return None

    def _polygons_from_rgb_array(
        self, arr: np.ndarray, skip_colors=((255, 255, 255),)
    ) -> dict[tuple[int, int, int], list[Polygon]]:
        """Convert an RGB NumPy array into Shapely polygons grouped by RGB value."""

        polygons_by_color = defaultdict(list)

        # Get unique RGB values
        colors = np.unique(arr.reshape(-1, 3), axis=0)

        for color in colors:
            color = tuple(int(c) for c in color)
            if color in skip_colors:
                continue

            # Create mask for this RGB value
            mask = np.all(arr == color, axis=-1)
            mask = (
                mask.astype(np.uint8) * 255
            )  # convert boolean mask to uint8 (0 and 255) for contour detection

            # Extract contours with opencv
            contours, hierachy = cv2.findContours(
                mask,
                mode=cv2.RETR_TREE,  # retrieve all contours and reconstruct the full hierarchy of nested contours
                method=cv2.CHAIN_APPROX_SIMPLE,
            )
            if hierachy is None:  # no contours found
                continue

            hierachy = hierachy[
                0
            ]  # cv2.findContours returns hierarchy with shape (1, n_contours, 4)

            # extract polygons from contours and hierarchy
            nodes = self._build_contours_hierarchy(contours, hierachy)
            polygon_dicts = []

            for i, node in enumerate(nodes):
                if (
                    node["parent"] == -1
                ):  # only process outer contours (those without parent)
                    polygon_dicts.extend(
                        list(
                            self._extract_polygons_from_hierarchy_contours(
                                nodes, i, is_outer=True
                            )
                        )
                    )

            # convert polygon dicts to shapely Polygons and add to the result
            for poly_dict in polygon_dicts:
                polygon = self._create_polygon(poly_dict["shell"], poly_dict["holes"])
                if polygon is not None:
                    polygons_by_color[color].append(polygon)

        return polygons_by_color

    def _add_regions_from_layout(
        self, page: OcrdPage, layout_mask: np.ndarray, skip_colors=((255, 255, 255),)
    ) -> None:
        """Convert segmentation mask to PAGE regions."""

        layout_mask = layout_mask.astype(np.uint8)

        # get polygons from the layout mask
        polygons_by_color = self._polygons_from_rgb_array(
            layout_mask, skip_colors=skip_colors
        )

        region_idx = 0
        for color, polygons in polygons_by_color.items():
            info = self.eynollah_inference_colors.get(color, None)

            if info is None:
                self.logger.warning(
                    "Skipping color %s",
                    color,
                )
                continue

            label, region_type, region_label, subtype = info

            if region_type is None:
                continue  # skip background

            length = len(polygons)
            self.logger.debug(
                "Found %d polygons for color %s (label: %s)", length, color, label
            )
            zero_padding = len(str(length))
            for poly in polygons:
                coords = CoordsType(points_from_polygon(poly.exterior.coords))
                region = region_type(
                    id=f"region_{region_idx+1:0{zero_padding}d}_{label}",
                    Coords=coords,
                )
                if subtype and hasattr(region, "set_type"):
                    region.set_type(subtype)

                # add the region to the PAGE XML structure
                getattr(page, f"add_{region_label}")(
                    region
                )  # e.g. page.add_TextRegion(region)

                # handle holes in the polygon as separate regions with different id
                for hole in poly.interiors:
                    hole_coords = CoordsType(points_from_polygon(hole.coords))
                    hole_region = region_type(
                        id=f"region_{region_idx+1:0{zero_padding}d}_{label}_hole",
                        Coords=hole_coords,
                    )
                    if subtype and hasattr(hole_region, "set_type"):
                        hole_region.set_type(subtype)

                    getattr(page, f"add_{region_label}")(
                        hole_region
                    )  # e.g. page.add_TextRegion(hole_region)
                region_idx += 1

    def process_page_pcgts(
        self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None
    ) -> OcrdPageResult:
        """Override process_page_pcgts of the Processor class to perform
        Eynollah inference on the input page(s) and return the result as an OcrdPageResult.
        """
        assert input_pcgts
        assert input_pcgts[0]
        assert self.parameter  # default values or from CLI with -p or -P

        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()

        # get the image file path for the page
        img_filename = page.imageFilename
        img_filepath = os.path.join(self.workspace.directory, img_filename)

        # update image file path in the EynollahInference instance
        self.detector.image = img_filepath

        self.logger.info("Running Eynollah inference on page %s", page_id)

        # run inference and get the resulting layout
        with tempfile.TemporaryDirectory() as temp_dir:
            layout_path = Path(temp_dir) / f"{page_id}_layout.png"

            # update the save_layout parameter in the EynollahInference instance
            self.detector.save_layout = str(layout_path)

            # run inference
            inferred_result = self.detector.predict(image_dir=img_filepath)

            # get the layout
            img_seg_overlayed, only_layout = self.detector.visualize_model_output(
                inferred_result,
                self.detector.img_org,  # assigned in predict method
                self.detector.task,  # assigned when initializing the EynollahInference instance
            )

        # convert segmentation mask to PAGE regions
        self.logger.info(
            "Converting Eynollah layout to PAGE regions for page %s", page_id
        )
        self.logger.debug("Layout image shape: %s", only_layout.shape)  # for debugging
        self._add_regions_from_layout(page, only_layout)

        # convert layout to image
        # [:, :, ::-1] converts BGR to RGB for PIL
        only_layout_img = Image.fromarray(
            only_layout[:, :, ::-1].astype(np.uint8), mode="RGB"
        )
        img_seg_overlayed_img = Image.fromarray(
            img_seg_overlayed[:, :, ::-1].astype(np.uint8), mode="RGB"
        )

        # record alternative image with layout and layout overlayed on original image
        alt_img = AlternativeImageType(
            comments="Eynollah inference layout result",
        )
        page.add_AlternativeImage(alt_img)
        result.images.append(OcrdPageResultImage(only_layout_img, "layout", alt_img))

        alt_img_overlayed = AlternativeImageType(
            comments="Eynollah inference layout overlayed on original image",
        )
        page.add_AlternativeImage(alt_img_overlayed)
        result.images.append(
            OcrdPageResultImage(
                img_seg_overlayed_img, "layout_overlayed", alt_img_overlayed
            )
        )

        return result


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(EynollahInferenceProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
