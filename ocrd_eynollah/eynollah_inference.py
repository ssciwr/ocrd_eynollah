from typing import Optional
import click
import os
from pathlib import Path
import json
import tempfile
import cv2
import numpy as np
from collections import defaultdict
from skimage import measure
from shapely.geometry import Polygon

from ocrd import Processor, OcrdPage, OcrdPageResult, OcrdPageResultImage
from ocrd_models.ocrd_page import (
    CoordsType,
    TextRegionType,
    ImageRegionType,
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
    # (R, G, B): (label, region type, region label, subtype)
    (255, 255, 255): (
        "background",
        None,
        None,
        None,
    ),  # background, same color as original
    (0, 204, 0): (
        "artificial_boundary",
        ImageRegionType,
        "ImageRegion",
        "artificial_boundary",
    ),
    (60, 76, 231): (
        "text",
        TextRegionType,
        "TextRegion",
        None,
    ),  # text, orginally (0, 0, 255)
    (219, 152, 52): (
        "image",
        ImageRegionType,
        "ImageRegion",
        None,
    ),  # image, orginally (0, 125, 255)
    (34, 126, 230): (
        "heading",
        TextRegionType,
        "TextRegion",
        "heading",
    ),  # heading, orginally (125, 0, 255)
    (182, 89, 155): (
        "separator",
        ImageRegionType,
        "ImageRegion",
        "separator",
    ),  # separator, orginally (125, 125, 125)
}


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

        model_dir = self.resolve_resource(self.parameter["model"])
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

    def _polygons_from_rgb_array(
        self, arr: np.ndarray, skip_colors=((0, 0, 0),)
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

            # Extract contours
            contours = measure.find_contours(mask.astype(np.uint8), level=0.5)

            for contour in contours:
                coords = [(float(c[1]), float(c[0])) for c in contour]

                if len(coords) >= 3:
                    poly = Polygon(coords)  # assuming the polygon does not have holes

                    if poly.is_valid and poly.area > 0:
                        polygons_by_color[color].append(poly)

        # # Merge polygons per color
        # for color, polys in polygons_by_color.items():
        #     polygons_by_color[color] = unary_union(polys)

        return polygons_by_color

    def _add_regions_from_layout(
        self, page: OcrdPage, layout_mask: np.ndarray, skip_colors=((0, 0, 0),)
    ) -> None:
        """Convert segmentation mask to PAGE regions."""

        layout_mask = layout_mask.astype(np.uint8)

        # get polygons from the layout mask
        polygons_by_color = self._polygons_from_rgb_array(
            layout_mask, skip_colors=skip_colors
        )

        region_idx = 0
        for color, polygons in polygons_by_color.items():
            info = eynollah_inference_colors.get(color, None)

            if info is None:
                self.logger.warning(
                    "Color %s not found in the results of Eynollah inference, skipping",
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
                    type=subtype,
                )
                # add the region to the PAGE XML structure
                getattr(page, f"add_{region_label}")(
                    region
                )  # e.g. page.add_TextRegion(region)
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
