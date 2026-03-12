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

from ocrd import Processor, Workspace, OcrdPage, OcrdPageResult, OcrdPageResultImage
from ocrd_models.ocrd_page import (
    CoordsType,
    TextRegionType,
    ImageRegionType,
    AlternativeImageType,
)
from ocrd_utils import points_from_bbox, points_from_polygon
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from eynollah.training.inference import sbb_predict as EynollahInference


# color coding for Eynollah inference results,
# used in visualize_model_output() method of sbb_predict class
# the below colors are extracted from kimlee87/eynollah.git@new_inference_color
# the original colors from eynollah.git@main are commented for reference
# colors here are in RGB, not BGR as in Eynollah visualize_model_output()
eynollah_inference_colors = {
    # (R, G, B): (label, region type, region label, subtype)
    (255, 255, 255): (
        "background",
        None,
        None,
        None,
    ),  # background, same color as original
    (231, 76, 60): (
        "image",
        ImageRegionType,
        "ImageRegion",
        None,
    ),  # image, orginally (0, 0, 255)
    (52, 152, 219): (
        "text",
        TextRegionType,
        "TextRegion",
        None,
    ),  # text, orginally (0, 125, 255)
    (230, 126, 34): (
        "heading",
        TextRegionType,
        "TextRegion",
        "heading",
    ),  # heading, orginally (125, 0, 255)
    (155, 89, 182): (
        "separator",
        ImageRegionType,
        "ImageRegion",
        None,
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

        model_dir = self.parameter["model"]
        device = self.parameter.get(
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
            cpu=(device == "cpu"),
            out=None,  # no output directory for now
        )

        self.detector.start_new_session_and_model()

    def shutdown(self) -> None:
        # TODO check if we need to close tensorflow session
        # But the session is store locally and not in the class
        if hasattr(self, "detector"):
            del self.detector

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
                self.detector.img_org,  # assined in predict method
                self.detector.task,  # assined when initializing the EynollahInference instance
            )

        # convert segmentation mask to PAGE regions
        self._add_regions_from_layout(page, only_layout)

        # add output PAGE-XML to workspace
        file_id = layout_path.stem

        self.workspace.add_file(
            ID=file_id,
            file_grp=self.output_file_grp,
            pageId=page_id,
            mimetype="application/vnd.prima.page+xml",
            local_filename=os.path.join(self.output_file_grp, file_id + ".xml"),
            content=pcgts.to_xml(),
        )

        # record alternative image with layout overlayed
        alt_img = AlternativeImageType(
            filename=os.path.join(self.output_file_grp, file_id + "_overlayed.png"),
            comments="Eynollah inference result overlayed on original image",
        )
        result.pcgts.Page.add_AlternativeImage(alt_img)

        # also add the layout image as an OcrdPageResultImage
        result.images.append(OcrdPageResultImage(only_layout, ".layout", alt_img))

        return result

    def _polygons_from_rgb_array(arr, skip_colors=[(0, 0, 0)]):
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
                    poly = Polygon(coords)

                    if poly.is_valid and poly.area > 0:
                        polygons_by_color[color].append(poly)

        # # Merge polygons per color
        # for color, polys in polygons_by_color.items():
        #     polygons_by_color[color] = unary_union(polys)

        return polygons_by_color

    def _add_regions_from_layout(
        self, page: OcrdPage, layout_mask: np.ndarray, skip_colors=[(0, 0, 0)]
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

            for poly in polygons:
                coords = CoordsType(points_from_polygon(poly))
                region = region_type(
                    id=f"region_{region_idx+1:04d}_{label}", Coords=coords, type=subtype
                )
                # add the region to the PAGE XML structure
                getattr(page, f"add_{region_label}")(
                    region
                )  # e.g. page.add_TextRegion(region)


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(EynollahInferenceProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
