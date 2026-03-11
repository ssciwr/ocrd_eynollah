from typing import Optional
import click
import os
from pathlib import Path
import json
import tempfile
import cv2
import numpy as np

from ocrd import Processor, Workspace, OcrdPage, OcrdPageResult, OcrdPageResultImage
from ocrd_models.ocrd_page import CoordsType, TextRegionType, ImageRegionType
from ocrd_utils import points_from_bbox, points_from_polygon
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from eynollah.training.inference import sbb_predict as EynollahInference


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
        pass  # not really sure about this

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

        # Add output PAGE-XML to workspace
        file_id = self.output_file_grp + "_" + page_id

        self.workspace.add_file(
            ID=file_id,
            file_grp=self.output_file_grp,
            pageId=page_id,
            mimetype="application/vnd.prima.page+xml",
            local_filename=os.path.join(self.output_file_grp, file_id + ".xml"),
            content=pcgts.to_xml(),
        )

        return OcrdPageResult(pcgts)

    def _add_regions_from_layout(self, page, layout_mask):
        """Convert segmentation mask to PAGE regions."""

        layout_mask = layout_mask.astype(np.uint8)

        labels = np.unique(layout_mask)

        region_counter = 0

        for label in labels:

            if label == 0:
                continue  # background

            binary = (layout_mask == label).astype(np.uint8)

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:

                if cv2.contourArea(contour) < 200:
                    continue

                polygon = contour.squeeze()

                if polygon.ndim != 2:
                    continue

                points = " ".join(f"{x},{y}" for x, y in polygon)

                region = TextRegionType(
                    id=f"region_{region_counter}",
                    Coords=CoordsType(points=points),
                )

                page.add_TextRegion(region)

                region_counter += 1


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(EynollahInferenceProcessor, *args, **kwargs)
