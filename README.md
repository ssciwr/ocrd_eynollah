# ocrd_eynollah

OCR-D wrapper for the Eynollah inference.

**Work in progress.**

## Installation

For CPU use:

```python
python -m pip install -e .[tests]
```

## Usage

First, creating a workspace and adding image files to it:

```bash
mkdir myworkspace
cd myworkspace
ocrd workspace init
ocrd workspace add \
  -G {FILE_GRP} \
  -i {FILE_ID} \
  -m {MIMETYPE} \
  -g {PAGE_ID} \
  {PATH_TO_FILE} 
```

For example, `OCR-D-IMG` for `FILE_GRP`, `FILE_001` for `FILE_ID`, `image/tiff` for `MIMETYPE`, `PAGE_001` for `PAGE_ID` and `/path/to/file.tif` for `PATH_TO_FILE`.

Then download a specific trained Eynollah model or all available models, if needed. See [ocrd-tool.json](ocrd_eynollah/ocrd-tool.json) for the list of available models.

```bash
ocrd resmgr download ocrd-eynollah-inference eynollah-scale-bin-20260325-artbound-noheadings
```

or

```bash
ocrd resmgr download ocrd-eynollah-inference '*'
```

On Linux, the models will be downloaded to `~/.local/share/ocrd-resources/ocrd-eynollah-inference/model_name`

Finally, run the Eynollah inference via an ocrd processor:

```bash
ocrd-eynollah-inference \
  -I {INPUT_FILE_GRP} \
  -O {OUTPUT_FILE_GRP} \
  -P model {MODEL_NAME}
```

For example:

```bash
ocrd-eynollah-inference \
  -I OCR-D-IMG \
  -O OCR-D-EYNOLLAH \
  -P model eynollah-scale-bin-20260325-artbound-noheadings
```

Results will be stored in the `OUTPUT_FILE_GRP` file group, including:
- PAGE XML files with the detected layout regions and their coordinates,
- Alternative image with the layout overlayed on the original image, and 
- Alternative image with only the layout visualization