# Workflow: RICS export

## Purpose
This step converts an input image stack into:
- a **RICS correlation map**
- an **uncertainty map**
- optionally corrected/processed intermediate stacks

## Steps
1. Open the **RICS Export** tab.
2. Select **Input file** (single) or **Input folder** (batch).
3. Select the **channel** to analyze.
4. Choose analysis parameters:
   - **Crop factor**
   - **Window size (odd)**
   - **Correct drift** (optional)
5. Click **Export RICS**.

## Output
The application writes TIFF outputs to disk (filenames may include patterns such as
`*_RICScorr.tif` and `*_RICSunc.tif` depending on your worker implementation).
See the **Outputs** page for conventions.