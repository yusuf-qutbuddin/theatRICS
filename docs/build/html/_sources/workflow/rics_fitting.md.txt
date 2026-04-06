# Workflow: RICS fitting

## Purpose
Fit a diffusion model to an exported RICS correlation map.

## Inputs
- A RICS correlation map TIFF (often `*_RICScorr.tif`)
- Microscope timing and PSF parameters

## Steps (single file)
1. Open **RICS Fitting** tab.
2. Select the **RICS map file**.
3. Set:
   - Pixel size (nm)
   - Pixel dwell (µs)
   - Line time (ms)
   - PSF size XY (µm)
   - PSF aspect ratio
4. Choose the **diffusion model**.
5. Set crop factors for fitting (fast/slow).
6. Click **Run 2D/3D Fitting**.

## Batch mode
1. Choose an **input folder** containing RICS maps.
2. Set parameters as above.
3. Run fitting; results are saved to your specified results file.

## Outputs
- A results CSV (summary per file)
- An NPZ file per fit (model arrays / residuals)
- Optional SVG figures (depending on settings)