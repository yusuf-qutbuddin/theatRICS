# Quickstart

This quickstart assumes you have an image stack in **CZI** or **TIFF** format.

## Typical workflow (RICS)
1. Start the program:
   ```bash
   theatrics
   ```
2. Go to **RICS Export** tab.
3. Select an input file (or folder for batch).
4. Choose:
   - Channel
   - Crop factor
   - Window size (odd)
   - Optional drift correction
5. Click **Export RICS**.
6. Go to **RICS Fitting** tab.
7. Load the produced `*_RICScorr.tif` (or run batch fitting).
8. Enter microscope parameters (pixel size, dwell time, line time, PSF).
9. Click **Run 2D/3D Fitting**.

## Typical workflow (pSFCS)
1. Go to **SFCS** tab.
2. Select input file and channel.
3. Optional: enable bleach correction.
4. Click **Correlate** to generate the autocorrelation curve.

## Diffusion map
1. Go to **RICS Fitting** tab.
2. In *Diffusion Map Fitting Parameters*, select an input file.
3. Set window size and offset.
4. Click **Generate Diffusion Map**.