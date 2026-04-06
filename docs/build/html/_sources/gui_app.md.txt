# GUI overview

The application is organized into tabs:

## Image Simulation
Generate synthetic stacks for testing RICS/SFCS analysis.

Common parameters:
- **Image shape** (pixels)
- **Number of frames**
- **Pixel dwell time (µs)** and **pixel size (nm)**
- **Number of particles**, **brightness**, **diffusion coefficients**
- **PSF sigma**

Outputs are saved as TIFF stacks.

## RICS Export
Compute RICS correlation maps and uncertainty maps from an input stack.

Inputs:
- Single file or batch folder
- Channel selection

Key parameters:
- **Crop factor**: fraction of the field used for correlation (reduces edge artefacts).
- **Window size**: local window size (odd integer).
- **Correct drift**: optional drift correction before correlation.

## RICS Fitting
Fit diffusion models to the exported RICS correlation map.

Key parameters:
- **Pixel size (nm)**
- **Pixel dwell (µs)**
- **Line time (ms)**
- **PSF size XY (µm)** and **aspect ratio**
- **Diffusion model**: 2D, 3D, 2-component (depending on your implementation)
- Optional **1D fast axis fit**

Batch fitting is supported via a folder selection.

## SFCS
Compute perpendicular scanning FCS autocorrelation curves.

Key parameters:
- Channel
- Number of CPU cores
- Optional bleach correction

## Results & Logs
Contains the internal log output, plus plot export and session save/load features.