# Workflow: FRAP analysis

This tab performs **FRAP (Fluorescence Recovery After Photobleaching)** analysis on CZI image files containing circular ROI annotations.

The implementation uses a **2D diffusion model with an infinite reservoir** based on the Soumpasis recovery formulation, with optional correction for **imaging bleach**.

---

## Overview

The FRAP workflow supports:

- **Single-file analysis** of one `*.czi` file
- **Batch analysis** of a folder containing multiple FRAP CZI files
- Automatic extraction of:
  - frame interval
  - pixel size (if available from metadata)
  - ROI geometry
- Automatic identification of:
  - bleach frame
  - control ROI
  - FRAP ROIs
- Embedded display of:
  - raw ROI intensities
  - fitted recovery curves
  - bleach-corrected fit
  - residuals
- Export of:
  - raw data Excel file
  - summary Excel file
  - SVG overview figure

---

## Input requirements

The FRAP tab expects a **CZI file** containing:

- time-resolved image data
- circular ROI annotations in metadata
- at least **2 ROIs**
  - one ROI is treated as the **control**
  - the remaining ROIs are treated as **FRAP ROIs**

If fewer than 2 valid ROIs are found, the analysis fails.

---

## Single-file analysis

1. Open the **FRAP** tab.
2. Under **Single CZI**, click **Browse** and select a CZI file.
3. Set the analysis parameters:
   - **Pattern** (used only in batch mode)
   - **Imaging bleach correction**
   - **Fallback pixel size (µm)**, optional
   - **Initial D (px²/frame)**
   - **D lower bound**
   - **D upper bound**
4. Click **Run FRAP**.

The analysis result is displayed inside the GUI and exported next to the input file.

---

## Batch analysis

1. Under **Batch folder**, click **Browse** and select a root folder.
2. Set the filename pattern, typically:

   ```text
   *FRAP*.czi
   ```

3. Set analysis parameters as above.
4. Click **Run FRAP**.

The software recursively searches the selected folder for matching CZI files and analyzes each one.

During batch analysis:
- the progress bar is updated
- the embedded display can update to the most recently finished file
- all results are written next to the individual files

---

## Model

The implemented FRAP model is based on **Soumpasis (1983)** for circular bleach regions.

### Recovery term

The recovery function uses:

- a circular bleach radius `R`
- a diffusion coefficient `D`
- the Soumpasis expression involving modified Bessel functions

### With imaging bleach correction
If **Imaging bleach correction** is enabled, the model includes an additional exponential decay term for acquisition-related bleaching.

This version fits:

- `x_0` — bleach frame position
- `R` — ROI radius
- `F_0` — pre-bleach signal
- `f_bl` — bleached fraction
- `f_mob` — mobile fraction
- `D` — diffusion coefficient
- `t_b` — imaging bleach decay constant

### Without imaging bleach correction
If imaging bleach correction is disabled, the exponential decay term is omitted.

This version fits:

- `x_0`
- `R`
- `F_0`
- `f_bl`
- `f_mob`
- `D`

---

## Control normalisation

The control ROI is used to correct for global intensity loss unrelated to recovery.

For each frame:

1. The control trace is normalised by its pre-bleach mean.
2. Each FRAP trace is divided by the normalised control trace.
3. This corrects for illumination drift and global photobleaching.

This control-normalised trace is then used for fitting.

---

## Automatic detection steps

### Bleach frame
The bleach frame is detected automatically from the largest intensity drop across all ROI traces.

### Control ROI
The ROI with the **smallest bleach drop** is chosen as the control ROI.

### FRAP ROIs
All remaining ROIs are treated as FRAP ROIs.

---

## Display panels

The embedded Matplotlib display contains four panels.

### 1. Raw intensity (pre-normalisation)
Shows:
- control ROI trace
- FRAP ROI traces
- bleach time marker

### 2. Normalised fit
Shows:
- control-normalised FRAP traces
- fitted recovery curves
- optional imaging bleach baseline term

### 3. Bleach-corrected fit
Shows:
- bleach-corrected recovery
- fitted mobile fraction curve
- estimated half-time markers

### 4. Fit residuals
Shows:
- residual trends after bleaching
- a smoothed residual trace per FRAP ROI

---

## Parameters in the GUI

### Pattern
Filename pattern used in batch mode, e.g.

```text
*FRAP*.czi
```

### Imaging bleach correction
Enable or disable the imaging-bleach model term.

Recommended: **enabled** for most acquisitions.

### Fallback pixel size (µm)
Used only if the CZI metadata does not contain pixel size information.

If left blank and metadata is missing:
- diffusion is still fitted in `px²/frame`
- diffusion in `µm²/s` cannot be reported

### Initial D (px²/frame)
Initial guess for diffusion coefficient.

### D lower bound / D upper bound
Bounds for the fitted diffusion coefficient in `px²/frame`.

---

## Output files

For an input file such as:

```text
sample1_FRAP.czi
```

the software writes:

### Per file
- `sample1_FRAP_raw_data.xlsx`
- `sample1_FRAP_summary.xlsx`
- `sample1_FRAP_overview.svg`

### Description

#### `*_FRAP_raw_data.xlsx`
Contains:
- frame index
- time
- raw ROI intensities
- control-normalised traces
- fitted model traces for FRAP ROIs

#### `*_FRAP_summary.xlsx`
Contains:
- ROI geometry
- fitted parameters
- mobile fraction
- immobile fraction
- diffusion coefficient in `px²/frame`
- diffusion coefficient in `µm²/s` (if pixel size known)
- recovery time constant
- half-time
- optional imaging bleach time constant
- warnings

#### `*_FRAP_overview.svg`
Vector graphic export of the embedded FRAP display figure.

---

## Batch summary

The FRAP analysis code also supports condition-level aggregation from summary Excel files.

This is based on:
- condition folders
- replicate subfolders
- `*_FRAP_summary.xlsx` files inside replicates

From these, the analysis can compute:
- mean half-time
- mean mobile fraction
- mean diffusion coefficient
- outlier filtering using MAD-based robust z-scores

If implemented in the GUI workflow, this can be extended to generate:
- `FRAP_condition_summary.png`
or a GUI-based figure export.

---

## Fit outputs and interpretation

### Mobile fraction
`f_mob` is the estimated fraction of the bleached population that is mobile and able to recover.

### Immobile fraction
Computed as:

```text
1 - f_mob
```

### Diffusion coefficient
Reported as:
- `D (px²/frame)`
- `D (µm²/s)` if pixel size is available

### Half-time
Recovery half-time:

```text
t½ = τ · ln(2)
```

where `τ` is estimated from the fitted diffusion parameters.

---

## Warnings

The fitter may mark results as potentially unreliable, for example if:

- `D` is at the lower fit bound
- the estimated half-time is much longer than the acquisition duration

These warnings are written to the summary Excel file.

---

## Troubleshooting

### No ROIs found
The CZI file must contain circular ROI annotations in metadata.

### Less than 2 ROIs
At least one control ROI and one FRAP ROI are required.

### Pixel size missing
If pixel size cannot be read from metadata and no fallback is provided:
- fitting still runs
- `D (µm²/s)` is reported as unavailable

### Poor fit quality
Try:
- enabling imaging bleach correction
- adjusting the initial diffusion guess
- widening or tightening D bounds
- checking whether the bleach ROI and control ROI are correctly annotated

### Batch folder finds no files
Check the filename pattern, for example:

```text
*FRAP*.czi
```

and ensure the selected folder contains matching files in subdirectories if recursive search is expected.