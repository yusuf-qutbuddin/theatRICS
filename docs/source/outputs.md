# Outputs

theatRICS writes analysis results to disk so that intermediate and final outputs can be reused outside the GUI and across sessions.

Output types depend on the workflow, but commonly include:

- **TIFF** for image-based correlation outputs
- **CSV** for fit curves and summary tables
- **NPZ** for stored model arrays and auxiliary numerical outputs
- **SVG** for publication-quality figures
- **Excel (`.xlsx`)** for FRAP raw data and summaries
- **TXT / JSON** for logs and saved GUI sessions

---

## General output behavior

### Per-file outputs
Most analysis tabs write outputs **next to the input file** or into a nearby `Results/` folder.

### Batch outputs
For batch workflows:
- some per-file outputs remain next to each file
- summary tables may be written once in the **outer selected folder**

### Plot exports
Displayed figures can be exported directly from the GUI, typically in:
- `SVG`
- `PNG` (depending on workflow and settings)

---

## Image Simulation outputs

The **Image Simulation** tab writes simulated image stacks as TIFF files.

### Typical output
- `simulation_output.tif`

### Contents
The TIFF contains the generated frame stack.

### Display-only outputs
The GUI also shows:
- first frame
- last frame
- average image
- intensity trace

These are visual diagnostics and are not necessarily written automatically unless exported through the GUI.

---

## RICS Export outputs

The **RICS Export** tab generates correlation outputs from microscopy image stacks.

### Typical outputs
- RICS correlation map TIFF
- uncertainty map TIFF
- optionally corrected image stack TIFF
- optionally intermediate TIFF files

### Common naming conventions
Depending on your worker implementation, file names may include patterns such as:
- `*_RICScorr.tif`
- `*_RICSunc.tif`

### Contents
- `RICScorr` — correlation map
- `RICSunc` — uncertainty / standard deviation map

---

## RICS Fitting outputs

The **RICS Fitting** tab produces fitted model results from exported RICS maps.

### Typical outputs
- fit summary CSV
- NPZ file containing arrays used in fitting
- SVG figure of the fit
- optional diffusion map outputs

### Common contents of the summary CSV
- diffusion coefficient
- model type
- offset
- file path
- optional 1D fit results

### Figure outputs
The GUI fit display can be saved as:
- `*.svg`

For example:
- `my_rics_map_fit.svg`

### NPZ outputs
These can include arrays such as:
- RICS map
- fitted model
- residuals
- optional 1D model and residual arrays

---

## Diffusion map outputs

Diffusion map generation writes spatially resolved fitting outputs.

### Typical outputs
- diffusion map
- number map
- brightness map
- auxiliary arrays stored in NPZ or similar formats

### Displayed outputs
The GUI may show:
- diffusion map
- brightness map
- number map
- representative 1D fits

---

## SFCS outputs

The **SFCS** tab computes scanning FCS correlations.

### Typical outputs
- correlation curves
- uncertainty estimates
- exported plot files

### Common plot outputs
For example:
- `*_correlation.svg`

### Contents
Plots typically contain:
- `G(τ)` curve
- optional uncertainty band
- intensity traces
- aligned/original previews

---

## FCS Fitting outputs

The **FCS Fitting** tab writes outputs for each fitted correlation CSV.

### Per-file outputs
For an input base name `mycurve` and model `g2diffSFCS`, the software typically writes:

- `Results/mycurve_g2diffSFCS.svg`
- `Results/mycurve_g2diffSFCS.csv`
- `Results/mycurve_g2diffSFCS_iMSD.csv` (if applicable)

### Description

#### `Results/<file>_<model>.svg`
A publication-quality vector export of the displayed FCS fit figure.

#### `Results/<file>_<model>.csv`
A table containing:
- `tau`
- `G`
- `sigma G`
- fitted correlation curve (`cc Fit`)

#### `Results/<file>_<model>_iMSD.csv`
Written only for models where iMSD is meaningful.

Contains:
- `tau`
- `iMSD`

### Batch summary output
In recursive batch mode, the software writes one summary CSV in the **outer selected folder**, typically:

- `Results/<model>_fit_summary.csv`

This summary contains one row per fitted file.

### Typical summary columns
Depending on model:
- diffusion coefficient(s)
- particle number / amplitude parameters
- blinking parameters
- offset
- characteristic times
- BIC
- goodness-of-fit statistics
- original filename

---

## FRAP outputs

The **FRAP** tab writes outputs next to each analyzed CZI file.

For an input file such as:

```text
sample1_FRAP.czi
```

the software typically writes:

- `sample1_FRAP_raw_data.xlsx`
- `sample1_FRAP_summary.xlsx`
- `sample1_FRAP_overview.svg`

### `*_FRAP_raw_data.xlsx`
Contains:
- frame index
- time
- raw ROI intensities
- control-normalised traces
- model fit values for FRAP ROIs

### `*_FRAP_summary.xlsx`
Contains:
- ROI metadata
- fitted parameters
- mobile fraction
- immobile fraction
- diffusion coefficient in `px²/frame`
- diffusion coefficient in `µm²/s` (if pixel size is available)
- recovery time constant
- half-time
- optional imaging bleach decay constant
- warnings

### `*_FRAP_overview.svg`
A vector export of the FRAP display figure shown in the GUI.

### Optional batch-level outputs
If a condition-summary workflow is used, additional condition-level outputs may be produced, such as:
- `FRAP_condition_summary.png`

depending on how batch aggregation is configured.

---

## Logs and session files

The **Results & Logs** tab allows saving auxiliary outputs.

### Log export
- `*.txt`

Contains the textual log shown in the GUI.

### Session export
- `*.json`

Contains saved GUI parameters and session state.

### Plot export
The GUI can export currently available figures, typically as:
- `PNG`
- `SVG`

depending on workflow and implementation.

---

## Vesicle Finder outputs

### Crop outputs

For a CZI file named `sample.czi`, square crop outputs are written to:

```text
sample_vesicle_crops/
    vesicle_1.tif
    vesicle_3.tif
```

Each TIFF contains all selected frames stacked along axis 0 with shape `(n_frames, crop_size, crop_size)`.

### Straightening outputs

For a CZI file named `sample.czi`, straightening outputs are written to:

```text
sample_straightened/
    vesicle_1_straightened.tif
    vesicle_1_intensity_profile.csv
    vesicle_1_total_intensity.csv
    straighten_overview.svg
```

| File | Contents |
|------|----------|
| `*_straightened.tif` | 3D TIFF: `(n_frames, thickness_px, n_angle_points)` — the unrolled membrane |
| `*_intensity_profile.csv` | Mean intensity across thickness for each angle and frame |
| `*_total_intensity.csv` | Total membrane intensity per frame |
| `straighten_overview.svg` | Combined display figure |

### Debug outputs

When debug saving is enabled, intermediate images are saved to:

```text
sample_debug/
    01_raw_normalized.tif
    04_binary_after_threshold.tif
    05_binary_dilated.tif
    06_interior.tif
    07_distance_interior.tif
    08_distance_smooth.tif
```
---

## Notes on file placement

### Local `Results/` folders
Several workflows write outputs into a `Results/` folder next to the input file.

### Outer-folder summaries
Recursive batch processing may place summary outputs in the outer selected folder instead of repeating them in every subfolder.

This is especially useful for:
- FCS fitting batch mode
- condition-level FRAP aggregation

---

## Reproducibility

Because theatRICS writes numerical outputs in standard formats such as CSV, TIFF, NPZ, and Excel, the results can be:
- reloaded into Python or MATLAB
- inspected manually
- included in downstream analysis pipelines
- used to regenerate publication figures