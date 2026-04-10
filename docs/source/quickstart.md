# Quickstart

This quickstart provides a practical overview of the main workflows available in theatRICS.

Depending on your experiment, you may use one or more of the following:
- **RICS Export**
- **RICS Fitting**
- **SFCS**
- **FCS Fitting**
- **FRAP**
- **Image Simulation**

---

## Start the program

After installation, launch the GUI with:

```bash
theatrics
```

---

## Workflow 1: RICS analysis from image stacks

Use this workflow when starting from microscopy image stacks such as TIFF or CZI data.

### Step 1 — Export RICS maps
1. Open the **RICS Export** tab.
2. Select an **input file** or **input folder**.
3. Choose:
   - channel
   - crop factor
   - window size
   - optional drift correction
4. Click **Export RICS**.

This produces correlation maps and uncertainty maps.

### Step 2 — Fit the exported RICS map
1. Open the **RICS Fitting** tab.
2. Select the exported RICS map file.
3. Enter the microscope parameters:
   - pixel size
   - pixel dwell time
   - line time
   - PSF size
   - PSF aspect ratio
4. Choose a diffusion model.
5. Click **Run 2D/3D Fitting**.

You can optionally enable:
- **1D Fast Axis Fit**
- **Diffusion map generation**

### Outputs
Typical outputs include:
- RICS correlation TIFF
- uncertainty TIFF
- fit summary CSV
- NPZ fit arrays
- SVG fit figures

---

## Workflow 2: SFCS analysis

Use this workflow for perpendicular scanning FCS experiments.

### Steps
1. Open the **SFCS** tab.
2. Select an input file.
3. Choose the channel.
4. Optionally enable **Bleach Correction**.
5. Set the number of CPU cores.
6. Click **Correlate**.

### Display
The GUI will show:
- original data preview
- aligned data preview
- intensity trace
- correlation curve

### Outputs
Typical outputs include:
- correlation curves
- uncertainty estimates
- exported SVG plots

---

## Workflow 3: FCS fitting from exported CSV curves

Use this workflow when you already have correlation curves saved as CSV files.

### Single-file fitting
1. Open the **FCS Fitting** tab.
2. Select a **Single CSV** file.
3. Choose a fitting model.
4. Set:
   - tau range
   - PSF radius
   - PSF aspect ratio
   - experiment temperature
5. Adjust model-dependent initial parameters if needed.
6. Click **Run Fit**.

### Batch fitting
1. Select a **Batch folder**.
2. Set a **Batch file pattern**, for example:

   ```text
   *_xy_intensity_trace_correlation.csv
   ```

3. Choose the model and fit parameters.
4. Click **Run Fit**.

The software recursively searches matching files in subfolders.

### Calibration models
For models whose names contain `Cal`, the GUI also shows:
- **Given D**
- **Given D temperature**

These are hidden for non-calibration models.

### Outputs
Per file:
- SVG figure
- fitted curve CSV
- iMSD CSV where applicable

For batch mode:
- one summary CSV in the outer selected folder

---

## Workflow 4: FRAP analysis

Use this workflow for fluorescence recovery after photobleaching data stored in CZI files with circular ROI annotations.

### Single-file analysis
1. Open the **FRAP** tab.
2. Select a **Single CZI** file.
3. Set:
   - imaging bleach correction
   - optional fallback pixel size
   - initial diffusion estimate
   - diffusion bounds
4. Click **Run FRAP**.

### Batch analysis
1. Select a **Batch folder**.
2. Set a pattern such as:

   ```text
   *FRAP*.czi
   ```

3. Click **Run FRAP**.

The software recursively processes matching FRAP files.

### Automatic FRAP processing
The FRAP workflow automatically:
- identifies the bleach frame
- identifies the control ROI
- normalizes traces with the control ROI
- fits FRAP recovery curves

### Outputs
Per file:
- `*_FRAP_raw_data.xlsx`
- `*_FRAP_summary.xlsx`
- `*_FRAP_overview.svg`

---

## Workflow 5: Simulated data generation

Use this workflow to generate synthetic image stacks for testing and validation.

### Steps
1. Open the **Image Simulation** tab.
2. Set:
   - image size
   - number of frames
   - number of particles
   - diffusion coefficients
   - simulation type
   - output file path
3. Click **Run Simulation**.

### Outputs
The GUI shows:
- first and last frame
- average projection
- intensity trace

The simulation is also saved as a TIFF stack.

---

## Workflow 6: Diffusion map generation

Diffusion maps are generated from the **RICS Fitting** tab.

### Steps
1. Open **RICS Fitting**.
2. In the **Diffusion Map Fitting Parameters** section:
   - select an input file
   - choose channel
   - set window size
   - set offset
3. Click **Generate Diffusion Map**.

### Outputs
Typical outputs include:
- diffusion map
- brightness map
- number map
- auxiliary result arrays

---

## Monitoring progress

During long-running jobs, the GUI provides:
- a **status bar**
- a **progress bar**
- a **Cancel Running Task** button
- textual logs in the **Results & Logs** tab

For large batch jobs, the display may update only after the current file or at the end of the batch to keep the interface responsive.

---

## Saving and exporting

The **Results & Logs** tab allows you to:
- save log output
- export plots
- save a GUI session
- reload a previous session

Plots are usually exported in high-quality formats such as SVG, while numerical outputs are saved as CSV, NPZ, TIFF, or Excel files depending on the workflow.

---

## Suggested starting points

### If you work with image stacks
Start with:
- **RICS Export**
- then **RICS Fitting**

### If you work with scan correlation curves
Start with:
- **SFCS**
- then optionally **FCS Fitting**

### If you work with photobleaching recovery experiments
Start with:
- **FRAP**

### If you want to test the pipeline
Start with:
- **Image Simulation**