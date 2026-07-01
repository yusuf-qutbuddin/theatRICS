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

## Workflow 6: ICS analysis

Use this workflow to track changes in molecular correlation over time using
temporal Image Correlation Spectroscopy on TIFF image stacks.

### Single-file analysis
1. Open the **ICS** tab.
2. Select a **Single TIFF** file.
3. Set:
   - block length
   - frame skip
   - threshold multiplier
4. Optionally enable **Save block images**.
5. Click **Run ICS**.

### Batch analysis
1. Select a **Batch folder** containing sample subfolders.
2. Set a **File pattern**, for example:

   ```text
   *.tiff
   ```

3. Set analysis parameters as above.
4. Click **Run ICS**.

The software processes each subfolder as one sample and writes a combined
summary CSV and plot in the parent folder.

### Outputs
Per file:
- block statistics CSV
- overview SVG figure
- optional block diagnostic TIFFs

Batch:
- combined CSV across all samples
- combined overview SVG plot

---

## Workflow 7: AFM condensate analysis

Use this workflow to measure heights and wetting angles of condensates or
particles sitting on a membrane surface from JPK QI AFM images.

### Steps
1. Open the **AFM** tab.
2. Click **Browse** and select a `.jpk-qi-image` file.
3. Choose the **Channel** (typically `height_trace`).
4. Click **Load File**.
5. Click on the AFM image:
   - first click: start point on the bare membrane
   - second click: end point on the bare membrane on the other side
   - the line must cross the object
6. Inspect the profile panel:
   - adjust the **Fit points** slider to optimise the wetting angle
   - adjust **Threshold fraction** if contact points are placed incorrectly
7. Repeat for additional objects.
8. Click **Save results**.

### Outputs
- per-profile CSV with full height arrays
- summary CSV with all scalar measurements
- SVG figure export

---


## Workflow 8: Vesicle / GUV detection and membrane analysis

Use this workflow to detect GUVs in CZI microscopy data and analyze their membranes.

### Detection
1. Open the **Vesicle Finder** tab.
2. Select a CZI file.
3. Choose the detection channel.
4. Select the detection method:
   - **hough**: for fluorescence membrane images (bright ring)
   - **hough_transmitted**: for transmitted-light images
   - **weighted_intensity**: for both image types, improved and modified from Kohyama et al. 2022
   - **cellpose**: for filled or ring-shaped objects with deep learning
5. Set radius range in µm.
6. Click **Detect Vesicles**.
7. Click on detected vesicles to select them (selected = green, unselected = cyan).

### Cropping
Click **Crop Selected** or **Export All** to export square crops for each selected vesicle across all frames.

### Membrane straightening
1. After detection, set the membrane thickness in µm.
2. Choose the intensity channel (can differ from the detection channel).
3. Click **Straighten Selected** or **Straighten All**.
4. The display shows the unrolled membrane strip, intensity heatmap, and total intensity trace.

### Troubleshooting detection
- Enable **Save debug images** to inspect intermediate processing steps.
- Check `08_distance_smooth.tif` to verify that one bright peak per GUV is visible.
- If only one vesicle is detected instead of several, try reducing the min radius or adjusting the threshold method.
- If detection is slow, reduce the search range parameter.
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