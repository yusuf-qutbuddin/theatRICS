# GUI overview

The theatRICS graphical interface is organized into several analysis tabs. Each tab is focused on a specific workflow and shares a common logging, progress, and export system.

---

## General interface features

Across the application, the GUI provides:

- **Tabbed workflows** for different analysis tasks
- A **status bar** showing the current task state
- A **progress bar** for long-running computations
- A **Results & Logs** tab for textual output and session tracking
- A **Cancel Running Task** button for stopping active processing jobs
- A **Restart** button for restarting the application
- A **Toggle Dark Mode** button for changing the visual theme

Long-running calculations are executed in background worker processes so that the interface can remain responsive.

---

## Image Simulation

The **Image Simulation** tab generates synthetic image stacks for testing and benchmarking RICS-style analyses.

### Main inputs
- **Image shape (pixels)**  
  width and height of the simulated frames
- **Number of cores**  
  number of CPU workers used for simulation
- **Number of frames**
- **Pixel dwell time (µs)**
- **Pixel size (nm)**
- **Brightness (kHz)**
- **Number of particles**
- **Diffusion coefficients in X and Y**
- **Rotation**
- **Background**
- **PSF sigma (pixels)**
- **Simulation type**  
  one of:
  - `isotropic`
  - `anisotropic`
  - `anisotropic_rotated`
- **Output path**

### Actions
- **Run Simulation**
- **Load Existing**

### Display
The simulation display shows:
- first frame
- last frame
- time-averaged image
- average intensity trace over time from a centered crop

### Typical use
This tab is useful for:
- validating fitting procedures
- testing anisotropic diffusion scenarios
- generating known-reference data

---

## RICS Export

The **RICS Export** tab computes correlation maps and uncertainty maps from image stacks.

### Main inputs
- **Input file**  
  a single image stack
- **Input folder**  
  used for batch analysis
- **Channel to use**
- **Crop factor**
- **Window size (odd)**
- **Correct drift**

### Actions
- **Export RICS**
- **Load RICS**

### Display
The RICS export display can show:
- raw image preview
- drift-corrected image preview
- RICS correlation map
- uncertainty map
- 3D surface view of the RICS map

### Outputs
Typical outputs include:
- correlation TIFF
- uncertainty TIFF
- optionally corrected intermediate data

### Typical use
This is the first analysis step when starting from microscopy image stacks.

---

## RICS Fitting

The **RICS Fitting** tab fits diffusion models to exported RICS maps.

### Main inputs
- **RICS map file**
- **Input folder** for batch fitting
- **Results file**
- **Pixel size (nm)**
- **Pixel dwell (µs)**
- **Line time (ms)**
- **PSF size XY (µm)**
- **PSF aspect ratio**
- **Crop factor fast**
- **Crop factor slow**
- **Diffusion model**
- **Channel**
- **Optional 1D Fast Axis Fit**

### Diffusion map subsection
The same tab also contains a **Diffusion Map Fitting Parameters** section with:
- **Input file for diffusion map**
- **Window size**
- **Offset**
- **Channel**

### Actions
- **Run 2D/3D Fitting**
- **Generate Diffusion Map**

### Display
The fitting display can show:
- 3D RICS data surface
- fitted model surface
- residual surface
- fast-axis cross section
- slow-axis cross section
- optional 1D fit residuals
- optional diffusion map / brightness map / number map

### Outputs
Typical outputs include:
- fit summary CSV
- NPZ fit arrays
- exported SVG plots

### Typical use
This tab is used after RICS export to estimate diffusion coefficients and model quality.

---

## SFCS

The **SFCS** tab performs perpendicular scanning FCS analysis.

### Main inputs
- **Input file**
- **Channel**
- **Bleach correction**
- **Number of cores**

### Actions
- **Correlate**

### Display
The SFCS display can show:
- original frame crop
- aligned frame crop
- intensity trace
- bleach-corrected intensity trace
- autocorrelation curve `G(τ)`
- uncertainty band, if available

### Outputs
Typical outputs include:
- correlation curves
- exported SVG figures
- logged summary values

### Typical use
Use this tab when analyzing perpendicular scanning FCS line scans rather than image-based RICS data.

---

## FCS Fitting

The **FCS Fitting** tab fits correlation curves exported as CSV files using a selection of FCS diffusion and blinking models.

### Supported modes
- **Single-file fitting**
- **Recursive batch fitting** over all matching files in subfolders of a selected folder

### Main inputs
- **Single CSV**
- **Batch folder**
- **Batch file pattern**  
  for example:
  - `*_xy_intensity_trace_correlation.csv`
- **Model**
- **Tau min (s)**
- **Tau max (s)**
- **PSF radius (µm)**
- **PSF aspect ratio**
- **Experiment T (°C)**

### Calibration-only inputs
For calibration models (model names containing `Cal`), the GUI additionally shows:
- **Given D (µm²/s)**
- **Given D temp (°C)**

These fields are hidden for non-calibration models.

### Model-dependent initial parameters
The tab contains a dynamic **Initial parameters** editor:
- only parameters relevant to the selected model are shown
- default values are taken from the legacy fitting code
- user-modified values are preserved while switching models

Examples of model-dependent parameters include:
- `N`
- `tau diffusion`
- `delta`
- `F_Blink`
- `offset`
- `f1`
- `rho_D`
- `rho_B`
- `Gamma`
- `Alpha`
- `G0`
- `tau characteristic decay`
- MEMFCS parameters

### Actions
- **Run Fit**

### Display
The FCS fit display shows:
1. observed correlation and fitted model
2. weighted residuals
3. iMSD or a log-log correlation representation depending on model
4. histogram of residual values

### Outputs
Per file, the GUI exports:
- SVG figure
- fitted correlation CSV
- iMSD CSV where applicable

For batch mode:
- a single summary CSV is written in the outer selected folder

### Typical use
This tab is intended for fitting precomputed correlation curves rather than directly processing images.

---

## FRAP

The **FRAP** tab analyzes fluorescence recovery after photobleaching from CZI image files containing circular ROI annotations.

### Supported modes
- **Single-file analysis**
- **Batch analysis** over matching FRAP files in subfolders

### Main inputs
- **Single CZI**
- **Batch folder**
- **Pattern**  
  for example:
  - `*FRAP*.czi`
- **Imaging bleach correction**
- **Fallback pixel size (µm)**
- **Initial D (px²/frame)**
- **D lower bound**
- **D upper bound**

### Automatic processing steps
The FRAP workflow automatically:
- reads image frames from the CZI file
- extracts frame interval from metadata
- extracts pixel size from metadata when available
- reads circular ROI annotations
- detects the bleach frame
- identifies the control ROI
- normalizes FRAP traces using the control ROI
- fits each FRAP ROI using the selected FRAP model

### Display
The FRAP display shows:
1. raw intensity traces before normalisation
2. control-normalised traces with fitted recovery curves
3. bleach-corrected fit representation
4. residual traces

### Outputs
For each file, the software writes:
- `*_FRAP_raw_data.xlsx`
- `*_FRAP_summary.xlsx`
- `*_FRAP_overview.svg`

### Typical use
This tab is used for ROI-based FRAP analysis from annotated CZI time series.

---

## Vesicle Finder

The **Vesicle Finder** tab detects vesicles (such as GUVs) in CZI images or time series, allows interactive selection, and exports either square crops or unrolled membrane strips.

### Supported modes
- **Single-frame detection**: detect vesicles in frame 0 and display interactively
- **Batch cropping**: export selected vesicles across all selected frames
- **Membrane straightening**: unroll the membrane ring into a flat strip for intensity analysis

### Detection methods
- `hough`: Hough circle transform, best for fluorescence membrane images
- `hough_transmitted`: gradient-based Hough, best for transmitted-light images
- `weighted_intensity`: peripheral intensity maximization (improved and modified from Kohyama et al. 2022), robust for both image types
- `cellpose`: deep learning segmentation with optional shape filtering and circle fitting
- `otsu`: simple threshold-based fallback

### All spatial parameters in µm
Pixel size is read automatically from CZI metadata. All radius, distance, and margin parameters are entered in µm and converted to pixels internally.

### Display
- Left panel: raw image with detected vesicle overlays (circles or contours)
- Right panel: segmentation label map

### Membrane straightening
For each detected vesicle with a known radius, the software can unroll the annular membrane region into a flat strip. The output shows:
- the straightened membrane image (frame 0)
- a heatmap of intensity vs position along membrane vs time
- total membrane intensity vs time

### Debug images
Enabling **Save debug images** writes intermediate processing images to a debug folder alongside the CZI file. This is useful for diagnosing detection failures.

### Outputs
- Square crop TIFFs per selected vesicle (all frames)
- Straightened membrane TIFFs per vesicle
- Intensity profile CSVs per vesicle
- Total intensity CSV per vesicle
- Overview SVG figure

---

## Results & Logs

The **Results & Logs** tab collects textual output from all workflows.

### Features
- scrolling log view
- **Clear Log**
- **Save Results**
- **Export All Plots**
- **Save Session**
- **Load Session**

### Typical use
Use this tab to:
- review processing history
- inspect warnings and failures
- save text output
- store and restore GUI parameter states

---

## Notes on responsiveness and background processing

Most heavy computations in the GUI are executed in separate worker processes. This design allows:
- progress tracking
- cancellation support
- batch processing
- reduced UI blocking during long-running fits or exports

However, very frequent figure redraws or per-file updates in large batch jobs can still reduce responsiveness. For large batches, the software may update only logs and progress during processing and refresh the full display at the end.

---

## Typical analysis routes

### Simulate → export → fit
1. Generate synthetic data in **Image Simulation**
2. Compute correlation maps in **RICS Export**
3. Estimate diffusion parameters in **RICS Fitting**

### Image-based experiment
1. Export RICS from microscopy data
2. Fit RICS maps
3. Generate diffusion maps if required

### Scan-correlation experiment
1. Correlate line-scan data in **SFCS**
2. Optionally fit exported FCS-style curves in **FCS Fitting**

### Recovery experiment
1. Analyze photobleaching time series in **FRAP**
2. Inspect mobile fraction, half-time, and diffusion outputs