# Troubleshooting

This page collects common issues that may occur when using theatRICS locally or when building the documentation.

---

## Installation and startup

### The command `theatrics` is not found
Make sure the package was installed successfully:

```bash
pip install theatrics
```

If you installed from source, use:

```bash
pip install -e .
```

Then run:

```bash
theatrics
```

### GUI does not start
Possible reasons include:
- missing Tkinter support
- a broken Python environment
- missing dependencies

On Linux, Tkinter may require a system package such as:

```bash
python3-tk
```

### Import errors for scientific packages
Check that required Python packages are installed, especially:
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `lmfit`
- `scikit-image`
- `statsmodels`
- `tifffile`
- `pylibCZIrw`

---

## RICS export and fitting

### RICS export fails on input file
Check:
- file format is supported (`.czi`, `.tif`)
- channel index is valid
- file is not corrupted

### RICS map loads but fitting fails
Common reasons:
- incorrect microscope parameters
- incorrect PSF size or aspect ratio
- unsuitable crop factors
- invalid diffusion model choice

### Diffusion coefficient looks unrealistic
Check:
- pixel size
- dwell time
- line time
- PSF calibration
- whether the chosen model matches the experiment

### Batch RICS processing stops unexpectedly
Possible reasons:
- one corrupted file in the folder
- worker process crash
- too many files for available memory

### PTU file will not load
- install tttrlib: `pip install tttrlib`
- verify the file is a valid PicoQuant PTU CLSM image
- check that the requested channel index exists in the file;
  available channels are logged automatically when a PTU file is
  selected in the GUI

### PTU metadata not found
Some PTU files may not store pixel size or dwell time in the standard
header tags. If metadata cannot be read:
- the pixel size, dwell time, and line time fields in the Fitting tab
  will not be auto-populated
- enter the values manually in the RICS Fitting parameter fields

Check the **Results & Logs** tab for the traceback.

---

## SFCS

### Correlation fails or returns empty output
Check:
- selected channel exists
- input file contains the expected scan data
- bleach correction setting is appropriate
- number of CPU cores is not too large

### SFCS figure is not saved
Make sure:
- the input file path is valid
- the output directory is writable
- the analysis completed successfully

---

## FCS fitting

### Batch mode finds no files
Check the **Batch file pattern**.

For example, if your files are named like:

```text
sample1_xy_intensity_trace_correlation.csv
```

then a suitable pattern is:

```text
*_xy_intensity_trace_correlation.csv
```

If the pattern is too broad, unrelated CSV files may also be picked up.

### FCS single-file fit fails
Check:
- the selected CSV really contains correlation data
- the expected columns are present
- the tau range is valid
- the model matches the data

### CSV format is incompatible
The legacy fitter expects a CSV structure where:
- column 0 = `tau`
- column 1 = `G`
- column 2 = count rate (read from the file)
- column 3 = `sigma_G`

If your export format differs, the fitter will need column-index adjustments.

### Calibration parameters are missing
The **Given D** and **Given D temp** fields are only shown for models containing `Cal`.

If you selected a non-calibration model, this is expected behavior.

### Blinking parameter naming confusion
Some legacy fitting code may use:
- `F_Blink`
- `F_B`

The GUI provides compatibility by mapping both names internally for blinking models.

### Offset parameter is missing
The offset parameter is shown only for models whose names contain `Offset`.

### Batch fitting makes the GUI unresponsive
This usually happens when the GUI redraws the full figure for every finished file.

Recommended behavior:
- update only progress and logs during the batch
- redraw the fit display only for the final file or at reduced frequency

### Cancel does not stop the current file immediately
In batch fitting, cancellation typically takes effect **between files**.

If one file fit is already running inside `curve_fit` or `lmfit`, it may finish before the batch stops.

This is normal behavior unless explicit cancellation checks are added inside the fit routine itself.

### Batch summary CSV appears in the wrong place
In recursive batch mode, the summary CSV is written once in the **outer selected folder** (typically inside a `Results/` subfolder), while per-file outputs remain next to each individual input file.

---

## FRAP analysis

### No ROIs found
The FRAP workflow requires circular ROI annotations in the CZI metadata.

If no suitable circles are found, the file cannot be analyzed.

### Less than 2 ROIs detected
At least:
- one control ROI
- one FRAP ROI

are required.

### Pixel size is missing
If pixel size is not present in metadata and no fallback is provided:
- FRAP fitting still runs
- diffusion in `µm²/s` cannot be reported
- diffusion in `px²/frame` is still available

### FRAP fit fails for one ROI
This can happen if:
- the ROI is too noisy
- bleaching is weak
- the recovery is incomplete
- the diffusion bounds are too restrictive

Try:
- changing the initial diffusion guess
- widening D bounds
- enabling or disabling imaging bleach correction

### Bleach frame looks wrong
The bleach frame is detected automatically from the largest intensity drop.

This can fail if:
- bleaching was weak
- another ROI fluctuates strongly
- the acquisition contains motion or artifacts

If needed, the detection logic would need manual override support.

### Wrong control ROI is selected
The control ROI is chosen as the ROI with the smallest bleach drop.

This is usually sensible, but may fail if:
- the control ROI also experiences bleaching
- one FRAP ROI was only weakly bleached
- the ROI annotation set is inconsistent

### FRAP batch finds no files
Check the pattern, for example:

```text
*FRAP*.czi
```

and ensure the selected folder contains matching files.

### FRAP condition summary is missing
Condition-level summaries require a folder structure where:
- condition folders contain replicate folders
- replicate folders contain `*_FRAP_summary.xlsx`

If that structure is missing, condition summary generation may be skipped.

---

## Vesicle Finder

### No vesicles detected
- Try a different threshold method (`mean` or `triangle` recommended for membrane images)
- Check that the min/max radius range covers the actual vesicle sizes
- Enable debug images to see which processing step fails
- For Hough: lower the threshold fraction

### Only one vesicle detected instead of several
- Check `08_distance_smooth.tif` — it should show one bright peak per GUV
- If peaks are merged, the GUVs may be too close together
- Try increasing the min distance parameter

### Vesicle detection is slow
- Reduce the search range
- Install OpenCV for faster Hough detection: `pip install opencv-python`
- The weighted intensity method uses a focused vectorized search and should be fast for typical numbers of GUVs

### Cellpose finds an arc instead of a full circle
- Enable **Fit circles to detected masks** in the Cellpose parameters
- This fits a geometric circle using least-squares and reconstructs the full GUV

### Membrane straightening not available
- Straightening requires vesicles with a known radius
- Use Hough, weighted intensity, or Cellpose with fit circles enabled
- Otsu detections do not produce radius estimates

### Debug images are not saved
- Ensure the **Save debug images** checkbox is checked before clicking Detect Vesicles
- Check that the output directory next to the CZI file is writable

### Pixel size missing
- Enter a fallback pixel size in the Fallback pixel size field
- Without pixel size, µm parameters cannot be converted to pixels and detection will fail

---

## ICS

### All G values are NaN
This usually means the mask is empty — no pixels passed the threshold.

Try:
- reducing the **Threshold multiplier**
- checking that the correct TIFF file is selected
- verifying that the stack contains temporal intensity variation

### Stack is too short
The stack must contain at least `block_length` frames.

Reduce the block length or use a longer acquisition.

### Batch finds no files
Check the **File pattern** matches your file extension, for example:

```text
*.tiff
```

or for `.tif` files:

```text
*.tif
```

Also ensure the selected parent folder contains sample subfolders with
matching files inside them.

### Block images are not saved
Ensure **Save block images** is checked before running the analysis and
that the output directory is writable.

### Combined batch CSV is missing
This happens when no sample subfolders contain valid matching TIFF files,
or when all files fail during processing.

Check the **Results & Logs** tab for error messages.

---

## AFM

### "AFMReader is not installed"
Install the required package:

```bash
pip install AFMReader
```

### File will not load
Check:
- the file is a valid JPK QI image
- the selected channel exists in the file
- the file is not corrupted

### Height image looks inverted or all-black
Try a different channel, for example `height_retrace` instead of
`height_trace`.

### No contact points found
This can happen if:
- the profile does not cross the object
- the object is too flat relative to noise
- the threshold fraction is too high

Try:
- redrawing the profile so it crosses the full width of the object
- decreasing the **Threshold fraction**

### Wetting angle looks wrong
Check that:
- both click points are on the flat membrane, not on the object
- the **Fit points** slider is not set too high (which would include
  flat membrane in the slope fit)
- the profile crosses the object centrally

### Profile is very noisy
Try:
- increasing the **Smooth window** parameter
- increasing the **Profile points** for finer spatial sampling

### Save results fails
Check that:
- a save folder has been selected
- the folder is writable
- at least one profile has been drawn before clicking Save results

---


## GUI responsiveness and background tasks

### Progress bar moves but the GUI feels frozen
This often happens when the main thread performs expensive plotting or export operations too frequently.

Examples:
- redrawing a Matplotlib figure after every file in a large batch
- writing large outputs synchronously for every queue event

Recommended strategy:
- keep fitting in worker processes
- throttle display updates
- update the final display only after the batch finishes

### Cancel button does not appear to work
Check:
- the relevant worker process and cancel event are included in the GUI cleanup/cancel logic
- the current task is not inside a long non-interruptible fit call

If a worker only checks cancellation between files, cancel will stop the batch after the current file finishes.

### Worker terminated unexpectedly
This usually means the background process crashed.

Look at the traceback in the **Results & Logs** tab. Common reasons include:
- invalid input data
- file format mismatch
- missing metadata
- failed fit convergence
- unhandled exceptions inside worker code

---

## Session and export issues

### Session file will not load
The JSON session file may be:
- incomplete
- from an older version of the GUI
- manually edited and no longer valid

### Plot export fails
Check:
- destination directory exists
- directory is writable
- the relevant figure has actually been created

### Log export is empty
Make sure the log contains output before saving. The **Results & Logs** tab only exports what is currently present in the log widget.

---

## If problems continue

When reporting a bug, it is helpful to include:
- operating system
- Python version
- theatRICS version
- the exact error traceback
- a short description of the input file type
- whether the problem occurs in single-file mode or batch mode

For file-format-specific issues, it is especially useful to mention:
- file extension
- whether metadata is present
- whether the problem happens for one file or all files