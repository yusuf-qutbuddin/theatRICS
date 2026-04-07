# Workflow: FCS fitting

This tab fits Fluorescence Correlation Spectroscopy (FCS) autocorrelation curves using a selection of diffusion/blinking models.

The GUI supports:

- **Single-file fitting**: choose one exported correlation CSV
- **Batch fitting**: choose a folder and fit all matching CSV files recursively
- **Model-dependent initial parameters**: the initial-parameter editor shows only parameters relevant to the selected model
- **Publication export**: the displayed fit is saved as **SVG**, and numerical outputs are saved as **CSV**

---

## Input data format

The fitter expects a correlation CSV file where the first columns are:

- `tau` (lag time, seconds)
- `G(tau)` (correlation)
- `count_rate` (scalar; typically stored in a fixed row/column in your export)
- `sigma_G` (uncertainty of `G(tau)`)

The code reads the CSV using `pandas.read_csv(..., header=None)` and then uses:

- `tau = col 0`
- `G = col 1`
- `count_rate = data.iloc[0, 2]`
- `sigma_G = col 3`

So your CSV must contain at least these columns.

---

## Single-file fitting

1. Open the **FCS Fitting** tab.
2. Under **Single CSV**, click **Browse** and select a correlation `*.csv`.
3. Choose a **Model** (see below).
4. Set fit limits:
   - **Tau min (s)**
   - **Tau max (s)**
5. Set optical parameters used by the model:
   - **PSF radius (µm)**
   - **PSF aspect ratio**
6. Set temperature information:
   - **Experiment T (°C)**

7. (Optional) Adjust **Initial parameters** shown for the selected model.
8. Click **Run Fit**.

The fit result is displayed inside the GUI (Matplotlib canvas) and exported to disk.

---

## Batch fitting

1. Under **Batch folder**, click **Browse** and select a folder.
2. The program searches recursively for matching CSV files (default pattern `*.csv` unless you restrict it in code).
3. Choose model and parameters as above.
4. Click **Run Fit**.

During batch fitting, the GUI can optionally update the display per finished file (depending on your worker/batch configuration).

---

## Models

The model list includes (examples):

- `g2diff`, `g2diffSFCS`, `g2diffOffset`, `g2diffBlink`
- `g3diff`, `g3diffOffset`, `g3diffBlink`, `g3diffBlinkOffset`
- `g3diffTwoComponents`, `g3diffTwoComponentsBlink`
- `g3diffDoubleBlink`
- `g3diffLargeParticles`
- `g3anomalousDiff`, `g3anomalousDiffBlink`
- `siFCS`, `siFCSTwoComponents`
- `g3diffMEMFCS`
- Calibration variants: `...Cal`

### Blinking and offset conventions

- Any model name containing **`Blink`** uses blinking parameters (e.g. `delta`, `F_Blink` / `F_B`).
- Any model name containing **`Offset`** includes an additive offset parameter (`offset`).

The GUI shows the relevant initial parameters automatically.

---

## Initial parameters (model-dependent)

The initial parameter editor displays only the parameters relevant to the currently selected model.

- Values are pre-filled from legacy defaults.
- You can overwrite any shown value.
- Parameters not shown remain at their default values internally.

Typical initial parameters include:
- `N`, `tau diffusion`
- Blinking: `delta`, `F_Blink` (and compatibility key `F_B`)
- Two-component diffusion: `f1`, `rho_D`
- MEMFCS: `tau_D limits`, `number of diffusion components`, `number of iterations`
- Anomalous diffusion: `Gamma`, `Alpha`
- Offset models: `offset`

---

## Calibration parameters (Given D)

For calibration-type models (model name contains `Cal`), the GUI displays:

- **Given D (µm²/s)**
- **Given D temp (°C)**

These are used to temperature-correct a known diffusion coefficient to the experiment temperature.

For non-calibration models, these fields are hidden and not used.

---

## Display and exported outputs

The GUI display typically shows:

- Correlation data and fitted model `G(tau)`
- Weighted residuals
- iMSD (if applicable) or log-log correlation for models where iMSD does not apply
- Histogram of weighted residuals

### Output folder

For each fitted CSV file, outputs are written next to the file into:

`Results/`

### Output files

For an input base name `mycurve` and model `g2diffSFCS`, the program saves:

- `Results/mycurve_g2diffSFCS.svg`  
  (SVG export of the displayed figure)
- `Results/mycurve_g2diffSFCS.csv`  
  (tau, G, sigma_G, fitted curve)
- `Results/mycurve_g2diffSFCS_iMSD.csv`  
  (tau, iMSD) — only for models where iMSD is meaningful
- `Results/g2diffSFCS_fit_summary.csv`  
  (fit parameters; appended per file)

---

## Troubleshooting

### “CSV columns are wrong” / indexing errors
Your CSV must contain at least 4 columns as described above (tau, G, count_rate, sigma_G).
If your exporter uses a different layout, adapt the column indices in the fitter.

### No SVG output
SVG export is performed by the GUI display function. Ensure the fit completed and the Results folder is writable.

### Batch is slow
Batch fitting can be slow for large datasets and complex models. If the GUI updates the plot for every file, that also adds overhead.
You can choose to update the display only for the last completed file.