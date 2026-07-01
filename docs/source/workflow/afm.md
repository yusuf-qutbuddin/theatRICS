# Workflow: AFM profile analysis

The **AFM** tab loads atomic force microscopy height images from
JPK `.jpk-qi-image` files and performs interactive line-profile
analysis for measuring condensate or particle heights and wetting
angles on membrane surfaces.

---

## Overview

The AFM workflow supports:

- Loading JPK QI image files
- Displaying the height image as a false-colour map
- Drawing one or more **line profiles** interactively by clicking
  two points on the image
- Automatic baseline correction using the two clicked endpoint heights
- Savitzky-Golay smoothing of the raw profile
- Detection of **contact points** (where the object meets the membrane)
- Fitting of **wetting angles** (contact angles) on both sides
- Measurement of:
  - **peak height** above the membrane baseline
  - **FWHM** (full width at half maximum)
  - **contact width** (distance between left and right contact points)
- Export of all profiles and measurements to CSV and SVG

---

## Input requirements

- A **JPK QI image file** (`.jpk-qi-image` or `.jpk`)
- The `AFMReader` Python package must be installed

Install AFMReader:

```bash
pip install AFMReader
```

---

## Loading a file

1. Open the **AFM** tab.
2. Under **JPK file**, click **Browse** and select a `.jpk-qi-image` file.
3. Choose the **Channel** to load:
   - `height_trace`
   - `height_retrace`
   - `adhesion_force_trace`
   - `stiffness_trace`
4. Click **Load File**.

The height image is displayed as a false-colour map using the
`afmhot` colormap. Axis units are in µm.

---

## Drawing a line profile

After loading a file:

1. **Click** anywhere on the AFM image to place the **start point**.
   - The start point should be placed on the **bare membrane**
     (flat reference surface) on one side of the object.
2. **Click** again to place the **end point**.
   - The end point should also be on the **bare membrane** on the
     other side.
   - The line must **cross the object** (condensate or particle).

The two clicked heights define the linear baseline representing
the membrane plane. The software:
1. extracts the height profile along the line
2. applies Savitzky-Golay smoothing
3. subtracts the linear baseline
4. detects the contact points
5. fits the wetting angles
6. computes measurements
7. updates the display

Coordinates of each click are logged in the **Results & Logs** tab.

You can draw **multiple profiles** on the same image. Each new
profile is added to the profile list and displayed.

---

## Profile parameters

### Fit points
Number of data points used to fit the slope at each contact point.

Adjusted with a slider (3–60 points). The current value is shown
next to the slider.

Changing this slider **immediately recomputes the wetting angles**
for the currently selected profile without re-extracting the profile.

### Smooth window
Window length for Savitzky-Golay smoothing (must be odd and greater
than `smooth poly order`).

Default: 15

### Smooth poly order
Polynomial order for Savitzky-Golay smoothing.

Default: 3

### Profile points
Number of interpolated sample points along the line.

More points give finer spatial resolution along the profile.

Default: 300

### Threshold fraction
Fraction of the peak height used to define the contact points.

A contact point is the first index from each side where the
baseline-corrected height exceeds:

```
threshold = threshold_fraction × peak_height
```

Default: 0.05 (5% of peak height)

---

## Display panels

The AFM display contains two panels:

### Left panel: height image
Shows the loaded AFM height image with:
- false-colour map (`afmhot` colormap)
- colorbar in nm
- axis units in µm
- drawn profile lines (cyan) labeled with profile numbers
- click markers (+) for placed points

### Right panel: line profile
Shows the most recently selected profile with:
- raw height data (blue, semi-transparent)
- linear baseline (brown dashed)
- baseline-corrected height (blue, solid)
- filled region above zero
- contact point markers (circles) at the membrane level
- tangent lines at each contact point
- data points used for slope fitting (colored circles)
- title showing computed angles

The profile panel title displays:

```
θ_left = X.X°  |  θ_right = X.X°  |  θ_mean = X.X°
```

---

## Profile list

Completed profiles are listed in the **Profiles** listbox with:
- profile number
- peak height in nm
- FWHM in nm
- mean wetting angle in degrees

Click a profile in the listbox to display it in the right panel.

---

## Wetting angle computation

The wetting angle is defined as the angle between:
- the membrane plane (horizontal, after baseline correction)
- the tangent to the object surface at the contact point

For each contact point:
1. `n_fit_points` data points are selected starting at the contact
   point going into the object (inward direction).
2. A line is fitted through these points using `numpy.polyfit`.
3. The wetting angle is:

```
θ = arctan(|slope|)   [degrees]
```

The mean wetting angle is:

```
θ_mean = (θ_left + θ_right) / 2
```

---

## Measurements

For each profile, the following quantities are reported:

| Measurement | Description |
|------------|-------------|
| `peak_height_nm` | Maximum baseline-corrected height in nm |
| `fwhm_nm` | Full width at half maximum in nm |
| `contact_width_nm` | Distance between left and right contact points in nm |
| `theta_left_deg` | Wetting angle at the left contact point in degrees |
| `theta_right_deg` | Wetting angle at the right contact point in degrees |
| `theta_mean_deg` | Mean wetting angle in degrees |

---

## Actions

### Reset last
Removes the most recently drawn profile from the list and redraws
the image and profile display.

### Clear all
Removes all profiles and resets the display.

### Save results
Opens a folder selection dialog and saves:

- `<stem>_profile<n>.csv` per profile  
  Full profile arrays: `distance_nm`, `height_raw_nm`,
  `height_smoothed_nm`, `baseline_nm`, `height_corrected_nm`

- `<stem>_AFM_summary.csv`  
  One row per profile with all scalar measurements

- `<stem>_AFM_overview.svg`  
  Vector export of the current AFM display figure

---

## Outputs

For an input file `sample.jpk-qi-image`, saving to a folder
produces:

```text
<selected folder>/
    sample_profile1.csv
    sample_profile2.csv
    ...
    sample_AFM_summary.csv
    sample_AFM_overview.svg
```

### `*_profile<n>.csv`
Full numerical arrays for one profile:

| Column | Contents |
|--------|----------|
| `distance_nm` | Distance along the profile in nm |
| `height_raw_nm` | Raw sampled height in nm |
| `height_smoothed_nm` | Savitzky-Golay smoothed height in nm |
| `baseline_nm` | Linear baseline in nm |
| `height_corrected_nm` | Baseline-corrected height in nm |

### `*_AFM_summary.csv`
One row per profile with columns:

- `Profile` (number)
- `peak_height_nm`
- `fwhm_nm`
- `contact_width_nm`
- `theta_left_deg`
- `theta_right_deg`
- `theta_mean_deg`
- `start_px_x`, `start_px_y` (click coordinates)
- `end_px_x`, `end_px_y`

### `*_AFM_overview.svg`
Vector export of the current two-panel figure.

---

## Typical use

The AFM tab is designed for measuring the height and wetting
properties of **condensates** or **particles** sitting on a
**supported lipid bilayer** or other flat membrane surface.

### Typical workflow

1. Load the height image.
2. Identify a condensate of interest.
3. Draw a profile crossing the condensate, placing both endpoints
   on the flat membrane surface on either side.
4. Inspect the profile panel:
   - verify the baseline looks flat on both sides of the object
   - verify the contact points are at the base of the object
   - adjust **Fit points** with the slider to change which part
     of the object edge is used for angle calculation
   - adjust **Threshold fraction** if contact points are placed
     at the wrong height level
5. Repeat for additional condensates or profiles.
6. Click **Save results**.

### Optimising contact point detection

If the contact points are placed too high on the object:
- decrease **Threshold fraction** (e.g. from 0.05 to 0.02)

If the contact points are placed on noise rather than the
object edge:
- increase **Threshold fraction** (e.g. from 0.05 to 0.1)

### Optimising the wetting angle

If the fitted slope includes flat membrane rather than the
object edge:
- decrease **Fit points**

If the fitted slope is noisy because too few points are used:
- increase **Fit points**

The slider provides real-time feedback without requiring a
new profile extraction.

---

## Troubleshooting

### "AFMReader is not installed"
Install the required package:

```bash
pip install AFMReader
```

### File will not load
Check:
- the file is a valid JPK QI image
- the selected channel exists in the file
- the file is not corrupted or still being written

### Height image looks inverted or all-black
Try a different channel (e.g. `height_retrace` instead of
`height_trace`).

### No contact points found
This can happen if:
- the profile does not cross the object (both endpoints are on
  the same side)
- the object is too flat relative to the noise level
- the threshold fraction is too high

Try:
- redrawing the profile to cross the object more centrally
- decreasing the **Threshold fraction**

### Wetting angle looks wrong
Check that:
- both click points are on the flat membrane (not on the object)
- the profile crosses the full width of the object
- the **Fit points** slider is not set too high (including flat
  membrane in the slope fit)

### Profile is very noisy
Try:
- increasing the **Smooth window** parameter
- increasing the **Profile points** for finer sampling

### Saving fails
Check that:
- a save folder has been selected
- the folder is writable
- at least one profile has been drawn