# Workflow: Vesicle Finder

The **Vesicle Finder** tab detects vesicles (such as GUVs) in CZI microscopy images or time series, allows interactive selection, and exports cropped time series and membrane straightening outputs centered on each vesicle.

---

## Overview

The workflow has two main phases:

1. **Detect**: segment vesicles in the first frame and display an interactive overlay for user review and selection
2. **Crop / Straighten**: track each vesicle across frames and either export square crops or unroll the membrane into a flat strip

All spatial parameters are entered in **µm**. The software reads the pixel size automatically from the CZI metadata and converts internally.

---

## Input requirements

- A **CZI file** containing one or more image frames
- The file can be a single image or a time series
- At least one channel must contain visible vesicle structures (e.g. membrane dye, transmitted light)

---

## Detection methods

Five segmentation methods are available, selectable from a dropdown:

### Hough Circle Transform (`hough`)

Best for ring-shaped objects with a bright membrane and dark interior, such as fluorescence-labelled GUVs.

The Hough method:
- detects edges using Canny edge detection
- searches for circular patterns in the edge image
- returns center coordinates and radius for each detected circle

The implementation tries **OpenCV** first (faster), and falls back to **scikit-image** if OpenCV is not installed.

Install OpenCV for faster detection:

```bash
pip install opencv-python
```

### Hough Circle Transform for transmitted light (`hough_transmitted`)

Best for transmitted-light images where the GUV edge is a subtle refraction boundary rather than a bright fluorescent ring.

This method:
- applies heavy Gaussian blur to suppress internal fringes
- computes the gradient magnitude to highlight the membrane edge
- runs Hough circle detection on the preprocessed gradient image

### Weighted peripheral intensity (`weighted_intensity`)

Improved and modified from the method from **Kohyama et al., Nature Communications 2022**.

This method:
1. Thresholds the image to find coarse candidate regions
2. Connects membrane arc fragments using binary dilation
3. Finds GUV interiors using flood-fill from the image border
4. Computes a distance transform of the interior regions
5. Identifies GUV centers as peaks in the smoothed distance transform
6. For each candidate, scans a focused neighborhood around the estimated center and radius
7. Returns the circle that maximizes total peripheral intensity (`mean × n_points` along the circumference)

This method works well for both fluorescence membrane images and transmitted-light images.

#### Threshold methods for weighted intensity

The initial thresholding step supports several methods:

| Method | Best use case |
|--------|--------------|
| `mean` | Fluorescence membrane images (bright ring, mostly dark background) |
| `triangle` | Images with small bright foreground on large dark background |
| `huang` | General images with bimodal histogram |
| `otsu` | High-contrast images with clear foreground/background separation |
| `yen` | Fluorescence images |
| `li` | Iterative minimum cross-entropy method |

`mean` and `triangle` typically work best for GUV fluorescence membrane images.

### Cellpose (`cellpose`)

Uses a deep learning model to segment objects.

Best for:
- filled fluorescent objects (bright interior)
- ring-shaped GUVs with the **Invert image** option enabled

Install Cellpose:

```bash
pip install cellpose
```

Optional GPU support requires PyTorch with CUDA.

### Otsu threshold (`otsu`)

Simple intensity thresholding with connected-component labeling. Works for high-contrast images. Always available without extra dependencies.

---

## Parameters

### General parameters

- **CZI file**: the input file
- **Channel**: which channel to use for detection
- **Fallback pixel size (µm)**: used only if the CZI metadata does not contain pixel size
- **Frame start / end / step**: which frames to process
  - Example: frames 0–100, every 5th frame → start=0, end=100, step=5
- **Crop margin (µm)**: half-size of the square crop window around each vesicle center
- **Min area (µm²)**: minimum region area for Otsu/Cellpose fallback segmentation
- **Detection method**: one of `hough`, `hough_transmitted`, `weighted_intensity`, `cellpose`, `otsu`
- **Save debug images**: saves intermediate processing images to a `<filename>_debug/` folder next to the CZI file

### Hough Circle parameters

Shown when `hough` or `hough_transmitted` is selected:

- **Min radius (µm)**: smallest circle to search for
- **Max radius (µm)**: largest circle to search for
- **Radius step (µm)**: step between tested radii (scikit-image backend only)
- **Canny sigma**: edge detection smoothing (scikit-image backend only)
- **Min distance (µm)**: minimum distance between detected circle centers
- **Threshold**: accumulator threshold fraction (lower = more detections)

### Weighted Intensity parameters

Shown when `weighted_intensity` is selected:

- **Min radius (µm)**: smallest GUV radius to search for
- **Max radius (µm)**: largest GUV radius to search for
- **Search range (µm)**: how far from the estimated center to scan in the focused search
- **Threshold method**: thresholding method for initial candidate detection

### Cellpose parameters

Shown when `cellpose` is selected:

- **Model**: Cellpose model variant (`cyto3`, `cyto2`, `cyto`, `nuclei`)
- **Est. diameter (µm)**: estimated vesicle diameter hint (blank = auto-detect)
- **Use GPU**: enable GPU acceleration if PyTorch CUDA is available
- **Invert image**: invert image before detection (recommended for fluorescence ring-shaped GUVs)
- **Min circularity (0–1)**: reject non-circular detections
- **Max eccentricity (0–1)**: reject elongated detections
- **Min solidity (0–1)**: reject irregular detections
- **Preprocess for transmitted light**: apply gradient-based preprocessing
- **Fit circles to detected masks**: fit a geometric circle to each Cellpose mask using least-squares (recommended for ring-shaped GUVs where Cellpose detects arcs rather than filled regions)

---

## Unit conversion

All user-facing spatial parameters are in **µm**.

The software:
1. reads the pixel size from CZI metadata automatically
2. converts all µm values to pixels before processing
3. reports results in both pixels and µm

If the pixel size cannot be read from metadata, the **Fallback pixel size** field is used. If neither is available, the analysis fails with an error.

---

## Detection phase

1. Open the **Vesicle Finder** tab.
2. Select a CZI file.
3. Choose the detection channel.
4. Set the detection method and its parameters.
5. Click **Detect Vesicles**.

The software:
- reads the selected frames from the CZI
- reads the pixel size from metadata
- segments frame 0 using the chosen method
- displays the result

### Display

The preview shows two panels:

**Left panel**: the raw image with detection overlays
- Hough/weighted intensity detections are drawn as circles
- Cellpose/Otsu detections are drawn as contour dots
- each vesicle is labeled with its ID number
- selected vesicles are shown in green, unselected in cyan

**Right panel**: the segmentation label map

### Listbox

Detected vesicles are listed with:
- vesicle label
- radius in pixels and µm (where available)
- equivalent diameter in µm
- area in µm²

---

## Selection

Vesicles can be selected for cropping or straightening in two ways:

### Click on the image
Click inside a vesicle on the left panel to toggle its selection:
- green = selected
- cyan = unselected

### Click in the listbox
Select one or more entries using click or Ctrl+click.

---

## Cropping phase

### Crop Selected
Export only user-selected vesicles.

### Export All
Export all detected vesicles.

### Processing behavior

For each frame in the selected range:
1. the frame is re-segmented independently using the same method
2. each selected vesicle is matched to the closest detection by centroid distance (nearest-centroid tracker)
3. a square crop of size `2 × crop_margin` is extracted around the matched centroid

If a vesicle is not detected in a particular frame, a black (zero-filled) frame is inserted to keep the time series consistent.

### Outputs

For a CZI file named `sample.czi`, outputs are written to:

```text
sample_vesicle_crops/
    vesicle_1.tif
    vesicle_3.tif
    ...
```

Each TIFF contains all selected frames for that vesicle as a multi-frame stack.

---

## Membrane straightening

The membrane straightening feature unrolls the annular band around each detected GUV into a flat rectangular strip.

### Concept

```
Original (circular):          Straightened (rectangular):

      ╭──────╮                ┌────────────────────────────┐
    ╱  thick  ╲               │   membrane pixels unrolled │
   │  ╭────╮  │     →        │                            │
    ╲  ╰────╯ ╱               └────────────────────────────┘
      ╰──────╯                 x axis = position along membrane (µm)
                               y axis = radial position (µm)
```

The x-axis of the rectangular strip spans the full circumference (0° to 360°, mapped to µm). The y-axis spans the membrane thickness from inner to outer edge.

### Parameters

- **Thickness (µm)**: full thickness of the annular band to sample
- **Intensity channel**: which channel to use for membrane intensity measurement (can be different from the detection channel)

### Actions

- **Straighten Selected**: straighten only user-selected vesicles
- **Straighten All**: straighten all detected vesicles with a known radius

Note: straightening requires vesicles with a known radius. This is available for Hough, weighted intensity, and Cellpose with fit circles enabled.

### Display

For each straightened vesicle, the display shows three panels:

**Panel 1: Straightened strip (frame 0)**
- x-axis: position along the membrane circumference in µm
- y-axis: radial position in µm
- color: raw fluorescence intensity (grayscale)

**Panel 2: Intensity heatmap (time series)**
- x-axis: position along the membrane in µm
- y-axis: frame number (time)
- color: mean fluorescence intensity across the membrane thickness
- shows whether intensity is uniform or varies around the membrane or over time

For single-frame data, shows a 1D intensity profile instead.

**Panel 3: Total membrane intensity vs time**
- x-axis: frame number
- y-axis: total integrated membrane intensity
- useful for detecting photobleaching or intensity changes over time

For single-frame data, shows a bar chart.

### Outputs

For each vesicle, the software saves:

```text
sample_straightened/
    vesicle_1_straightened.tif       ← 3D TIFF (frames × thickness × circumference)
    vesicle_1_intensity_profile.csv  ← mean intensity vs angle and frame
    vesicle_1_total_intensity.csv    ← total intensity per frame
    straighten_overview.svg          ← combined figure
```

---

## Debug images

When **Save debug images** is checked, intermediate processing images are saved to `<filename>_debug/` alongside the CZI file.

For the weighted intensity method, the following images are saved:

| File | Contents |
|------|----------|
| `01_raw_normalized.tif` | Normalized input image |
| `04_binary_after_threshold.tif` | Binary mask after thresholding |
| `05_binary_dilated.tif` | Binary mask after dilation (connects membrane arcs) |
| `06_interior.tif` | Interior mask (dark regions inside GUVs) |
| `07_distance_interior.tif` | Distance transform of interior (peaks at GUV centers) |
| `08_distance_smooth.tif` | Smoothed distance transform used for peak detection |

These images are useful for diagnosing why detection fails and for choosing optimal parameters.

---

## Troubleshooting

### No vesicles detected (Hough)
- lower the threshold (e.g. 0.2 instead of 0.3)
- ensure the min/max radius range covers the actual vesicle sizes
- reduce the min distance if vesicles are close together
- check that the correct channel is selected
- try the `hough_transmitted` method for transmitted-light images

### No vesicles detected (weighted intensity)
- try different threshold methods (`mean` or `triangle` recommended for membrane images)
- increase the search range
- ensure the min/max radius range is appropriate
- enable debug images to inspect which step is failing
- if the binary image shows only small fragments, the dilation may need to be larger (the code adapts automatically, but very thin arcs may still fail)

### Only one vesicle detected instead of several
- the weighted intensity method uses distance transform peaks to separate GUVs
- if GUVs are very close together the peaks may merge
- check the `08_distance_smooth.tif` debug image to verify separate peaks are visible
- increase the min distance parameter to force more peaks

### Hough detects contact zones instead of vesicles
- this happens when the detection method finds filled bright regions rather than ring edges
- switch to the **Hough** method which is specifically designed for circular edges
- for transmitted light use **hough_transmitted**

### Pixel size not found
- enter a fallback pixel size manually
- without pixel size, µm parameters cannot be converted to pixels

### Cellpose finds arc instead of full circle
- enable **Fit circles to detected masks**
- this fits a geometric circle to the arc using least-squares and returns the full circle

### Straightening not available
- straightening requires a known radius
- use Hough, weighted intensity, or Cellpose with fit circles enabled
- Otsu-only detections do not have a radius estimate

### Detection is slow (weighted intensity)
- reduce the search range
- increase the radius step
- the weighted intensity scan is vectorized and uses focused search around estimated centers
- further speedup is possible with GPU (CuPy) if available

### Hough is slow (scikit-image backend)
- install OpenCV for much faster Hough circle detection:
  ```bash
  pip install opencv-python
  ```
- OpenCV is used automatically if available