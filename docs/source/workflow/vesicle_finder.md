# Workflow: Vesicle Finder

The **Vesicle Finder** tab detects vesicles (such as GUVs) in CZI microscopy images or time series, allows interactive selection, and exports cropped time series centered on each vesicle.

---

## Overview

The workflow has two phases:

1. **Detect**: segment vesicles in the first frame and display an interactive overlay for user review and selection
2. **Crop**: re-segment every selected frame, track each vesicle by nearest centroid, and export one TIFF per vesicle

---

## Input requirements

- A **CZI file** containing one or more image frames
- The file can be a single image or a time series
- At least one channel must contain visible vesicle structures (e.g. membrane dye)

---

## Detection methods

Three segmentation methods are available, selectable from a dropdown:

### Hough Circle Transform (recommended for GUVs)

Best for ring-shaped objects such as GUVs with bright membranes and dark interiors.

The Hough method:
- detects edges using Canny edge detection
- searches for circular patterns in the edge image
- returns center coordinates and radius for each detected circle

The implementation tries **OpenCV** first (faster), and falls back to **scikit-image** if OpenCV is not installed.

Install OpenCV for faster detection:

```bash
pip install opencv-python
```

### Cellpose (recommended for filled objects)

Best for filled cell-like objects where the interior is bright.

Cellpose uses a deep learning model to segment objects. GPU acceleration is supported if PyTorch with CUDA is installed.

Install Cellpose:

```bash
pip install cellpose
```

For GPU support, also install PyTorch with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Otsu threshold (fallback)

A simple intensity-based thresholding method. Works for high-contrast images but may fail on complex data. Always available without extra dependencies.

---

## Parameters

All spatial parameters are entered in **µm**. The software reads the pixel size from the CZI metadata and converts internally. If metadata is unavailable, a fallback pixel size can be provided.

### General parameters

- **CZI file**: the input file to analyze
- **Channel**: which channel to use for detection
- **Fallback pixel size (µm)**: used only if the CZI metadata does not contain pixel size information
- **Frame start / end / step**: which frames to process
  - Example: frames 0–100, every 5th frame → start=0, end=100, step=5
- **Crop margin (µm)**: half-size of the square crop window around each vesicle center
- **Min area (µm²)**: minimum region area for Otsu/Cellpose fallback segmentation
- **Detection method**: one of `hough`, `cellpose`, `otsu`

### Hough Circle parameters

Shown only when the Hough method is selected:

- **Min radius (µm)**: smallest circle radius to search for
- **Max radius (µm)**: largest circle radius to search for
- **Radius step (µm)**: step between tested radii (smaller = more precise but slower). Used only by the scikit-image fallback; OpenCV determines this internally.
- **Canny sigma**: Gaussian smoothing sigma for edge detection. Used only by the scikit-image fallback.
- **Min distance (µm)**: minimum distance between detected circle centers
- **Threshold fraction**: fraction of the strongest accumulator peak used as detection threshold. Lower values produce more detections but may include false positives.

### Cellpose parameters

Shown only when the Cellpose method is selected:

- **Model**: Cellpose model variant (`cyto3`, `cyto2`, `cyto`, `nuclei`)
- **Est. diameter (µm)**: estimated vesicle diameter hint for Cellpose. Leave blank for automatic estimation.
- **Use GPU**: enable GPU acceleration if PyTorch CUDA is available

---

## Unit conversion

All user-facing spatial parameters are in **µm**.

The software:
1. reads the pixel size from CZI metadata automatically
2. converts all µm values to pixels before processing
3. reports results in both pixels and µm

If the pixel size cannot be read from the metadata, the **Fallback pixel size** field is used. If neither is available, the analysis fails with an error message.

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
- Hough detections are drawn as circles
- Cellpose/Otsu detections are drawn as contour dots
- each vesicle is labeled with its ID number
- selected vesicles are shown in green, unselected in cyan

**Right panel**: the segmentation label map using a colorful colormap

### Listbox

Detected vesicles are listed with:
- vesicle label
- radius in pixels and µm (Hough method)
- equivalent diameter in µm
- area in µm²

---

## Selection

Vesicles can be selected for cropping in two ways:

### Click on the image
Click inside a vesicle on the left panel to toggle its selection:
- green outline = selected
- cyan outline = unselected

### Click in the listbox
Select one or more entries using standard click / Ctrl+click.

The display updates automatically when the selection changes.

---

## Cropping phase

### Crop Selected
Click **Crop Selected** to export only the user-selected vesicles.

### Export All
Click **Export All** to export all detected vesicles.

### Processing behavior

For each frame in the selected range:
1. the frame is re-segmented independently using the same method and parameters
2. each selected vesicle is matched to the closest detection by centroid distance
3. a square crop of size `2 × crop_margin` (in pixels) is extracted around the matched centroid

This re-segmentation approach handles vesicles that move slowly across frames.

### Centroid tracking

The tracker works as follows:
- on the first frame, the centroid of each selected vesicle is stored as a reference
- on each subsequent frame, all detected vesicles are compared to the reference by Euclidean distance
- the closest match is chosen
- the reference centroid is updated to the new position

This allows the tracker to follow slowly drifting vesicles. If two vesicles are very close, the tracker may confuse them.

### Missing detections

If a vesicle is not detected in a particular frame:
- a black (zero-filled) frame is inserted to keep the time series length consistent

---

## Outputs

For a CZI file named `sample.czi`, outputs are written to:

```text
sample_vesicle_crops/
    vesicle_1.tif
    vesicle_3.tif
    ...
```

Each TIFF contains all selected frames for that vesicle as a multi-frame stack.

The crop size is `2 × crop_margin` in both dimensions (square window).

---

## Hough detection: OpenCV vs scikit-image

The Hough circle detection has two backends:

### OpenCV (preferred)
- significantly faster, especially on large images
- uses `cv2.HoughCircles()` with the gradient method
- requires `opencv-python` to be installed

### scikit-image (fallback)
- always available (part of `scikit-image`)
- slower because it tests a grid of discrete radii
- uses `radius_step` and `canny_sigma` parameters

The software automatically selects the best available backend and logs which one is used.

---

## Cellpose: CPU vs GPU

Cellpose can run on CPU (default) or GPU:

### CPU mode
- always works
- slower for large images

### GPU mode
- requires PyTorch with CUDA
- enabled by checking **Use GPU** in the Cellpose parameters
- the checkbox is automatically disabled if CUDA is not available
- typically 5–20× faster than CPU

---

## Troubleshooting

### No vesicles detected (Hough)
- try lowering the threshold fraction (e.g. 0.2 instead of 0.3)
- ensure the min/max radius range covers the actual vesicle sizes
- reduce the min distance if vesicles are close together
- check that the correct channel is selected
- try increasing canny sigma for noisy images (scikit-image backend)

### No vesicles detected (Cellpose)
- try a different model (e.g. `cyto2` instead of `cyto3`)
- adjust the estimated diameter
- enable GPU for better accuracy on large images

### Hough detects interfaces instead of vesicles
This happens when the segmentation method fills bright regions rather than detecting ring shapes.
- switch from Cellpose/Otsu to **Hough** for ring-shaped GUVs
- the Hough method is specifically designed for circular membrane structures

### Pixel size not found
- check that the CZI file contains valid metadata
- enter a fallback pixel size manually
- without pixel size, the software cannot convert µm parameters to pixels

### Vesicle tracking fails across frames
- reduce frame step to decrease movement between frames
- if vesicles are very close to each other, the nearest-centroid tracker may swap them
- consider increasing the crop margin so the vesicle stays within the window even if the centroid shifts slightly

### Crop extends outside the image
- the cropper zero-pads regions outside the image boundary
- this appears as black borders in the output TIFF

### Cellpose is slow
- enable GPU if available
- reduce the number of frames
- increase the frame step
- consider using Hough instead for circular vesicles

### OpenCV is not installed
- the software falls back to scikit-image automatically
- install OpenCV for faster Hough detection:

```bash
pip install opencv-python
```