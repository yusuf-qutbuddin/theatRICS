from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import tifffile
import xml.etree.ElementTree as ET
from pylibCZIrw import czi as pyczi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, disk as morph_disk
from scipy import ndimage
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
# Try importing cellpose; set flag if unavailable
try:
    from cellpose import models as cp_models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False
# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def read_pixel_size_from_czi(czi_path: str) -> Optional[float]:
    """
    Read pixel size in µm from CZI metadata.
    Returns None if not found.
    """
    try:
        with pyczi.open_czi(str(czi_path)) as czidoc:
            root = ET.fromstring(czidoc.raw_metadata)
            for dist in root.iter():
                if dist.tag.split('}')[-1] == 'Distance':
                    if dist.get('Id', '') == 'X':
                        for child in dist:
                            if child.tag.split('}')[-1] == 'Value' and child.text:
                                try:
                                    val_m = float(child.text.strip())
                                    if val_m > 0:
                                        return val_m * 1e6  # metres → µm
                                except ValueError:
                                    pass
    except Exception:
        pass
    return None

def um_to_px(value_um: float, pixel_size_um: float) -> int:
    """Convert a distance in µm to pixels (rounded to nearest int)."""
    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError("Pixel size not available; cannot convert µm to pixels")
    return max(1, int(round(value_um / pixel_size_um)))
def segment_hough_circles(
    image: np.ndarray,
    min_radius: int = 50,
    max_radius: int = 500,
    radius_step: int = 5,
    canny_sigma: float = 2.0,
    min_distance: int = 100,
    threshold_fraction: float = 0.3,
    max_circles: int = 20,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detect circular vesicles using Hough Circle Transform.
    Tries OpenCV first (faster), falls back to skimage.
    """
    if OPENCV_AVAILABLE:
        try:
            return segment_hough_circles_opencv(
                image,
                min_radius=min_radius,
                max_radius=max_radius,
                canny_threshold=100.0,
                accumulator_threshold=max(10.0, threshold_fraction * 100.0),
                min_distance=min_distance,
                max_circles=max_circles,
            )
        except Exception:
            pass

    return _segment_hough_circles_skimage(
        image,
        min_radius=min_radius,
        max_radius=max_radius,
        radius_step=radius_step,
        canny_sigma=canny_sigma,
        min_distance=min_distance,
        threshold_fraction=threshold_fraction,
        max_circles=max_circles,
    )

def _segment_hough_circles_skimage(
    image: np.ndarray,
    min_radius: int = 50,
    max_radius: int = 500,
    radius_step: int = 5,
    canny_sigma: float = 2.0,
    min_distance: int = 100,
    threshold_fraction: float = 0.3,
    max_circles: int = 20,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detect circular vesicles using the Hough Circle Transform.

    This is ideal for ring-shaped objects (bright membrane, dark interior)
    such as GUVs imaged with a membrane dye.

    Parameters
    ----------
    image : 2D array (Y, X)
    min_radius : smallest circle radius to search for (pixels)
    max_radius : largest circle radius to search for (pixels)
    radius_step : step between tested radii (pixels). Smaller = more precise but slower.
    canny_sigma : Gaussian smoothing sigma for edge detection
    min_distance : minimum distance between detected circle centers (pixels)
    threshold_fraction : fraction of the strongest accumulator peak used as detection threshold.
                         Lower = more detections (possibly false positives).
    max_circles : maximum number of circles to return
    
    Returns
    -------
    labels : 2D int array (Y, X), each circle filled with a unique label
    circles : list of dicts with keys: label, centroid_y, centroid_x, radius, area, equivalent_diameter
    """
    # normalize to float 0–1
    img = image.astype(np.float64)
    if img.max() > 0:
        img = img / img.max()

    # edge detection
    edges = canny(img, sigma=canny_sigma)

    # range of radii to test
    radii = np.arange(min_radius, max_radius + 1, radius_step)

    if len(radii) == 0:
        return np.zeros(image.shape[:2], dtype=np.int32), []

    # Hough transform
    hough_res = hough_circle(edges, radii)

    # find peaks in the accumulator
    # threshold is relative to the strongest peak
    accum_max = np.max(hough_res) if hough_res.size > 0 else 1.0
    threshold = accum_max * threshold_fraction

    accums, cx_arr, cy_arr, rad_arr = hough_circle_peaks(
        hough_res, radii,
        min_xdistance=min_distance,
        min_ydistance=min_distance,
        threshold=threshold,
        num_peaks=max_circles,
        total_num_peaks=max_circles,
    )

    # build label image: fill each circle
    h, w = image.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    circles = []

    for i, (acc, cx, cy, r) in enumerate(zip(accums, cx_arr, cy_arr, rad_arr), start=1):
        # create circular mask
        yy, xx = np.ogrid[:h, :w]
        circle_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2

        # only fill pixels not already claimed by a stronger detection
        fill_mask = circle_mask & (labels == 0)
        labels[fill_mask] = i

        area = int(np.sum(circle_mask))
        circles.append({
            "label": i,
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "radius": int(r),
            "bbox": (
                max(0, int(cy - r)),
                max(0, int(cx - r)),
                min(h, int(cy + r)),
                min(w, int(cx + r)),
            ),
            "area": area,
            "equivalent_diameter": float(2 * r),
        })

    return labels, circles

def segment_hough_circles_opencv(
    image: np.ndarray,
    min_radius: int = 50,
    max_radius: int = 500,
    canny_threshold: float = 100.0,
    accumulator_threshold: float = 30.0,
    min_distance: int = 100,
    max_circles: int = 20,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detect circular vesicles using OpenCV HoughCircles.
    Always searches from radius 0 internally for robustness.
    Post-filters to user's [min_radius, max_radius].
    """
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV not installed.")

    img = image.astype(np.float64)
    if img.max() > 0:
        img = (img / img.max() * 255).astype(np.uint8)
    else:
        return np.zeros(image.shape[:2], dtype=np.int32), []

    # enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # fixed moderate blur
    blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)

    # ALWAYS search from 0 to max_radius internally
    # dp=1.0 is more reliable than 1.5 for large circles
    # param1 = Canny high threshold (low = param1/2 internally)
    # param2 = accumulator threshold (lower = more sensitive)
    circles = None
    for dp_val in [1.0, 1.5, 2.0]:
        for param2_val in [accumulator_threshold, accumulator_threshold * 0.5, 10.0, 5.0]:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=dp_val,
                minDist=max(min_distance, 10),
                param1=canny_threshold,
                param2=max(3.0, param2_val),
                minRadius=0,
                maxRadius=max_radius,
            )
            if circles is not None and len(circles[0]) > 0:
                break
        if circles is not None and len(circles[0]) > 0:
            break

    h, w = image.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    circle_list = []

    if circles is not None:
        all_circles = np.round(circles[0]).astype(int)

        # POST-FILTER only: apply user's min/max radius
        filtered = [(cx, cy, r) for (cx, cy, r) in all_circles
                     if min_radius <= r <= max_radius]

        # sort by radius descending (largest first gets priority in label map)
        filtered.sort(key=lambda c: c[2], reverse=True)

        if len(filtered) > max_circles:
            filtered = filtered[:max_circles]

        for i, (cx, cy, r) in enumerate(filtered, start=1):
            yy, xx = np.ogrid[:h, :w]
            circle_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
            fill_mask = circle_mask & (labels == 0)
            labels[fill_mask] = i

            area = int(np.sum(circle_mask))
            circle_list.append({
                "label": i,
                "centroid_y": float(cy),
                "centroid_x": float(cx),
                "radius": int(r),
                "bbox": (
                    max(0, int(cy - r)),
                    max(0, int(cx - r)),
                    min(h, int(cy + r)),
                    min(w, int(cx + r)),
                ),
                "area": area,
                "equivalent_diameter": float(2 * r),
            })

    return labels, circle_list

def segment_hough_transmitted(
    image: np.ndarray,
    min_radius: int = 50,
    max_radius: int = 500,
    min_distance: int = 100,
    threshold_fraction: float = 0.3,
    max_circles: int = 20,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Hough circle detection optimized for transmitted-light images.
    Uses gradient-based preprocessing to suppress internal fringes.
    """
    from skimage.filters import gaussian
    from skimage.feature import canny
    
    img = image.astype(np.float64)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    
    # heavy blur to kill internal fringes
    blurred = gaussian(img, sigma=10.0)
    
    # gradient magnitude
    gy, gx = np.gradient(blurred)
    gradient = np.sqrt(gx**2 + gy**2)
    
    # smooth gradient
    gradient = gaussian(gradient, sigma=3.0)
    
    # normalize
    gradient = gradient / gradient.max() if gradient.max() > 0 else gradient
    
    # now try OpenCV on the preprocessed gradient image
    if OPENCV_AVAILABLE:
        try:
            img_uint8 = (gradient * 255).astype(np.uint8)
            
            circles = None
            for param2_val in [20, 15, 10, 5]:
                circles = cv2.HoughCircles(
                    img_uint8,
                    cv2.HOUGH_GRADIENT,
                    dp=1.0,
                    minDist=max(min_distance, 10),
                    param1=50,
                    param2=param2_val,
                    minRadius=0,
                    maxRadius=max_radius,
                )
                if circles is not None and len(circles[0]) > 0:
                    break
            
            if circles is not None:
                h, w = image.shape[:2]
                labels = np.zeros((h, w), dtype=np.int32)
                circle_list = []
                
                all_circles = np.round(circles[0]).astype(int)
                filtered = [(cx, cy, r) for (cx, cy, r) in all_circles
                           if min_radius <= r <= max_radius]
                filtered.sort(key=lambda c: c[2], reverse=True)
                
                if len(filtered) > max_circles:
                    filtered = filtered[:max_circles]
                
                for i, (cx, cy, r) in enumerate(filtered, start=1):
                    yy, xx = np.ogrid[:h, :w]
                    circle_mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
                    fill_mask = circle_mask & (labels == 0)
                    labels[fill_mask] = i
                    
                    area = int(np.sum(circle_mask))
                    circle_list.append({
                        "label": i,
                        "centroid_y": float(cy),
                        "centroid_x": float(cx),
                        "radius": int(r),
                        "bbox": (max(0, int(cy-r)), max(0, int(cx-r)),
                                min(h, int(cy+r)), min(w, int(cx+r))),
                        "area": area,
                        "equivalent_diameter": float(2*r),
                    })
                
                return labels, circle_list
        except Exception:
            pass
    
    # fallback: use skimage Hough on edge image
    edges = canny(gradient, sigma=1.0)
    radii = np.arange(min_radius, max_radius + 1, max(1, (max_radius - min_radius) // 50))
    
    if len(radii) == 0:
        return np.zeros(image.shape[:2], dtype=np.int32), []
    
    hough_res = hough_circle(edges, radii)
    accum_max = np.max(hough_res) if hough_res.size > 0 else 1.0
    threshold = accum_max * threshold_fraction
    
    accums, cx_arr, cy_arr, rad_arr = hough_circle_peaks(
        hough_res, radii,
        min_xdistance=min_distance,
        min_ydistance=min_distance,
        threshold=max(1.0, threshold),
        num_peaks=max_circles,
        total_num_peaks=max_circles,
    )
    
    h, w = image.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    circle_list = []
    
    for i, (acc, cx, cy, r) in enumerate(zip(accums, cx_arr, cy_arr, rad_arr), start=1):
        yy, xx = np.ogrid[:h, :w]
        circle_mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
        fill_mask = circle_mask & (labels == 0)
        labels[fill_mask] = i
        
        area = int(np.sum(circle_mask))
        circle_list.append({
            "label": i,
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "radius": int(r),
            "bbox": (max(0, int(cy-r)), max(0, int(cx-r)),
                    min(h, int(cy+r)), min(w, int(cx+r))),
            "area": area,
            "equivalent_diameter": float(2*r),
        })
    
    return labels, circle_list

def segment_cellpose(
    image: np.ndarray,
    model_type: str = "cyto3",
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu: bool = False,
    invert: bool = False,
    filter_circularity: float = 0.0,
    filter_eccentricity: float = 1.0,
    filter_solidity: float = 0.0,
    preprocess_transmitted: bool = False,
    fit_circles: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Segment vesicles using Cellpose with optional shape filtering
    and transmitted-light preprocessing.

    Parameters
    ----------
    image : 2D array (Y, X)
    model_type : Cellpose model name
    diameter : estimated object diameter in pixels (None = auto)
    flow_threshold : Cellpose flow threshold
    cellprob_threshold : Cellpose cell probability threshold
    gpu : use GPU if available
    invert : invert image (for ring-shaped GUVs with bright membrane)
    filter_circularity : minimum circularity (0-1). 0 = no filter.
        circularity = 4π × area / perimeter²
        1.0 = perfect circle
    filter_eccentricity : maximum eccentricity (0-1). 1 = no filter.
        0.0 = perfect circle, 1.0 = line
    filter_solidity : minimum solidity (0-1). 0 = no filter.
        solidity = area / convex_hull_area
        1.0 = perfectly convex

    Returns
    -------
    labels : 2D int array, 0 = background, 1..N = vesicle IDs
    If fit_circles=True, fits a circle to each detected mask and returns
    filled circular labels + circle info (like Hough output).
    If fit_circles=False, returns raw Cellpose masks + regionprops info.
    """
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose is not installed.")

    img = image.copy().astype(np.float64)

    if preprocess_transmitted:
        img = _preprocess_transmitted_light(img)

    model = cp_models.Cellpose(model_type=model_type, gpu=gpu)

    masks_list, flows, styles, diams = model.eval(
        [img],
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=[0, 0],
        invert=invert,
    )
    masks = masks_list[0].astype(np.int32)

    if filter_circularity > 0 or filter_eccentricity < 1.0 or filter_solidity > 0:
        masks = _filter_masks_by_shape(
            masks,
            min_circularity=filter_circularity,
            max_eccentricity=filter_eccentricity,
            min_solidity=filter_solidity,
        )

    if fit_circles:
        return _convert_masks_to_circles(masks, image.shape)
    else:
        return masks


def _convert_masks_to_circles(
    masks: np.ndarray,
    image_shape: tuple,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Convert Cellpose masks to filled circles by fitting a circle
    to each mask's pixels.
    """
    h, w = image_shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    circles = []

    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]

    label_counter = 0
    for lbl in unique_labels:
        mask = masks == lbl
        fit = _fit_circle_to_mask(mask)

        if fit is None:
            continue

        label_counter += 1
        cx = fit["centroid_x"]
        cy = fit["centroid_y"]
        r = fit["radius"]

        yy, xx = np.ogrid[:h, :w]
        circle_mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
        fill_mask = circle_mask & (labels == 0)
        labels[fill_mask] = label_counter

        area = int(np.sum(circle_mask))
        circles.append({
            "label": label_counter,
            "centroid_y": cy,
            "centroid_x": cx,
            "radius": int(round(r)),
            "bbox": (
                max(0, int(cy - r)),
                max(0, int(cx - r)),
                min(h, int(cy + r)),
                min(w, int(cx + r)),
            ),
            "area": area,
            "equivalent_diameter": float(2 * r),
        })

    return labels, circles


def _preprocess_transmitted_light(image: np.ndarray) -> np.ndarray:
    """
    Preprocess transmitted-light image for vesicle detection.
    
    Strategy:
    1. Strong Gaussian blur to remove internal fringes/texture
    2. Compute gradient magnitude (highlights the membrane edge only)
    3. Second Gaussian blur to connect edge fragments
    4. Normalize to 0-255
    """
    from skimage.filters import gaussian
    from skimage.exposure import rescale_intensity
    
    img = image.astype(np.float64)
    
    # normalize to 0-1
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    
    # step 1: heavy blur to remove internal texture/fringes
    # sigma should be large enough to smooth out fringes but small
    # enough to keep the membrane edge
    blurred = gaussian(img, sigma=10.0)
    
    # step 2: compute gradient magnitude
    # this highlights intensity transitions = membrane edge
    gy, gx = np.gradient(blurred)
    gradient = np.sqrt(gx**2 + gy**2)
    
    # step 3: smooth the gradient to connect edge fragments
    gradient_smooth = gaussian(gradient, sigma=5.0)
    
    # step 4: normalize to 0-255
    result = rescale_intensity(gradient_smooth, out_range=(0.0, 255.0))
    
    return result


def _filter_masks_by_shape(
    masks: np.ndarray,
    min_circularity: float = 0.0,
    max_eccentricity: float = 1.0,
    min_solidity: float = 0.0,
) -> np.ndarray:
    """
    Filter labeled regions by circularity, eccentricity, and solidity.
    Removes regions that don't pass the thresholds.
    Returns a relabeled mask.
    """
    props = regionprops(masks)
    keep_labels = set()

    for p in props:
        # circularity = 4π × area / perimeter²
        perimeter = p.perimeter
        if perimeter > 0:
            circularity = 4.0 * np.pi * p.area / (perimeter ** 2)
        else:
            circularity = 0.0

        eccentricity = p.eccentricity
        solidity = p.solidity

        if (circularity >= min_circularity and
            eccentricity <= max_eccentricity and
            solidity >= min_solidity):
            keep_labels.add(p.label)

    # zero out rejected regions
    filtered = masks.copy()
    for p in props:
        if p.label not in keep_labels:
            filtered[filtered == p.label] = 0

    # relabel sequentially
    filtered = label(filtered > 0).astype(np.int32)
    return filtered


    

def segment_otsu_watershed(image: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    Fallback segmentation using Otsu threshold + connected components.

    Parameters
    ----------
    image : 2D array (Y, X)
    min_area : minimum region area in pixels

    Returns
    -------
    labels : 2D int array
    """
    if image.dtype != np.float64:
        img = image.astype(np.float64)
    else:
        img = image

    thresh = threshold_otsu(img)
    binary = img > thresh
    binary = binary_opening(binary, morph_disk(3))

    labeled = label(binary)

    # remove small regions
    for prop in regionprops(labeled):
        if prop.area < min_area:
            labeled[labeled == prop.label] = 0

    # relabel sequentially
    labeled = label(labeled > 0)
    return labeled.astype(np.int32)

def segment_frame(
    image: np.ndarray,
    method: str = "hough",
    use_cellpose: bool = True,
    model_type: str = "cyto3",
    diameter: Optional[float] = None,
    min_area: int = 200,
    min_radius: int = 50,
    max_radius: int = 500,
    radius_step: int = 5,
    canny_sigma: float = 2.0,
    hough_min_distance: int = 100,
    hough_threshold_fraction: float = 0.3,
    cellpose_gpu: bool = False,
    cellpose_invert: bool = False,
    filter_circularity: float = 0.0,
    filter_eccentricity: float = 1.0,
    filter_solidity: float = 0.0,
    preprocess_transmitted: bool = False,
    fit_circles: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:

    if method == "hough":
        labels, vesicles = segment_hough_circles(
            image,
            min_radius=min_radius,
            max_radius=max_radius,
            radius_step=radius_step,
            canny_sigma=canny_sigma,
            min_distance=hough_min_distance,
            threshold_fraction=hough_threshold_fraction,
        )
        return labels, vesicles

    elif method == "cellpose":
        if use_cellpose and CELLPOSE_AVAILABLE:
            try:
                result = segment_cellpose(
                    image,
                    model_type=model_type,
                    diameter=diameter,
                    gpu=cellpose_gpu,
                    invert=cellpose_invert,
                    filter_circularity=filter_circularity,
                    filter_eccentricity=filter_eccentricity,
                    filter_solidity=filter_solidity,
                    preprocess_transmitted=preprocess_transmitted,
                    fit_circles=fit_circles,
                )
                if fit_circles:
                    labels, vesicles = result
                    return labels, vesicles
                else:
                    masks = result
                    vesicles = extract_vesicle_info(masks)
                    return masks, vesicles
            except Exception:
                pass
        labels = segment_otsu_watershed(image, min_area=min_area)
        vesicles = extract_vesicle_info(labels)
        return labels, vesicles

    elif method == "otsu":
        labels = segment_otsu_watershed(image, min_area=min_area)
        vesicles = extract_vesicle_info(labels)
        return labels, vesicles
    elif method == "hough_transmitted":
        labels, vesicles = segment_hough_transmitted(
            image,
            min_radius=min_radius,
            max_radius=max_radius,
            min_distance=hough_min_distance,
            threshold_fraction=hough_threshold_fraction,
            max_circles=20,
        )
        return labels, vesicles

    else:
        raise ValueError(f"Unknown segmentation method: {method}")


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def extract_vesicle_info(labels: np.ndarray) -> List[Dict[str, Any]]:
    """
    Extract centroid, bounding box, and area for each labeled region.

    Returns list of dicts with keys:
      label, centroid_y, centroid_x, bbox (min_row, min_col, max_row, max_col), area
    """
    props = regionprops(labels)
    vesicles = []
    for p in props:
        vesicles.append({
            "label": int(p.label),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
            "bbox": tuple(int(x) for x in p.bbox),  # (min_row, min_col, max_row, max_col)
            "area": int(p.area),
            "equivalent_diameter": float(p.equivalent_diameter),
        })
    return vesicles


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def crop_square(
    image: np.ndarray,
    center_y: float,
    center_x: float,
    half_size: int,
) -> np.ndarray:
    """
    Crop a square window from a 2D image, zero-padding if the crop extends outside.
    """
    h, w = image.shape[:2]
    cy, cx = int(round(center_y)), int(round(center_x))

    y0 = cy - half_size
    y1 = cy + half_size
    x0 = cx - half_size
    x1 = cx + half_size

    # output
    crop = np.zeros((2 * half_size, 2 * half_size), dtype=image.dtype)

    # source slice (clamped)
    sy0 = max(y0, 0)
    sy1 = min(y1, h)
    sx0 = max(x0, 0)
    sx1 = min(x1, w)

    # destination slice
    dy0 = sy0 - y0
    dy1 = dy0 + (sy1 - sy0)
    dx0 = sx0 - x0
    dx1 = dx0 + (sx1 - sx0)

    crop[dy0:dy1, dx0:dx1] = image[sy0:sy1, sx0:sx1]
    return crop


# ---------------------------------------------------------------------------
# CZI reading
# ---------------------------------------------------------------------------

def read_czi_frames(
    czi_path: str,
    channel: int = 0,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_step: int = 1,
) -> Tuple[np.ndarray, int]:
    """
    Read selected frames from a CZI file.

    Returns
    -------
    stack : 3D array (T_selected, Y, X)
    n_total_frames : total number of frames in the file
    """
    czi_path = str(czi_path)
    with pyczi.open_czi(czi_path) as czidoc:
        bbox = czidoc.total_bounding_box
        n_total = bbox.get('T', (0, 1))[1]

        if frame_end is None or frame_end > n_total:
            frame_end = n_total

        frame_indices = list(range(frame_start, frame_end, max(1, frame_step)))

        frames = []
        for t in frame_indices:
            plane = czidoc.read(plane={'T': t, 'C': channel, 'Z': 0})
            frames.append(np.squeeze(plane))

    stack = np.stack(frames, axis=0)
    return stack, n_total


# ---------------------------------------------------------------------------
# Full pipeline: one file
# ---------------------------------------------------------------------------

def process_vesicle_detection(
    czi_path: str,
    channel: int = 0,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_step: int = 1,
    crop_margin_um: float = 5.0,
    method: str = "hough",
    use_cellpose: bool = True,
    model_type: str = "cyto3",
    diameter: Optional[float] = None,
    min_area_um2: float = 1.0,
    min_radius_um: float = 1.0,
    max_radius_um: float = 20.0,
    radius_step_um: float = 0.5,
    canny_sigma: float = 2.0,
    hough_min_distance_um: float = 5.0,
    hough_threshold_fraction: float = 0.3,
    cellpose_gpu: bool = False,
    cellpose_invert: bool = False,
    filter_circularity: float = 0.0,
    filter_eccentricity: float = 1.0,
    filter_solidity: float = 0.0,
    preprocess_transmitted: bool = False,
    fit_circles: bool = False,
    fallback_pixel_size_um: Optional[float] = None,
    selected_labels: Optional[List[int]] = None,
    progress_queue=None,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Full vesicle detection + cropping pipeline.
    
    All spatial parameters are in µm. They are converted to pixels
    internally using the pixel size read from the CZI metadata.
    If metadata is unavailable, fallback_pixel_size_um is used.
    """
    if cancel_event is not None and cancel_event.is_set():
        return {"mode": "cancelled"}

    # Read pixel size from metadata
    pixel_size_um = read_pixel_size_from_czi(czi_path)
    if pixel_size_um is None:
        pixel_size_um = fallback_pixel_size_um

    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError(
            "Pixel size not found in CZI metadata and no fallback provided. "
            "Please enter a fallback pixel size."
        )

    # Convert µm → pixels
    crop_margin_px = um_to_px(crop_margin_um, pixel_size_um)
    min_radius_px = um_to_px(min_radius_um, pixel_size_um)
    max_radius_px = um_to_px(max_radius_um, pixel_size_um)
    radius_step_px = max(1, um_to_px(radius_step_um, pixel_size_um))
    hough_min_distance_px = um_to_px(hough_min_distance_um, pixel_size_um)
    min_area_px = max(1, int(round(min_area_um2 / (pixel_size_um ** 2))))

   
    
    # Cellpose diameter: if provided in µm, convert
    diameter_px = None
    if diameter is not None:
        diameter_px = um_to_px(diameter, pixel_size_um)

    if progress_queue:
        progress_queue.put(("progress", 5.0))

    stack, n_total = read_czi_frames(
        czi_path, channel, frame_start, frame_end, frame_step
    )
    n_frames = stack.shape[0]
    
    if progress_queue:
        progress_queue.put(("progress", 15.0))

    if selected_labels is None:
        # DETECT MODE
        labels, vesicles = segment_frame(
            stack[0],
            method=method,
            use_cellpose=use_cellpose,
            model_type=model_type,
            diameter=diameter_px,
            min_area=min_area_px,
            min_radius=min_radius_px,
            max_radius=max_radius_px,
            radius_step=radius_step_px,
            canny_sigma=canny_sigma,
            hough_min_distance=hough_min_distance_px,
            hough_threshold_fraction=hough_threshold_fraction,
            cellpose_gpu=cellpose_gpu,
            cellpose_invert=cellpose_invert,
            filter_circularity=filter_circularity,
            filter_eccentricity=filter_eccentricity,
            filter_solidity=filter_solidity,
            preprocess_transmitted=preprocess_transmitted,
            fit_circles=fit_circles,
        )

        # add µm info to each vesicle for display
        for v in vesicles:
            v["centroid_y_um"] = v["centroid_y"] * pixel_size_um
            v["centroid_x_um"] = v["centroid_x"] * pixel_size_um
            if "radius" in v:
                v["radius_um"] = v["radius"] * pixel_size_um
            v["equivalent_diameter_um"] = v["equivalent_diameter"] * pixel_size_um
            v["area_um2"] = v["area"] * (pixel_size_um ** 2)

        if progress_queue:
            progress_queue.put(("progress", 100.0))

        return {
            "mode": "detect",
            "vesicles": vesicles,
            "labels_frame0": labels,
            "preview_frame": stack[0],
            "n_total_frames": n_total,
            "n_selected_frames": n_frames,
            "czi_path": str(czi_path),
            "pixel_size_um": pixel_size_um,
        }

    else:
        # CROP MODE
        czi_stem = Path(czi_path).stem
        out_dir = Path(czi_path).parent / f"{czi_stem}_vesicle_crops"
        out_dir.mkdir(exist_ok=True)

        vesicle_stacks = {lbl: [] for lbl in selected_labels}

        for fi in range(n_frames):
            if cancel_event is not None and cancel_event.is_set():
                return {"mode": "cancelled"}

            frame = stack[fi]

            labels, vesicles = segment_frame(
                frame,
                method=method,
                use_cellpose=use_cellpose,
                model_type=model_type,
                diameter=diameter_px,
                min_area=min_area_px,
                min_radius=min_radius_px,
                max_radius=max_radius_px,
                radius_step=radius_step_px,
                canny_sigma=canny_sigma,
                hough_min_distance=hough_min_distance_px,
                hough_threshold_fraction=hough_threshold_fraction,
                cellpose_gpu=cellpose_gpu,
                cellpose_invert=cellpose_invert,
            )

            for sel_lbl in selected_labels:
                best = _find_best_match(sel_lbl, vesicles, labels, frame, crop_margin_px)
                if best is not None:
                    vesicle_stacks[sel_lbl].append(best)
                else:
                    vesicle_stacks[sel_lbl].append(
                        np.zeros((2 * crop_margin_px, 2 * crop_margin_px), dtype=frame.dtype)
                    )

            if progress_queue:
                pct = 15.0 + 80.0 * ((fi + 1) / n_frames)
                progress_queue.put(("progress", pct))

        output_paths = []
        for lbl in selected_labels:
            arr = np.stack(vesicle_stacks[lbl], axis=0)
            out_path = str(out_dir / f"vesicle_{lbl}.tif")
            tifffile.imwrite(out_path, arr, photometric="minisblack")
            output_paths.append(out_path)

        if progress_queue:
            progress_queue.put(("progress", 100.0))

        return {
            "mode": "crop",
            "output_paths": output_paths,
            "output_dir": str(out_dir),
            "n_vesicles": len(selected_labels),
            "n_frames": n_frames,
            "czi_path": str(czi_path),
            "pixel_size_um": pixel_size_um,
        }


# store frame-0 centroids globally within one run (simple approach)
_frame0_centroids: Dict[int, Tuple[float, float]] = {}


def _find_best_match(
    target_label: int,
    vesicles: List[Dict],
    labels: np.ndarray,
    frame: np.ndarray,
    crop_margin: int,
) -> Optional[np.ndarray]:
    """
    Find the vesicle in the current frame closest to the target's frame-0 centroid.
    Crop and return it.
    """
    global _frame0_centroids

    if not vesicles:
        return None

    # if we haven't stored the target centroid yet, store it from the first call
    if target_label not in _frame0_centroids:
        # try to find it in current vesicles list
        for v in vesicles:
            if v["label"] == target_label:
                _frame0_centroids[target_label] = (v["centroid_y"], v["centroid_x"])
                break
        else:
            # label not found in this frame; can't match
            return None

    target_cy, target_cx = _frame0_centroids[target_label]

    # find closest vesicle by centroid distance
    best_v = None
    best_dist = float("inf")
    for v in vesicles:
        dy = v["centroid_y"] - target_cy
        dx = v["centroid_x"] - target_cx
        dist = np.sqrt(dy**2 + dx**2)
        if dist < best_dist:
            best_dist = dist
            best_v = v

    if best_v is None:
        return None

    # update centroid for tracking across frames
    _frame0_centroids[target_label] = (best_v["centroid_y"], best_v["centroid_x"])

    return crop_square(frame, best_v["centroid_y"], best_v["centroid_x"], crop_margin)



def straighten_membrane(
    image: np.ndarray,
    center_y: float,
    center_x: float,
    radius: float,
    thickness_px: int,
    n_angle_points: Optional[int] = None,
) -> np.ndarray:
    """
    Unroll an annular band around a circle into a rectangular strip.

    Parameters
    ----------
    image : 2D array (Y, X)
    center_y, center_x : circle center in pixels
    radius : circle radius in pixels
    thickness_px : full thickness of the annular band in pixels
    n_angle_points : number of angular samples (default: circumference in pixels)

    Returns
    -------
    strip : 2D array (thickness_px, n_angle_points)
        Row 0 = inner edge of annulus
        Last row = outer edge of annulus
        Columns = angular position (0° to 360°)
    """
    from scipy.ndimage import map_coordinates

    half_t = thickness_px / 2.0

    # number of angular samples: default to circumference
    if n_angle_points is None:
        n_angle_points = max(10, int(round(2.0 * np.pi * radius)))

    # angles from 0 to 2π
    angles = np.linspace(0, 2 * np.pi, n_angle_points, endpoint=False)

    # radial positions from (R - half_thickness) to (R + half_thickness)
    radii = np.linspace(radius - half_t, radius + half_t, thickness_px)

    # build coordinate grids
    # shape: (thickness_px, n_angle_points)
    r_grid, a_grid = np.meshgrid(radii, angles, indexing='ij')

    # convert polar to cartesian
    y_coords = center_y + r_grid * np.sin(a_grid)
    x_coords = center_x + r_grid * np.cos(a_grid)

    # sample the image using bilinear interpolation
    coords = np.array([y_coords.ravel(), x_coords.ravel()])
    strip = map_coordinates(image.astype(np.float64), coords, order=1, mode='constant', cval=0.0)
    strip = strip.reshape(thickness_px, n_angle_points)

    return strip


def straighten_vesicle_timeseries(
    czi_path: str,
    channel: int,
    center_y: float,
    center_x: float,
    radius: float,
    thickness_px: int,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_step: int = 1,
    n_angle_points: Optional[int] = None,
    progress_queue=None,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Straighten membrane for all selected frames.

    Returns dict with:
      - strips: 3D array (n_frames, thickness_px, n_angle_points)
      - intensity_profile: 2D array (n_frames, n_angle_points) — mean across thickness
      - total_intensity: 1D array (n_frames,) — total membrane intensity per frame
      - angles_deg: 1D array (n_angle_points,) — angular positions in degrees
    """
    stack, n_total = read_czi_frames(czi_path, channel, frame_start, frame_end, frame_step)
    n_frames = stack.shape[0]

    if n_angle_points is None:
        n_angle_points = max(10, int(round(2.0 * np.pi * radius)))

    strips = np.zeros((n_frames, thickness_px, n_angle_points), dtype=np.float64)
    intensity_profile = np.zeros((n_frames, n_angle_points), dtype=np.float64)
    total_intensity = np.zeros(n_frames, dtype=np.float64)

    for fi in range(n_frames):
        if cancel_event is not None and cancel_event.is_set():
            return {"mode": "cancelled"}

        strip = straighten_membrane(
            stack[fi], center_y, center_x, radius, thickness_px, n_angle_points
        )
        strips[fi] = strip
        intensity_profile[fi] = np.mean(strip, axis=0)
        total_intensity[fi] = np.sum(strip)

        if progress_queue:
            pct = 100.0 * (fi + 1) / n_frames
            progress_queue.put(("progress", pct))

    angles_deg = np.linspace(0, 360, n_angle_points, endpoint=False)

    return {
        "mode": "straighten",
        "strips": strips,
        "intensity_profile": intensity_profile,
        "total_intensity": total_intensity,
        "angles_deg": angles_deg,
        "n_frames": n_frames,
        "thickness_px": thickness_px,
        "n_angle_points": n_angle_points,
        "center_y": center_y,
        "center_x": center_x,
        "radius": radius,
        "channel": channel,
        "czi_path": str(czi_path),
    }

def _fit_circle_to_mask(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Given a binary mask (e.g. a membrane arc from Cellpose),
    fit a circle to the mask pixels using least-squares.
    
    Returns dict with centroid_y, centroid_x, radius, or None if fitting fails.
    """
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return None
    
    # least-squares circle fit
    # minimize sum of (sqrt((x-cx)^2 + (y-cy)^2) - r)^2
    # using algebraic method: fit to x^2 + y^2 + Dx + Ey + F = 0
    # where cx = -D/2, cy = -E/2, r = sqrt(cx^2 + cy^2 - F)
    
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    
    cx = result[0] / 2.0
    cy = result[1] / 2.0
    r_squared = cx**2 + cy**2 + result[2]
    
    if r_squared <= 0:
        return None
    
    r = np.sqrt(r_squared)
    
    if r < 5 or r > max(mask.shape):
        return None
    
    return {
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "radius": float(r),
    }