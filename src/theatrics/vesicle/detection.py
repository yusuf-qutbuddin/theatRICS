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
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny, peak_local_max
from skimage.segmentation import watershed
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
# Debug utilities
# ---------------------------------------------------------------------------

_DEBUG_SAVE = False          # set to True to enable debug saving
_DEBUG_DIR: Optional[str] = None   # will be set automatically from czi_path


def enable_debug_saving(czi_path: str) -> str:
    """
    Enable debug image saving.
    Creates a folder next to the CZI file called '<stem>_debug/'.
    Returns the debug folder path.
    """
    global _DEBUG_SAVE, _DEBUG_DIR
    debug_dir = str(Path(czi_path).parent / f"{Path(czi_path).stem}_debug")
    Path(debug_dir).mkdir(exist_ok=True)
    _DEBUG_SAVE = True
    _DEBUG_DIR = debug_dir
    return debug_dir


def disable_debug_saving():
    """Disable debug image saving."""
    global _DEBUG_SAVE
    _DEBUG_SAVE = False


def save_debug_image(image: np.ndarray, name: str, normalize: bool = True):
    """
    Save a 2D array as a TIFF in the debug folder.

    Parameters
    ----------
    image : 2D or 3D array to save
    name : filename without extension (e.g. '01_raw', '02_blurred')
    normalize : if True, normalize to uint16 range for viewing
    """
    global _DEBUG_SAVE, _DEBUG_DIR

    if not _DEBUG_SAVE or _DEBUG_DIR is None:
        return

    out_path = str(Path(_DEBUG_DIR) / f"{name}.tif")

    img = np.squeeze(image).astype(np.float64)

    if normalize and img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 65535).astype(np.uint16)
    elif normalize:
        img = np.zeros_like(img, dtype=np.uint16)
    else:
        # save raw values as float32
        img = img.astype(np.float32)

    tifffile.imwrite(out_path, img)
    print(f"[DEBUG] Saved: {out_path}")








# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def threshold_huang(image: np.ndarray) -> float:
    """
    Huang's fuzzy thresholding method.
    Python implementation of the ImageJ Huang method.
    
    Reference:
    Huang L-K and Wang M-J J (1995) Image thresholding by minimizing
    the measures of fuzziness. Pattern Recognition 28(1): 41-51.
    """
    # build histogram
    hist, bin_edges = np.histogram(image.ravel(), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    n = len(hist)
    total = hist.sum()

    # cumulative sums
    cumsum = np.cumsum(hist)
    cumsum_val = np.cumsum(hist * bin_centers)

    # normalize
    mu = cumsum_val / np.maximum(cumsum, 1)

    # for each threshold t, compute the fuzzy entropy
    img_min = bin_centers[0]
    img_max = bin_centers[-1]
    img_range = img_max - img_min

    if img_range == 0:
        return float(img_min)

    best_t = 0
    min_entropy = np.inf

    for t in range(n):
        # background mean
        if cumsum[t] > 0:
            mu_b = cumsum_val[t] / cumsum[t]
        else:
            mu_b = img_min

        # foreground mean
        if total - cumsum[t] > 0:
            mu_f = (cumsum_val[-1] - cumsum_val[t]) / (total - cumsum[t])
        else:
            mu_f = img_max

        # fuzzy entropy: sum over all pixels
        entropy = 0.0
        for i in range(n):
            if hist[i] == 0:
                continue
            x = bin_centers[i]

            # membership to background
            if mu_b != img_min:
                mu_bg = 1.0 / (1.0 + abs(x - mu_b) / (img_range))
            else:
                mu_bg = 1.0 if x == img_min else 0.0

            # membership to foreground
            if mu_f != img_max:
                mu_fg = 1.0 / (1.0 + abs(x - mu_f) / (img_range))
            else:
                mu_fg = 1.0 if x == img_max else 0.0

            # Shannon entropy contribution
            def h(mu_val):
                if mu_val <= 0 or mu_val >= 1:
                    return 0.0
                return -mu_val * np.log(mu_val) - (1 - mu_val) * np.log(1 - mu_val)

            entropy += hist[i] * (h(mu_bg) + h(mu_fg))

        if entropy < min_entropy:
            min_entropy = entropy
            best_t = t

    return float(bin_centers[best_t])
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

def segment_weighted_intensity(
    image: np.ndarray,
    min_radius_px: int = 50,
    max_radius_px: int = 500,
    search_range_px: int = 10,
    min_circularity: float = 0.60,
    max_circularity: float = 1.00,
    threshold_method: str = "huang",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detect GUVs using the weighted peripheral intensity method from
    Kohyama et al., Nature Communications 2022.

    This is the Python equivalent of the ImageJ macro approach:
    1. Threshold the image to find rough GUV candidates
    2. For each candidate, scan positions and diameters to find
       the circle that maximizes total peripheral intensity
       (mean × nPixels along the circumference)
    3. Return the best-fit circles

    This method works well for:
    - Transmitted light images
    - Fluorescence membrane images (bright ring)
    - Fluorescence interior images (bright interior)

    Parameters
    ----------
    image : 2D array (Y, X)
    min_radius_px : minimum GUV radius in pixels
    max_radius_px : maximum GUV radius in pixels
    search_range_px : how many pixels around each detected region to scan
    min_circularity : minimum circularity for initial detection (0-1)
    max_circularity : maximum circularity for initial detection (0-1)
    threshold_method : thresholding method for initial detection

    Returns
    -------
    labels : 2D int array (Y, X), each circle filled with unique label
    circles : list of dicts with keys:
              label, centroid_y, centroid_x, radius, area, equivalent_diameter
    """
    from scipy.ndimage import map_coordinates

    img = image.astype(np.float64)

    # normalize
    img_norm = img - img.min()
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()
    save_debug_image(img_norm, "01_raw_normalized")
    h, w = image.shape[:2]

    # ── Step 1: coarse detection by thresholding ──
    if threshold_method == "huang":
        thresh = threshold_huang(img_norm)
    elif threshold_method == "yen":
        from skimage.filters import threshold_yen
        thresh = threshold_yen(img_norm)
    elif threshold_method == "triangle":
        from skimage.filters import threshold_triangle
        thresh = threshold_triangle(img_norm)
    elif threshold_method == "mean":
        from skimage.filters import threshold_mean
        thresh = threshold_mean(img_norm)
    elif threshold_method == "li":
        from skimage.filters import threshold_li
        thresh = threshold_li(img_norm)
    else:
        thresh = threshold_otsu(img_norm)

    binary = img_norm > thresh
    binary = binary_opening(binary, morph_disk(3))
    labeled_coarse = label(binary)
    save_debug_image(binary.astype(np.uint8) * 255, "02_binary_after_threshold", normalize=False)
    save_debug_image(labeled_coarse.astype(np.float32), "03_labeled_coarse", normalize=True)
     # ── Step 2: build candidates from bright pixel clusters ──
    candidates = []

    # Strategy A: try standard region-based candidate finding first
    # (works well for filled/large regions)
    for prop in regionprops(labeled_coarse):
        r_equiv = prop.equivalent_diameter / 2.0
        if r_equiv < min_radius_px or r_equiv > max_radius_px:
            continue
        if prop.perimeter > 0:
            circ = 4.0 * np.pi * prop.area / (prop.perimeter ** 2)
        else:
            circ = 0.0
        if not (min_circularity <= circ <= max_circularity):
            continue
        min_row, min_col, max_row, max_col = prop.bbox
        candidates.append({
            "x": min_col,
            "y": min_row,
            "w": max_col - min_col,
            "h": max_row - min_row,
        })

    # Strategy B: if no candidates found, use bounding boxes of
    # spatially clustered bright pixels
    # This handles thin membrane arcs which are individually small
    if not candidates:
        from skimage.morphology import binary_dilation, binary_erosion, disk as dsk
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from scipy import ndimage as sci_ndimage
        from skimage.filters import gaussian

        # ── B1: dilate membrane to close gaps in arcs ──
        dilation_radius = max(3, min_radius_px // 8)
        binary_dilated = binary_dilation(binary, dsk(dilation_radius))
        save_debug_image(
            binary_dilated.astype(np.uint8) * 255,
            "05_binary_dilated", normalize=False
        )

        # ── B2: fill the interior of each ring ──
        # flood-fill from the border: anything reachable from the border
        # without crossing a membrane pixel is "exterior"
        # everything else is "interior" of a GUV
        from scipy.ndimage import binary_fill_holes

        # invert: background becomes foreground
        inverted = ~binary_dilated

        # label connected components of the inverted image
        labeled_inv = label(inverted)

        # find the label of the border-connected region
        # (it touches all 4 edges)
        border_label = labeled_inv[0, 0]

        # interior = everything that is NOT background and NOT membrane
        interior = (labeled_inv > 0) & (labeled_inv != border_label)

        save_debug_image(
            interior.astype(np.uint8) * 255,
            "06_interior", normalize=False
        )

        # ── B3: distance transform of interior ──
        # now peaks are at the centers of GUV interiors
        distance = sci_ndimage.distance_transform_edt(interior)
        save_debug_image(
            distance.astype(np.float32),
            "07_distance_interior", normalize=True
        )

        # smooth to avoid many local maxima
        distance_smooth = gaussian(distance, sigma=max(3, min_radius_px // 10))

        # ── B4: find peaks = GUV centers ──
        min_peak_distance = max(min_radius_px, 10)
        coords = peak_local_max(
            distance_smooth,
            min_distance=min_peak_distance,
            labels=interior,
        )

        save_debug_image(
            distance_smooth.astype(np.float32),
            "08_distance_smooth", normalize=True
        )

        if len(coords) == 0:
            # fallback: use whole image
            candidates.append({"x": 0, "y": 0, "w": w, "h": h})
        else:
            # ── B5: for each peak, estimate radius and build bounding box ──
            for (cy_peak, cx_peak) in coords:
                # estimate radius from the distance value at this peak
                # distance_transform_edt gives distance to nearest background
                # for a circular interior, the peak value ≈ interior radius
                r_est = float(distance_smooth[cy_peak, cx_peak])

                # clamp to user's range
                r_est = max(min_radius_px, min(max_radius_px, r_est))

                # build bounding box with some margin
                margin = int(r_est * 0.2)
                x0 = max(0, int(cx_peak - r_est) - margin)
                y0 = max(0, int(cy_peak - r_est) - margin)
                x1 = min(w, int(cx_peak + r_est) + margin)
                y1 = min(h, int(cy_peak + r_est) + margin)

                candidates.append({
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                })

        # deduplicate overlapping candidates
        # (two peaks that are very close → one candidate)
        deduped = []
        for c in candidates:
            cx = c["x"] + c["w"] / 2
            cy = c["y"] + c["h"] / 2
            too_close = False
            for d in deduped:
                dx_c = d["x"] + d["w"] / 2
                dy_c = d["y"] + d["h"] / 2
                if np.sqrt((cx - dx_c)**2 + (cy - dy_c)**2) < min_radius_px:
                    too_close = True
                    break
            if not too_close:
                deduped.append(c)
        candidates = deduped

    # Strategy C: last resort — use whole image as one candidate
    if not candidates:
        candidates.append({
            "x": 0,
            "y": 0,
            "w": w,
            "h": h,
        })

    # ── Step 2: precompute angle arrays ──
    n_points = 360
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cos_a = np.cos(angles)  # shape (n_points,)
    sin_a = np.sin(angles)  # shape (n_points,)

    def peripheral_intensity_vectorized(img2d, cx_arr, cy_arr, r_arr):
        """
        Vectorized: compute peripheral intensity score for many circles at once.

        Parameters
        ----------
        cx_arr, cy_arr : 1D arrays of circle centers (n_circles,)
        r_arr : 1D array of radii (n_circles,)

        Returns
        -------
        scores : 1D array of mean*n_points scores (n_circles,)
        """
        n_circles = len(cx_arr)

        # xs[i, j] = x coordinate of circle i at angle j
        # shape: (n_circles, n_points)
        xs = cx_arr[:, None] + r_arr[:, None] * cos_a[None, :]
        ys = cy_arr[:, None] + r_arr[:, None] * sin_a[None, :]

        xs = np.clip(xs, 0, img2d.shape[1] - 1)
        ys = np.clip(ys, 0, img2d.shape[0] - 1)

        # bilinear interpolation manually (faster than map_coordinates in a loop)
        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, img2d.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, img2d.shape[0] - 1)

        dx = xs - x0
        dy = ys - y0

        # bilinear interpolation: 4 neighbors
        v00 = img2d[y0, x0]
        v01 = img2d[y0, x1]
        v10 = img2d[y1, x0]
        v11 = img2d[y1, x1]

        values = (v00 * (1 - dx) * (1 - dy) +
                  v01 * dx * (1 - dy) +
                  v10 * (1 - dx) * dy +
                  v11 * dx * dy)

        # values shape: (n_circles, n_points)
        # score = mean * n_points = sum
        scores = values.sum(axis=1)
        return scores

    # ── Step 3: precompute trig arrays ──
    n_points = 360
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    def score_circles_batch(img2d, cx_arr, cy_arr, r_arr):
        """
        Score many circles at once using vectorized bilinear interpolation.
        Returns 1D array of scores (sum of intensities along circumference).
        """
        n = len(cx_arr)
        # shape (n, n_points)
        xs = cx_arr[:, None] + r_arr[:, None] * cos_a[None, :]
        ys = cy_arr[:, None] + r_arr[:, None] * sin_a[None, :]

        xs = np.clip(xs, 0, img2d.shape[1] - 1)
        ys = np.clip(ys, 0, img2d.shape[0] - 1)

        # bilinear interpolation
        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, img2d.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, img2d.shape[0] - 1)

        dx = xs - x0
        dy = ys - y0

        v00 = img2d[y0, x0]
        v01 = img2d[y0, x1]
        v10 = img2d[y1, x0]
        v11 = img2d[y1, x1]

        values = (v00 * (1 - dx) * (1 - dy) +
                  v01 * dx       * (1 - dy) +
                  v10 * (1 - dx) * dy       +
                  v11 * dx       * dy)

        return values.sum(axis=1)

    # ── Step 4: for each candidate, do a focused scan ──
    circles = []
    labels_out = np.zeros((h, w), dtype=np.int32)
    label_counter = 0

    for cand in candidates:
        x0 = cand["x"]
        y0 = cand["y"]
        cw = cand["w"]
        ch = cand["h"]

        # Estimate center and radius from candidate bounding box
        cx_est = x0 + cw / 2.0
        cy_est = y0 + ch / 2.0
        r_est = min(cw, ch) / 2.0

        # clamp radius estimate to user's range
        r_est = np.clip(r_est, min_radius_px, max_radius_px)

        # ── Focused search: scan a grid around the estimated center ──
        # Instead of scanning the full bounding box, scan only a small
        # window around the estimated center and a small range of radii.

        # center search range: ± search_range_px pixels
        center_search = int(search_range_px)

        # radius search range: ± search_range_px pixels around estimate
        r_min_search = max(min_radius_px, r_est - search_range_px)
        r_max_search = min(max_radius_px, r_est + search_range_px)

        # step sizes
        # use 1 pixel step for center, 0.5 pixel for radius
        center_step = max(1, search_range_px // 10)
        r_step = max(0.5, search_range_px / 20.0)

        # build search grids
        cx_range = np.arange(
            cx_est - center_search,
            cx_est + center_search + center_step,
            center_step,
        )
        cy_range = np.arange(
            cy_est - center_search,
            cy_est + center_search + center_step,
            center_step,
        )
        r_range = np.arange(r_min_search, r_max_search + r_step, r_step)

        # clamp to image
        cx_range = cx_range[(cx_range >= 0) & (cx_range < w)]
        cy_range = cy_range[(cy_range >= 0) & (cy_range < h)]
        r_range = r_range[(r_range >= min_radius_px) & (r_range <= max_radius_px)]

        if len(cx_range) == 0 or len(cy_range) == 0 or len(r_range) == 0:
            continue

        # build all combinations
        cx_grid, cy_grid, r_grid = np.meshgrid(cx_range, cy_range, r_range)
        cx_flat = cx_grid.ravel()
        cy_flat = cy_grid.ravel()
        r_flat = r_grid.ravel()

        n_circles = len(cx_flat)

        # score in chunks to control memory
        chunk_size = 10000
        best_score = -np.inf
        best_idx = 0

        for start in range(0, n_circles, chunk_size):
            end = min(start + chunk_size, n_circles)
            scores = score_circles_batch(
                img_norm,
                cx_flat[start:end],
                cy_flat[start:end],
                r_flat[start:end],
            )
            local_best = int(np.argmax(scores))
            if scores[local_best] > best_score:
                best_score = float(scores[local_best])
                best_idx = start + local_best

        best_cx = float(cx_flat[best_idx])
        best_cy = float(cy_flat[best_idx])
        best_r = float(r_flat[best_idx])

        # store result
        label_counter += 1
        yy, xx = np.ogrid[:h, :w]
        circle_mask = ((yy - best_cy) ** 2 + (xx - best_cx) ** 2) <= best_r ** 2
        fill_mask = circle_mask & (labels_out == 0)
        labels_out[fill_mask] = label_counter

        area = int(np.sum(circle_mask))
        circles.append({
            "label": label_counter,
            "centroid_y": float(best_cy),
            "centroid_x": float(best_cx),
            "radius": int(round(best_r)),
            "bbox": (
                max(0, int(best_cy - best_r)),
                max(0, int(best_cx - best_r)),
                min(h, int(best_cy + best_r)),
                min(w, int(best_cx + best_r)),
            ),
            "area": area,
            "equivalent_diameter": float(2 * best_r),
            "peripheral_score": float(best_score),
        })

    return labels_out, circles




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
        vesicles = extract_vesicle_info(masks)
        return masks, vesicles


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
    weight_search_range: float = 2.0,
    threshold_method: str = "huang",
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
                labels, vesicles = segment_cellpose(
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
                return labels, vesicles
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
    elif method == "weighted_intensity":
        labels, vesicles = segment_weighted_intensity(
            image,
            min_radius_px=min_radius,
            max_radius_px=max_radius,
            search_range_px=weight_search_range,
            min_circularity=filter_circularity if filter_circularity > 0 else 0.60,
            max_circularity=1.0,
            threshold_method = threshold_method,
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
    weight_search_range: float = 2.0,
    threshold_method: str = 'huang',
    selected_labels: Optional[List[int]] = None,
    progress_queue=None,
    cancel_event=None,
    debug: bool = False,
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
    # right after pixel_size_um is confirmed valid:
    if debug:
        debug_dir = enable_debug_saving(czi_path)
       
        
    # Convert µm → pixels
    crop_margin_px = um_to_px(crop_margin_um, pixel_size_um)
    min_radius_px = um_to_px(min_radius_um, pixel_size_um)
    max_radius_px = um_to_px(max_radius_um, pixel_size_um)
    radius_step_px = max(1, um_to_px(radius_step_um, pixel_size_um))
    hough_min_distance_px = um_to_px(hough_min_distance_um, pixel_size_um)
    min_area_px = max(1, int(round(min_area_um2 / (pixel_size_um ** 2))))
    weight_search_range_px = um_to_px(weight_search_range, pixel_size_um)
   
    
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
            weight_search_range = weight_search_range_px,
            threshold_method=threshold_method,
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
                weight_search_range = weight_search_range_px,
                threshold_method=threshold_method,
                cellpose_gpu=cellpose_gpu,
                cellpose_invert=cellpose_invert,
                filter_circularity=filter_circularity,
                filter_eccentricity=filter_eccentricity,
                filter_solidity=filter_solidity,
                preprocess_transmitted=preprocess_transmitted,
                fit_circles=fit_circles,
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