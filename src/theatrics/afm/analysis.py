from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from pathlib import Path

try:
    from AFMReader.jpk import load_jpk
    AFMREADER_AVAILABLE = True
except ImportError:
    AFMREADER_AVAILABLE = False


# ────────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────────

def load_jpk_qi(filepath: str,
                channel: str = 'height_trace') -> tuple[np.ndarray, float]:
    """
    Load a .jpk-qi-image file.
    Returns (height_nm, pixel_size_nm).
    """
    if not AFMREADER_AVAILABLE:
        raise RuntimeError(
            "AFMReader is not installed. "
            "Install it with:  pip install AFMReader"
        )
    height_nm, pixel_size_nm = load_jpk(
        filepath, channel=channel, flip_image=True
    )
    return height_nm, float(pixel_size_nm)


def get_file_info(height_nm: np.ndarray,
                  pixel_size_nm: float) -> dict:
    """Return a dict of human-readable metadata about the loaded file."""
    return {
        "shape":         height_nm.shape,
        "pixel_size_nm": pixel_size_nm,
        "scan_size_um":  height_nm.shape[1] * pixel_size_nm / 1000.0,
        "height_min_nm": float(height_nm.min()),
        "height_max_nm": float(height_nm.max()),
    }


# ────────────────────────────────────────────────────────────────
# Profile extraction
# ────────────────────────────────────────────────────────────────

def extract_line_profile(height_data: np.ndarray,
                         pixel_size_nm: float,
                         start_px: tuple[float, float],
                         end_px:   tuple[float, float],
                         n_points: int = 300
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the height map along a straight line between two pixel
    coordinates using bilinear interpolation.

    Parameters
    ----------
    height_data   : 2D array (rows = y, cols = x)
    pixel_size_nm : nm per pixel
    start_px      : (x_px, y_px)  start in pixel coordinates
    end_px        : (x_px, y_px)  end   in pixel coordinates
    n_points      : number of sample points along the line

    Returns
    -------
    distances_nm : 1D array, distance along line in nm
    heights_nm   : 1D array, height at each point in nm
    """
    x0, y0 = start_px
    x1, y1 = end_px
    x_coords = np.linspace(x0, x1, n_points)
    y_coords = np.linspace(y0, y1, n_points)
    heights  = ndimage.map_coordinates(
        height_data, [y_coords, x_coords], order=1
    )
    total_dist_nm = (
        np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) * pixel_size_nm
    )
    distances = np.linspace(0, total_dist_nm, n_points)
    return distances, heights


# ────────────────────────────────────────────────────────────────
# Baseline correction
# ────────────────────────────────────────────────────────────────

def compute_baseline_and_correction(
        distances_nm: np.ndarray,
        heights_nm:   np.ndarray,
        h_start:      float,
        h_end:        float,
        smooth_window: int = 15,
        smooth_poly:   int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Savitzky-Golay smoothing then subtract a linear baseline
    defined by the two endpoint heights clicked by the user.

    Returns
    -------
    h_smooth : smoothed raw profile (nm)
    baseline : linear baseline (nm)
    h_adj    : baseline-corrected profile — membrane plane = 0 (nm)
    """
    n = len(heights_nm)

    # smoothing
    if n > smooth_window:
        h_smooth = savgol_filter(heights_nm, smooth_window, smooth_poly)
    else:
        h_smooth = heights_nm.copy()

    # linear baseline from the two clicked endpoint heights
    baseline = (
        h_start
        + (h_end - h_start) * distances_nm / distances_nm[-1]
    )
    h_adj = h_smooth - baseline

    return h_smooth, baseline, h_adj


# ────────────────────────────────────────────────────────────────
# Contact point detection and wetting angle
# ────────────────────────────────────────────────────────────────

def find_contact_and_angles(
        distances_nm: np.ndarray,
        h_adj:        np.ndarray,
        n_fit_points: int = 15,
        threshold_fraction: float = 0.05,
) -> dict | None:
    """
    Find left and right contact points and compute wetting angles.

    Contact point: first index from each side where h_adj exceeds
    threshold_fraction * peak_height.

    Wetting angle: slope of a line fitted through n_fit_points
    starting at the contact point and going into the condensate,
    expressed as arctan(|slope|) in degrees from the membrane plane.

    Returns None if contact points cannot be located.
    """
    n        = len(h_adj)
    peak_idx = int(np.argmax(h_adj))
    h_peak   = float(h_adj[peak_idx])

    if h_peak <= 0:
        return None

    threshold = threshold_fraction * h_peak

    # left contact — first point from left above threshold
    left_contact = None
    for i in range(peak_idx):
        if h_adj[i] > threshold:
            left_contact = i
            break

    # right contact — first point from right above threshold
    right_contact = None
    for i in range(n - 1, peak_idx, -1):
        if h_adj[i] > threshold:
            right_contact = i
            break

    if left_contact is None or right_contact is None:
        return None

    def _slope_at(ci: int, direction: int, npts: int) -> float:
        """
        Fit a line through npts points starting at ci going in
        direction (+1 = right, -1 = left) into the condensate.
        """
        idx = np.arange(ci, ci + direction * npts, direction)
        idx = idx[(idx >= 0) & (idx < n)]
        if len(idx) < 2:
            return float("nan")
        return float(np.polyfit(distances_nm[idx], h_adj[idx], 1)[0])

    left_slope  = _slope_at(left_contact,  +1, n_fit_points)
    right_slope = _slope_at(right_contact, -1, n_fit_points)

    result = {
        "left_contact":  left_contact,
        "right_contact": right_contact,
        "n_fit_points":  n_fit_points,
        "h_peak":        h_peak,
        "threshold":     threshold,
    }

    if not np.isnan(left_slope):
        result["theta_left_deg"]  = float(
            np.degrees(np.arctan(abs(left_slope)))
        )
        result["left_slope"]      = left_slope
        result["left_contact_nm"] = float(distances_nm[left_contact])

    if not np.isnan(right_slope):
        result["theta_right_deg"]  = float(
            np.degrees(np.arctan(abs(right_slope)))
        )
        result["right_slope"]      = right_slope
        result["right_contact_nm"] = float(distances_nm[right_contact])

    if "theta_left_deg" in result and "theta_right_deg" in result:
        result["theta_mean_deg"] = float(np.mean([
            result["theta_left_deg"],
            result["theta_right_deg"],
        ]))

    return result


# ────────────────────────────────────────────────────────────────
# Profile height / width measurements
# ────────────────────────────────────────────────────────────────

def measure_profile(distances_nm: np.ndarray,
                    h_adj:        np.ndarray,
                    angles:       dict | None) -> dict:
    """
    Compute simple scalar measurements from a corrected profile.

    Returns
    -------
    dict with peak_height_nm, fwhm_nm, contact_width_nm, and
    the wetting angles (if available).
    """
    peak_height = float(np.max(h_adj))

    # FWHM via half-max crossing
    half_max = peak_height / 2.0
    above    = np.where(h_adj >= half_max)[0]
    if len(above) >= 2:
        fwhm_nm = float(
            distances_nm[above[-1]] - distances_nm[above[0]]
        )
    else:
        fwhm_nm = float("nan")

    measurements = {
        "peak_height_nm": peak_height,
        "fwhm_nm":        fwhm_nm,
    }

    if angles is not None:
        lc = angles.get("left_contact_nm",  float("nan"))
        rc = angles.get("right_contact_nm", float("nan"))
        measurements["contact_width_nm"]  = float(rc - lc) if (
            not np.isnan(lc) and not np.isnan(rc)
        ) else float("nan")
        measurements["theta_left_deg"]    = angles.get(
            "theta_left_deg",  float("nan")
        )
        measurements["theta_right_deg"]   = angles.get(
            "theta_right_deg", float("nan")
        )
        measurements["theta_mean_deg"]    = angles.get(
            "theta_mean_deg",  float("nan")
        )

    return measurements


# ────────────────────────────────────────────────────────────────
# Full single-profile pipeline
# ────────────────────────────────────────────────────────────────

def process_profile(height_data:   np.ndarray,
                    pixel_size_nm: float,
                    start_px:      tuple[float, float],
                    end_px:        tuple[float, float],
                    h_start_nm:    float,
                    h_end_nm:      float,
                    n_fit_points:  int   = 15,
                    smooth_window: int   = 15,
                    smooth_poly:   int   = 3,
                    n_points:      int   = 300,
                    threshold_fraction: float = 0.05,
                    ) -> dict:
    """
    Run the complete pipeline for one profile and return a result dict
    that can be serialised and passed through the multiprocessing Queue.

    All arrays are converted to plain Python lists so they pass through
    the Queue without pickle issues on all platforms.
    """
    distances, heights = extract_line_profile(
        height_data, pixel_size_nm, start_px, end_px, n_points
    )

    h_smooth, baseline, h_adj = compute_baseline_and_correction(
        distances, heights,
        h_start=h_start_nm, h_end=h_end_nm,
        smooth_window=smooth_window, smooth_poly=smooth_poly,
    )

    angles = find_contact_and_angles(
        distances, h_adj,
        n_fit_points=n_fit_points,
        threshold_fraction=threshold_fraction,
    )

    measurements = measure_profile(distances, h_adj, angles)

    return {
        # arrays as lists for Queue serialisation
        "distances_nm": distances.tolist(),
        "heights_raw":  heights.tolist(),
        "h_smooth":     h_smooth.tolist(),
        "baseline":     baseline.tolist(),
        "h_adj":        h_adj.tolist(),
        # scalars
        "h_start_nm":   h_start_nm,
        "h_end_nm":     h_end_nm,
        "pixel_size_nm": pixel_size_nm,
        "start_px":     start_px,
        "end_px":       end_px,
        "n_fit_points": n_fit_points,
        # derived
        "angles":       angles,
        "measurements": measurements,
    }