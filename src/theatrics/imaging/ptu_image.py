"""
theatrics/imaging/ptu_image.py

Extract intensity images and FLIM (mean lifetime) images from
PicoQuant PTU CLSM files.

Faithful to the tttrlib examples:
  - microtime_histogram[0] is the counts array
  - TCSPC x-axis is in bin units throughout
  - CLSMImage is constructed with fill=True where possible
  - PIE windows are specified in raw TCSPC bin units
  - n_micro_time_bins is derived from micro_times.max()+1
    (the actually used range), not from header.number_of_micro_time_channels
    which includes unoccupied bins
"""
from __future__ import annotations

import os
import numpy as np
import tifffile

try:
    import tttrlib
    TTTRLIB_AVAILABLE = True
except ImportError:
    TTTRLIB_AVAILABLE = False


def _check_tttrlib():
    if not TTTRLIB_AVAILABLE:
        raise ImportError(
            "tttrlib is not installed. Install with: pip install tttrlib"
        )


def _parse_channels(channel_str: str) -> list[int]:
    """'0,1' → [0, 1];  '' → []"""
    return [int(c.strip()) for c in channel_str.split(",")
            if c.strip()] if channel_str.strip() else []


def _make_output_dir(ptu_path: str) -> str:
    stem    = os.path.splitext(os.path.basename(ptu_path))[0]
    out_dir = os.path.join(os.path.dirname(ptu_path), f"{stem}_images")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
def get_channel_info(ptu_path: str) -> dict:
    """
    Load a PTU file and return channel photon counts and TCSPC parameters
    so the GUI can display them to the user before running the full pipeline.

    Returns
    -------
    dict with keys:
        routing_channels        : list[int]
        photons_per_channel     : dict {channel: count}
        detector_channels       : list[int]   (fraction > 0.1% of total)
        marker_channels         : list[int]
        n_micro_time_bins       : int   micro_times.max() + 1
        n_micro_time_bins_header: int   header allocation
        micro_time_res_ns       : float
        macro_time_res_ns       : float
        n_frames                : int   (from CLSMImage)
        n_photons               : int
    """
    _check_tttrlib()
    tttr_data = tttrlib.TTTR(ptu_path)

    routing_channels = sorted(int(c) for c in np.unique(tttr_data.routing_channels))
    total            = len(tttr_data.macro_times)
    photons_per_ch   = {}
    detector_ch      = []
    marker_ch        = []

    for ch in routing_channels:
        idx  = tttr_data.get_selection_by_channel([ch])
        n    = int(idx.shape[0])
        photons_per_ch[ch] = n
        if n / max(total, 1) > 0.001:
            detector_ch.append(ch)
        else:
            marker_ch.append(ch)

    n_used   = int(tttr_data.micro_times.max()) + 1
    try:
        n_header = tttr_data.header.number_of_micro_time_channels
    except AttributeError:
        n_header = n_used

    micro_res_ns = float(tttr_data.header.micro_time_resolution) * 1e9
    macro_res_ns = float(tttr_data.header.macro_time_resolution) * 1e9

    # get n_frames from a CLSMImage
    try:
        clsm    = tttrlib.CLSMImage(tttr_data, fill=True,
                                    channels=[routing_channels[0]])
        n_frames = int(clsm.intensity.shape[0])
    except Exception:
        n_frames = -1

    return {
        "routing_channels":         routing_channels,
        "photons_per_channel":      photons_per_ch,
        "detector_channels":        detector_ch,
        "marker_channels":          marker_ch,
        "n_micro_time_bins":        n_used,
        "n_micro_time_bins_header": n_header,
        "micro_time_res_ns":        micro_res_ns,
        "macro_time_res_ns":        macro_res_ns,
        "n_frames":                 n_frames,
        "n_photons":                total,
    }

def _save_svg_preview(img_2d: np.ndarray, path: str,
                      title: str = "", cmap: str = "cividis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(img_2d, cmap=cmap)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_ptu_image_export(params: dict, progress_callback=None) -> dict:
    """
    Main entry point for PTU image / FLIM extraction.

    params keys
    ───────────
    ptu_path          : str
    irf_path          : str   (optional; "" = no FLIM)
    green_channels    : str   comma-separated, e.g. "0" or "0,1"
    red_channels      : str   comma-separated; "" = single-channel mode
    use_pie           : bool
    prompt_start_bin  : int   prompt window start in TCSPC bin units
    prompt_stop_bin   : int   prompt window stop  in TCSPC bin units
    delay_start_bin   : int   delay  window start in TCSPC bin units
    delay_stop_bin    : int   delay  window stop  in TCSPC bin units
    frame_start       : int
    frame_end         : int   -1 = all frames
    stack_frames_flim : bool
    min_photons_flim  : int
    save_tif          : bool
    save_flim         : bool
    save_svg          : bool
    out_dir           : str   "" = auto
    """
    _check_tttrlib()

    def _prog(pct: float):
        if progress_callback is not None:
            progress_callback(float(pct))

    _prog(0.0)

    ptu_path   = params["ptu_path"]
    irf_path   = params.get("irf_path", "").strip()
    green_ch   = _parse_channels(params.get("green_channels", "0"))
    red_ch     = _parse_channels(params.get("red_channels", "").strip())
    has_red    = len(red_ch) > 0
    use_pie    = bool(params.get("use_pie", False))
    frame_start     = int(params.get("frame_start", 0))
    frame_end       = int(params.get("frame_end", -1))
    stack_flim      = bool(params.get("stack_frames_flim", True))
    min_photons     = int(params.get("min_photons_flim", 30))
    save_tif        = bool(params.get("save_tif", True))
    save_flim_flag  = bool(params.get("save_flim", True))
    save_svg        = bool(params.get("save_svg", True))
    out_dir         = params.get("out_dir", "").strip() or _make_output_dir(ptu_path)
    os.makedirs(out_dir, exist_ok=True)
    saved_files: list[str] = []

    # ── load ─────────────────────────────────────────────────────────────────
    tttr_data    = tttrlib.TTTR(ptu_path)
    micro_res_ns = float(tttr_data.header.micro_time_resolution) * 1e9
    macro_res_ns = float(tttr_data.header.macro_time_resolution) * 1e9

    # number of TCSPC bins actually used by the photons in this file
    # (micro_times.max()+1), which may be much smaller than
    # header.number_of_micro_time_channels (the full allocated range)
    n_micro_time_bins = int(tttr_data.micro_times.max()) + 1

    _prog(5.0)

     # ── PIE window bins ───────────────────────────────────────────────────────
    # Windows are specified as relative fractions (0–1) in params and
    # converted to absolute TCSPC bin indices here, using the actually-used
    # range (n_micro_time_bins = micro_times.max()+1, not the full allocated
    # header value which includes unoccupied bins).
    #
    # Example: split_at=0.5 on a file with n_micro_time_bins=2500 gives
    #   prompt_range = (0,    1250)
    #   delay_range  = (1250, 2500)

    prompt_start_rel = float(params.get("prompt_start", 0.0))
    prompt_stop_rel  = float(params.get("prompt_stop",  0.5))
    delay_start_rel  = float(params.get("delay_start",  0.5))
    delay_stop_rel   = float(params.get("delay_stop",   1.0))

    prompt_range = (
        int(prompt_start_rel * n_micro_time_bins),
        int(prompt_stop_rel  * n_micro_time_bins),
    )
    delay_range = (
        int(delay_start_rel  * n_micro_time_bins),
        int(delay_stop_rel   * n_micro_time_bins),
    )

    # ── TCSPC histograms (counts vs bin index) ────────────────────────────────
    # microtime_histogram[0] is the full allocated array (e.g. 10922 bins)
    # but photons only occupy bins 0 to n_micro_time_bins-1 (e.g. 0–2499).
    # Crop to the used range so the x-axis and PIE window fractions are correct.
    tcspc_hist: dict[str, tuple] = {}

    if green_ch:
        sub    = tttr_data.get_tttr_by_channel(green_ch)
        counts = np.asarray(sub.microtime_histogram[0], dtype=float)
        counts = counts[:n_micro_time_bins]          # crop to used range
        tcspc_hist["green"] = (counts, np.arange(len(counts)))

    if has_red:
        sub    = tttr_data.get_tttr_by_channel(red_ch)
        counts = np.asarray(sub.microtime_histogram[0], dtype=float)
        counts = counts[:n_micro_time_bins]          # crop to used range
        tcspc_hist["red"] = (counts, np.arange(len(counts)))

    _prog(15.0)

    # ── CLSMImage intensity ───────────────────────────────────────────────────
    # Use fill=True in the constructor when no micro-time gating is needed
    # (the simple, recommended approach from the tttrlib examples).
    # For PIE gating, use fill() with micro_time_ranges after construction.

    intensity_images: dict[str, np.ndarray] = {}
    clsm_objects: dict[str, "tttrlib.CLSMImage"] = {}

    if green_ch:
        clsm_green = tttrlib.CLSMImage(tttr_data, fill=True, channels=green_ch)
        clsm_objects["green"] = clsm_green
        fs = max(0, frame_start)
        fe = clsm_green.intensity.shape[0] if frame_end < 0 else min(
            clsm_green.intensity.shape[0], frame_end + 1)
        intensity_images["green"] = clsm_green.intensity[fs:fe].sum(axis=0)

    n_frames = (clsm_objects["green"].intensity.shape[0]
                if "green" in clsm_objects else 0)
    fs = max(0, frame_start)
    fe = n_frames if frame_end < 0 else min(n_frames, frame_end + 1)

    if has_red:
        clsm_red = tttrlib.CLSMImage(tttr_data, fill=True, channels=red_ch)
        clsm_objects["red"] = clsm_red
        intensity_images["red"] = clsm_red.intensity[fs:fe].sum(axis=0)

        if use_pie:
            clsm_red_prompt = tttrlib.CLSMImage(tttr_data)
            clsm_red_prompt.fill(tttr_data, channels=red_ch,
                                 micro_time_ranges=[prompt_range])
            clsm_red_delay = tttrlib.CLSMImage(tttr_data)
            clsm_red_delay.fill(tttr_data, channels=red_ch,
                                micro_time_ranges=[delay_range])
            clsm_objects["red_prompt"] = clsm_red_prompt
            clsm_objects["red_delay"]  = clsm_red_delay
            intensity_images["red_prompt"] = clsm_red_prompt.intensity[fs:fe].sum(axis=0)
            intensity_images["red_delay"]  = clsm_red_delay.intensity[fs:fe].sum(axis=0)

    _prog(35.0)

    # ── save intensity TIFFs ──────────────────────────────────────────────────
    if save_tif:
        for label, img in intensity_images.items():
            path = os.path.join(out_dir, f"intensity_{label}.tif")
            tifffile.imwrite(path, img.astype(np.uint32), photometric="minisblack")
            saved_files.append(path)
            if save_svg:
                svg_path = os.path.join(out_dir, f"intensity_{label}.svg")
                _save_svg_preview(img, svg_path, title=f"Intensity — {label}")
                saved_files.append(svg_path)

    _prog(50.0)

    # ── FLIM ──────────────────────────────────────────────────────────────────
    flim_images: dict[str, np.ndarray] = {}
    irf = None
    tttr_irf_green_prompt = None
    tttr_irf_red_prompt   = None
    tttr_irf_red_delay    = None

    if irf_path and os.path.isfile(irf_path):
        irf = tttrlib.TTTR(irf_path)

        def _gated_irf(channels: list[int],
                       micro_range: tuple) -> "tttrlib.TTTR":
            mask = tttrlib.TTTRMask()
            mask.select_channels(irf, channels)
            mask.select_microtime_ranges(irf, [micro_range])
            return irf[mask.indices]

        flim_settings = {
            "tttr_data":                tttr_data,
            "minimum_number_of_photons": min_photons,
            "stack_frames":             stack_flim,
        }

        _prog(60.0)

        if green_ch:
            tttr_irf_green_prompt = _gated_irf(green_ch, prompt_range)
            tau_green = clsm_objects["green"].get_mean_lifetime(
                tttr_irf=tttr_irf_green_prompt, **flim_settings
            )
            flim_images["green"] = tau_green.sum(axis=0)

        _prog(75.0)

        if has_red and use_pie:
            if "red_prompt" in clsm_objects:
                tttr_irf_red_prompt = _gated_irf(red_ch, prompt_range)
                tau_red_prompt = clsm_objects["red_prompt"].get_mean_lifetime(
                    tttr_irf=tttr_irf_red_prompt, **flim_settings
                )
                flim_images["red_prompt"] = tau_red_prompt.sum(axis=0)

            if "red_delay" in clsm_objects:
                tttr_irf_red_delay = _gated_irf(red_ch, delay_range)
                tau_red_delay = clsm_objects["red_delay"].get_mean_lifetime(
                    tttr_irf=tttr_irf_red_delay, **flim_settings
                )
                flim_images["red_delay"] = tau_red_delay.sum(axis=0)

        _prog(88.0)

        if save_flim_flag:
            for label, img in flim_images.items():
                path = os.path.join(out_dir, f"flim_{label}.tif")
                tifffile.imwrite(path, img.astype(np.float32),
                                 photometric="minisblack")
                saved_files.append(path)
                if save_svg:
                    svg_path = os.path.join(out_dir, f"flim_{label}.svg")
                    _save_svg_preview(img, svg_path,
                                      title=f"Mean lifetime — {label} (ns)",
                                      cmap="CMRmap")
                    saved_files.append(svg_path)

    _prog(100.0)

    return {
        "out_dir":            out_dir,
        "intensity_images":   intensity_images,
        "flim_images":        flim_images,
        "tcspc_hist":         tcspc_hist,
        "n_frames":           n_frames,
        "n_micro_time_bins":  n_micro_time_bins,
        "micro_time_res_ns":  micro_res_ns,
        "macro_time_res_ns":  macro_res_ns,
        "prompt_range":       list(prompt_range),
        "delay_range":        list(delay_range),
        "use_pie":            use_pie,
        "saved_files":        saved_files,
        "ptu_path":           ptu_path,
    }