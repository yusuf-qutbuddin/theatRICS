import numpy as np
import tifffile as tiff  # pip install tifffile
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import multipletau
import matplotlib.pyplot as plt
from pylibCZIrw import czi as pyczi
import multiprocessing
import os
import pandas as pd

# np.random.seed(42)
# n_lines, n_pixels = 5000, 32
# data = np.random.poisson(10, (n_lines, n_pixels))
# for i in range(n_lines):
#     center = np.random.normal(n_pixels//2, 2)
#     data[i, max(0, int(center-3)):min(n_pixels, int(center+3))] += np.random.poisson(500, 6)

def read_frame(filepath,
               channel):
    with pyczi.open_czi(filepath) as czidoc:
        metadata = czidoc.metadata['ImageDocument']['Metadata']
        try:
            Frame_time_s = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel']['LaserScanInfo']['LineTime'])
        except:
            Frame_time_s = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel']['LaserScanInfo']['FrameTime'])

        total_bounding_rectangle = czidoc.total_bounding_rectangle
        data_frame = czidoc.read(roi=total_bounding_rectangle,
                                 plane={
                                     'C': channel})
    return data_frame, Frame_time_s

def fit_gaussian_chunk(args):
    """
    Fit a chunk of lines with curve_fit.
    args: (i0, block, n_pixels)
      i0: starting line index in the full frame
      block: ndarray shape (chunk_lines, n_pixels) uint16/float
      n_pixels: int
    returns: list of (i, peak, sigma)
    """
    i0, block, n_pixels = args
    x = np.arange(n_pixels)

    out = []
    for k in range(block.shape[0]):
        i = i0 + k
        line = block[k]
        out.append(fit_gaussian_line((i, line, x, n_pixels)))
    return out

def gaussian(x, amp, cen, sigma):
        return amp * np.exp(-(x - cen) ** 2 / (2 * sigma ** 2))

def fit_gaussian_line(args):
    """Fitting function for multiprocessing - unpacks (i, line_data, x, n_pixels)"""
    (i, line_data, x, n_pixels) = args
    
    y = line_data.astype(float)

    # Skip empty / almost flat lines
    if np.allclose(y, 0) or np.nanmax(y) <= 0:
        result = (i,np.argmin(y),5.0)
        return result

    y_smooth = gaussian_filter1d(y, sigma=1)

    # Safer initial guess
    amp0 = float(np.max(y_smooth))
    cen0 = float(np.argmax(y_smooth))
    sig0 = 5.0

    # Reasonable bounds
    bounds = (
        [0, 0, 0.5],  # lower
        [10 * amp0, n_pixels - 1, n_pixels]  # upper
    )

    try:
        popt, _ = curve_fit(
            gaussian, x, y_smooth,
            p0=[amp0, cen0, sig0],
            bounds=bounds,
            maxfev=2000
        )
        result = (i, popt[1], abs(popt[2]))
        return result
    except Exception:
        # Fallback: simple maximum
        result = (i, np.argmax(y_smooth), sig0)
        return result


def wohland_bootstrap(intensity_traces, line_time_s,G_lags, n_synthetic=500, n_pairs_per_lag=1000):
    """
    Wohland method: Synthetic ACFs via random time pair sampling
    n_synthetic: Number of synthetic correlation curves
    n_pairs_per_lag: Photon pairs sampled per lag time
    """
    

    trace = intensity_traces.astype(float) - np.mean(intensity_traces)  # DEMEAN FIRST
    n_samples = len(trace)
    lags = (G_lags / line_time_s).astype(int)

    # Precompute global normalization from original G
    mean_intensity = np.mean(intensity_traces)
    global_norm = mean_intensity ** 2  # <I>² for proper FCS normalization

    synthetic_Gs = []

    for _ in range(n_synthetic):
        G_synth = np.zeros(len(lags))

        for lag_idx, tau in enumerate(lags):
            if tau >= n_samples:
                G_synth[lag_idx] = 0
                continue

            t_starts = np.random.randint(0, n_samples - tau, n_pairs_per_lag)
            corr_pairs = trace[t_starts] * trace[t_starts + tau]
            G_synth[lag_idx] = np.mean(corr_pairs)

        # FIXED: Use global normalization, not G_synth[0]
        G_synth = G_synth / global_norm
        synthetic_Gs.append(G_synth)

    return np.std(synthetic_Gs, axis=0)

def run_autocorrelation(intensity_traces, line_time_s, root):
    
    G = multipletau.autocorrelate(intensity_traces, m=12, deltat=line_time_s, normalize=True)
    G_std = wohland_bootstrap(intensity_traces, line_time_s,G[:,0], n_synthetic=1000, n_pairs_per_lag=10000)
    countrate = np.full_like(G_std, np.mean(intensity_traces))
    correlate_df = pd.DataFrame({
        'lag_time': G[:, 0],
        'correlation': G[:, 1],
        'countrate': countrate,  # Same for row 1 & 2
        'std_dev': G_std
    })
    correlate_df.to_csv(root+'_correlation.csv',
                        index=False, header = False)
    # Plot
    return G, G_std


def read_file(filepath, channel):
    """
    Read a line-scan file for SFCS processing.
    Supports CZI (Zeiss) and PTU (PicoQuant Luminosa).

    Returns
    -------
    frame_data   : np.ndarray (n_lines, n_pixels)
    line_time_s  : float
    x            : np.ndarray  — pixel index array
    n_lines      : int
    n_pixels     : int
    root         : str          — path without extension
    """
    root, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext == ".czi":
        # ── original CZI path ────────────────────────────────────
        channel_to_use = channel
        frame_data, line_time_s = read_frame(filepath, channel_to_use)
        tiff.imwrite(root + ".tif", frame_data)
        n_lines  = frame_data.shape[0]
        n_pixels = frame_data.shape[1]
        frame_data = frame_data.reshape(n_lines, n_pixels)

    elif ext == ".ptu":
        # ── PTU line-scan path ───────────────────────────────────
        if not TTTRLIB_AVAILABLE:
            raise ImportError(
                "tttrlib is not installed. "
                "Install it with:  pip install tttrlib"
            )
        frame_data, line_time_s, meta, root = read_ptu_linescan(
            filepath, channel=channel
        )
        n_lines  = frame_data.shape[0]
        n_pixels = frame_data.shape[1]

        # save as TIFF for consistency with CZI workflow
        tiff.imwrite(root + ".tif", frame_data)

    else:
        raise ValueError(
            f"Unsupported file format for SFCS: {ext}. "
            f"Supported formats: .czi, .ptu"
        )

    x = np.arange(n_pixels)
    return frame_data, line_time_s, x, n_lines, n_pixels, root
def alignment(frame_data, n_pixels, n_lines, root, peaks):
    # #alignment
    center_target = n_pixels // 2
    aligned_data = np.zeros_like(frame_data)
    for i in range(n_lines):
        shift_amt = int(center_target - peaks[i])
        aligned_data[i] = np.roll(frame_data[i], shift_amt)

    # Save as TIFF; shape is (n_lines, n_pixels)
    tiff.imwrite(root + "_aligned.tif", aligned_data)
    return aligned_data

def calculate_intensity_trace(aligned_data, n_lines, n_pixels, sigmas, root):
    # Make the intensity trace here
    # Sum photons in ±2.5σ window per line for intensity trace
    intensity_traces = np.zeros(n_lines)
    center_target = n_pixels // 2

    for i in range(n_lines):
        half_width = 2.5 * sigmas[i]
        start = max(0, int(center_target - half_width))
        end = min(n_pixels, int(center_target + half_width))
        intensity_traces[i] = np.sum(aligned_data[i, start:end])

    intensity_df = pd.DataFrame({
        'line_number': range(len(intensity_traces)),
        'intensity': intensity_traces
    })
    intensity_df.to_csv(root + "_intensity_trace.csv", index=False)
    return intensity_traces


# ────────────────────────────────────────────────────────────────
# PTU line-scan support (PicoQuant Luminosa)
# ────────────────────────────────────────────────────────────────

try:
    import tttrlib
    TTTRLIB_AVAILABLE = True
except ImportError:
    TTTRLIB_AVAILABLE = False


def read_ptu_linescan_metadata(filepath: str) -> dict:
    """
    Read timing metadata from a PicoQuant Luminosa line-scan PTU file.

    For line-scan PTU files:
    - ImgHdr_PixX / ImgHdr_PixY are NOT present
    - n_pixels is derived from line_time_ms / pixel_dwell_time_ms
    - n_lines is counted from line-start marker events (channel 1)
    - line_time_ms is measured from median interval between
      consecutive line-start markers

    Parameters
    ----------
    filepath : str
        Path to the .ptu line-scan file.

    Returns
    -------
    dict with keys:
        pixel_dwell_time_us : float  — µs per pixel
        line_time_ms        : float  — ms per line (full line period)
        n_lines             : int    — number of lines in the scan
        n_pixels            : int    — pixels per line (derived)
        macro_res_s         : float  — macro-time resolution in s/tick
        line_start_channel  : int    — marker channel for line start
        line_stop_channel   : int    — marker channel for line stop
        photon_channels     : list   — routing channels carrying photons
        total_duration_s    : float  — total acquisition duration in s
    """
    if not TTTRLIB_AVAILABLE:
        raise ImportError(
            "tttrlib is not installed. "
            "Install it with:  pip install tttrlib"
        )

    tttr_data = tttrlib.TTTR(str(filepath))
    hd        = tttr_data.header.data

    # ── macro-time resolution ────────────────────────────────────
    macro_res_s = float(hd["MeasDesc_GlobalResolution"][0])

    # ── pixel dwell time ─────────────────────────────────────────
    # ImgHdr_TimePerPixel is in ms for line-scan PTU
    pixel_dwell_time_ms = float(hd["ImgHdr_TimePerPixel"][0])
    pixel_dwell_time_us = pixel_dwell_time_ms * 1e3   # ms → µs

    # ── marker channel indices ───────────────────────────────────
    # ImgHdr_LineStart=1, ImgHdr_LineStop=2 means marker channel 1
    # and 2 respectively (confirmed from debug output)
    line_start_channel = int(hd["ImgHdr_LineStart"][0])   # = 1
    line_stop_channel  = int(hd["ImgHdr_LineStop"][0])    # = 2

    # ── raw event arrays ─────────────────────────────────────────
    macro_times      = tttr_data.macro_times
    routing_channels = tttr_data.routing_channels
    event_types      = tttr_data.event_types

    # ── line-start markers ───────────────────────────────────────
    line_start_mask  = (event_types == 1) & (routing_channels == line_start_channel)
    line_start_times = macro_times[line_start_mask]
    n_lines          = int(len(line_start_times))

    if n_lines < 2:
        raise ValueError(
            f"Too few line-start markers found ({n_lines}). "
            f"Check that the file is a line-scan PTU."
        )

    # ── line period from median interval between line starts ──────
    diffs_ticks  = np.diff(line_start_times.astype(np.float64))
    median_ticks = float(np.median(diffs_ticks))
    line_time_ms = median_ticks * macro_res_s * 1e3   # s → ms

    # ── n_pixels derived from timing ─────────────────────────────
    # n_pixels = line_time / pixel_dwell_time (both in ms)
    n_pixels = max(1, int(round(line_time_ms / pixel_dwell_time_ms)))

    # ── photon routing channels ───────────────────────────────────
    photon_mask     = event_types == 0
    photon_channels = sorted(
        int(c) for c in np.unique(routing_channels[photon_mask])
    )

    # ── total duration ────────────────────────────────────────────
    total_duration_s = float(macro_times.max()) * macro_res_s

    return {
        "pixel_dwell_time_us": pixel_dwell_time_us,
        "line_time_ms":        line_time_ms,
        "n_lines":             n_lines,
        "n_pixels":            n_pixels,
        "macro_res_s":         macro_res_s,
        "line_start_channel":  line_start_channel,
        "line_stop_channel":   line_stop_channel,
        "photon_channels":     photon_channels,
        "total_duration_s":    total_duration_s,
    }


def read_ptu_linescan(filepath: str,
                      channel: int = 1) -> tuple:
    """
    Read a PicoQuant Luminosa line-scan PTU file and reconstruct
    the 2D intensity image (n_lines × n_pixels) by binning photons
    between consecutive line-start markers.

    This produces the same frame_data array shape as read_file()
    for CZI files, so the rest of the SFCS pipeline is unchanged.

    Parameters
    ----------
    filepath : str
        Path to the .ptu line-scan file.
    channel  : int
        Photon routing channel to use (default=1 for Luminosa
        detector 1). Use read_ptu_linescan_metadata() to see
        available channels.

    Returns
    -------
    frame_data   : np.ndarray, shape (n_lines, n_pixels), dtype float32
        Photon count image — rows = lines, columns = pixel positions.
    line_time_s  : float
        Line period in seconds (used as delta_t in autocorrelation).
    meta         : dict
        Full metadata dict from read_ptu_linescan_metadata().
    root         : str
        File path without extension (for saving outputs).
    """
    if not TTTRLIB_AVAILABLE:
        raise ImportError(
            "tttrlib is not installed. "
            "Install it with:  pip install tttrlib"
        )

    meta      = read_ptu_linescan_metadata(filepath)
    n_lines   = meta["n_lines"]
    n_pixels  = meta["n_pixels"]
    macro_res = meta["macro_res_s"]
    line_start_channel = meta["line_start_channel"]

    # validate channel
    if channel not in meta["photon_channels"]:
        raise ValueError(
            f"Routing channel {channel} has no photons. "
            f"Available photon channels: {meta['photon_channels']}"
        )

    # ── raw event arrays ─────────────────────────────────────────
    tttr_data        = tttrlib.TTTR(str(filepath))
    macro_times      = tttr_data.macro_times
    routing_channels = tttr_data.routing_channels
    event_types      = tttr_data.event_types

    # ── line-start marker times ───────────────────────────────────
    line_start_mask  = (event_types == 1) & (routing_channels == line_start_channel)
    line_start_times = macro_times[line_start_mask]

    # ── photon times for the selected channel ─────────────────────
    photon_mask  = (event_types == 0) & (routing_channels == channel)
    photon_times = macro_times[photon_mask]

    # ── build frame_data by binning photons per line ──────────────
    frame_data = np.zeros((n_lines, n_pixels), dtype=np.float32)

    # pixel width in macro-time ticks
    # use median line period for consistent binning
    diffs_ticks  = np.diff(line_start_times.astype(np.float64))
    median_ticks = float(np.median(diffs_ticks))
    pixel_ticks  = median_ticks / n_pixels

    # for each line, find photons between this line start and the next
    # use searchsorted for efficiency — O(n_photons * log(n_lines))
    line_start_arr = line_start_times.astype(np.int64)

    for i in range(n_lines):
        t_start = int(line_start_arr[i])

        # end of line: use next line-start if available,
        # else use start + median period
        if i < n_lines - 1:
            t_end = int(line_start_arr[i + 1])
        else:
            t_end = t_start + int(round(median_ticks))

        # photons in this line window
        lo = int(np.searchsorted(photon_times, t_start, side="left"))
        hi = int(np.searchsorted(photon_times, t_end,   side="left"))

        if hi > lo:
            # relative time within line → pixel index
            rel_times = photon_times[lo:hi].astype(np.float64) - t_start
            px_indices = np.floor(rel_times / pixel_ticks).astype(np.int32)

            # clamp to valid range (in case of timing jitter at line edges)
            np.clip(px_indices, 0, n_pixels - 1, out=px_indices)

            # accumulate photon counts into pixels
            np.add.at(frame_data[i], px_indices, 1)

    line_time_s = meta["line_time_ms"] * 1e-3
    root        = os.path.splitext(filepath)[0]

    return frame_data, line_time_s, meta, root
# Usage
if __name__ == "__main__":
    pass
    #
    # filepath = r'X:\AlNahas_Kareem\Raquel_mastersproject\20260220_sfcs\New-09_xy.czi'
    #
    # # Prepare arguments for multiprocessing
    # args_list = [(i, frame_data[i], x, n_pixels) for i in range(n_lines)]
    #
    # # Parallel Gaussian fitting - uses all available CPU cores
    # print("Fitting Gaussians with multiprocessing...")
    # pool = multiprocessing.Pool(processes=20)
    # results = pool.starmap(fit_gaussian_line, args_list)
    #
    # # Unpack results (maintain original order)
    # peaks = np.full(n_lines, 0.0)
    # sigmas = np.full(n_lines, 5.0)
    # for i, peak, sigma in results:
    #     peaks[i] = peak
    #     sigmas[i] = sigma
    #
    # print(f"Fitting complete. Found {np.sum(sigmas > 0)} valid peaks.")
    # # for i in range(n_lines):
    # #     y_smooth = gaussian_filter1d(frame_data[i], sigma=1)
    # #     try:
    # #         popt, _ = curve_fit(gaussian, x, y_smooth, p0=[np.max(y_smooth), n_pixels//2, 5])
    # #         peaks[i], sigmas[i] = popt[1], np.abs(popt[2])  # Ensure positive sigma
    # #     except:
    # #         peaks[i] = np.argmax(y_smooth)
    #
    #
    #
    # # Make the intensity trace here
    # # Sum photons in ±2.5σ window per line for intensity trace
    # intensity_traces = np.zeros(n_lines)
    # for i in range(n_lines):
    #     half_width = 2.5 * sigmas[i]
    #     start = max(0, int(center_target - half_width))
    #     end = min(n_pixels, int(center_target + half_width))
    #     intensity_traces[i] = np.sum(aligned_data[i, start:end])
    #
    # intensity_df = pd.DataFrame({
    #     'line_number': range(len(intensity_traces)),
    #     'intensity': intensity_traces
    # })
    # intensity_df.to_csv(root+"_intensity_trace.csv", index=False)
    #
    #
    # plt.plot(intensity_traces)
    # plt.xlabel('Line #')
    # plt.ylabel('Intensity')
    # plt.savefig(root+"_intensity_trace.svg")
    # plt.close()
    #
    #
    # run_autocorrelation(intensity_traces, line_time_s, root)
    #
    # #
    #