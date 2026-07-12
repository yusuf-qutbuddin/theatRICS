"""
theatrics/fcsfit/zeiss_raw_correlate.py

FCS correlation export for Zeiss ConfoCor3 / LSM980 .raw photon files.

Faithfully ported from user-supplied MATLAB scripts:
    - readConfoCor3.m         (binary .raw file parser)
    - cross_corr.m            (Laurence et al. 2006 fast photon-counting ACF/CCF)
    - cross_corr_weights.m    (weighted variant, for bleach correction)
    - get_blcorr_weights.m    (polynomial bleach-correction weight generator)
    - the main batch script   (segment-based Wohland SD, afterpulsing
                                subtraction, CSV export column layout)

Reference:
    A. Laurence, S. Fore, T. Huser, "Fast, flexible algorithm for
    calculating photon correlations," Opt. Lett. 31, 829-831 (2006)

Why PIE and FLCS are not available for .raw files
──────────────────────────────────────────────────
Zeiss .raw files carry no TCSPC micro-time information and no pulsed
interleaved excitation gating — only macro-time ("sync tick") arrival
times per channel (see readConfoCor3.m: ph_dtime is a dummy array of
ones). Both PIE and FLCS background correction fundamentally require
micro-time histograms, so neither is offered for .raw input (enforced
at the GUI level in gui_app.py).
"""

from __future__ import annotations

import os
import glob
import struct
import numpy as np
import pandas as pd
from scipy import stats as _stats

# ── optional numba acceleration for the photon-counting inner loop ─────────
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def is_raw_file(filepath: str) -> bool:
    """Return True if filepath has a .raw extension (case-insensitive)."""
    return os.path.splitext(filepath)[1].lower() == ".raw"


# ═════════════════════════════════════════════════════════════════════════
# Section 1 ── Binary file reader  (faithful port of readConfoCor3.m)
# ═════════════════════════════════════════════════════════════════════════

def read_confocor3_raw(filepath: str) -> dict:
    """
    Read a Zeiss ConfoCor3 / LSM980 .raw photon-count file.

    Faithful port of readConfoCor3.m. The MATLAB source reads:
      1. A 64-byte ASCII header block   ('Header' / TagHead.Ident)
      2. Four uint32 values              ('Identifier' / TagHead.Idx)
      3. Four uint32 values               ('Settings' / TagHead.Typ)
         -> Settings(4) (1-indexed, i.e. index 3 in 0-indexed Python)
            is TTResult_SyncRate.
      4. Eight uint32 values discarded (padding / unused header fields)
      5. The remaining file content as a stream of uint32 values, each
         being a DELTA (inter-photon time in sync-tick units). The
         absolute photon arrival times are the cumulative sum of these
         deltas (MATLAB: cumsum(T3Record)).

    Channel number is read from the LAST CHARACTER of the 64-byte ASCII
    header string, converted to a number (MATLAB: str2double(Header(end))).
    This matches how the calling script appends a channel-name suffix
    (e.g. '_ChS1') to the filename before calling this reader — the
    channel digit ends up as the last character of that filename, which
    becomes embedded in the 64-byte header block written by the Zeiss
    acquisition software.

    Parameters
    ----------
    filepath : str
        Path to a single-channel .raw file, e.g.
        "Measurement_R1_P0_K1_ChS1.raw"

    Returns
    -------
    dict with keys:
        ph_sync                    : np.ndarray (uint64) cumulative
                                      photon arrival times, in sync-tick
                                      units
        ph_channel                 : np.ndarray, constant channel index
                                      per photon (from header)
        ph_dtime                   : np.ndarray of ones (dummy — no
                                      micro-time info in .raw files)
        TTResult_SyncRate          : float, sync ticks per second
        MeasDesc_Resolution        : float, dummy (always 1, picoseconds)
        File_CreatingTime          : str, file modification timestamp
        MeasDesc_AcquisitionTime   : float, acquisition duration in ms
                                      (= last photon's arrival time
                                      converted to ms)
        header_raw                 : str, the raw 64-byte ASCII header
                                      (for diagnostics / channel parsing)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"RAW file not found: {filepath}")

    with open(filepath, "rb") as f:
        # 1. 64-byte ASCII header
        header_bytes = f.read(64)
        header_str = header_bytes.decode("ascii", errors="replace")

        # 2. Identifier: 4 x uint32  (TagHead.Idx)
        identifier = np.frombuffer(f.read(4 * 4), dtype="<u4")

        # 3. Settings: 4 x uint32   (TagHead.Typ)
        #    MATLAB Settings(4) (1-indexed) == settings[3] (0-indexed)
        settings = np.frombuffer(f.read(4 * 4), dtype="<u4")
        sync_rate = float(settings[3])

        # 4. 8 x uint32 padding / unused
        _ = np.frombuffer(f.read(8 * 4), dtype="<u4")

        # 5. Remaining file: stream of uint32 inter-photon deltas
        raw_bytes = f.read()

    if sync_rate <= 0:
        raise ValueError(
            f"Invalid TTResult_SyncRate ({sync_rate}) read from {filepath}. "
            f"File may be corrupted or not a valid ConfoCor3 .raw file."
        )

    deltas = np.frombuffer(raw_bytes, dtype="<u4")
    if deltas.size == 0:
        raise ValueError(f"No photon records found in {filepath}.")

    # cumsum(T3Record) in MATLAB -- absolute arrival times in sync ticks.
    # Use uint64 accumulation to avoid overflow for long acquisitions.
    ph_sync = np.cumsum(deltas.astype(np.uint64))

    # channel index: last character of the 64-byte header, as a number
    # (MATLAB: str2double(Header(end)))
    last_char = header_str.rstrip("\x00").strip()[-1:] if header_str.strip("\x00").strip() else ""
    try:
        channel_value = float(last_char)
    except ValueError:
        channel_value = np.nan

    ph_channel = np.full(ph_sync.shape, channel_value, dtype=np.float64)
    ph_dtime = np.ones(ph_sync.shape, dtype=np.float64)  # dummy, no microtime

    # file modification time, as MATLAB's dir(...).date fallback
    file_mtime = os.path.getmtime(filepath)
    import datetime
    file_creating_time = datetime.datetime.fromtimestamp(file_mtime).isoformat()

    meas_desc_acquisition_time_ms = float(ph_sync[-1]) * 1e3 / sync_rate

    return {
        "ph_sync":                   ph_sync,
        "ph_channel":                ph_channel,
        "ph_dtime":                  ph_dtime,
        "TTResult_SyncRate":         sync_rate,
        "MeasDesc_Resolution":       1.0,
        "File_CreatingTime":         file_creating_time,
        "MeasDesc_AcquisitionTime":  meas_desc_acquisition_time_ms,
        "header_raw":                header_str,
    }


# ═════════════════════════════════════════════════════════════════════════
# Section 2 ── Bleach correction weights  (faithful port of
#               get_blcorr_weights.m + blcorr_trace + get_dense_trace)
# ═════════════════════════════════════════════════════════════════════════

def _get_dense_trace(time_tags: np.ndarray, n_bins: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin photon arrival times into a dense intensity trace.

    Faithful port of get_dense_trace.m:
        [trace_y, trace_x_appended] = histcounts(time_tags, n_bins);
        trace_x = trace_x_appended(1:end-1);

    Parameters
    ----------
    time_tags : np.ndarray  photon arrival times (any consistent unit)
    n_bins    : int         number of histogram bins (MATLAB default: 1000)

    Returns
    -------
    trace_x : np.ndarray  left bin edges (length n_bins)
    trace_y : np.ndarray  photon counts per bin (length n_bins)
    """
    trace_y, bin_edges = np.histogram(time_tags, bins=n_bins)
    trace_x = bin_edges[:-1]
    return trace_x.astype(np.float64), trace_y.astype(np.float64)


def _blcorr_trace(
    trace_x: np.ndarray,
    trace_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Fit a bleaching/drift trend to a binned intensity trace using an
    F-test-driven polynomial order selection, then apply the Ries et al.
    2008 correction formula.

    Faithful port of blcorr_trace.m.

    Algorithm
    ---------
    1. Start with a 1st-degree polynomial fit.
    2. Increase the polynomial degree while the F-test comparing
       consecutive fits shows a statistically significant improvement
       (alpha = 0.05), up to a maximum degree of 10.
    3. Stop when the improvement from degree n-1 to degree n is no
       longer significant; use degree n-1 as the final choice.
    4. Apply the depletion-correction formula (same as
       theatrics.utils.bleach_correction.depletion_correct, but computed
       inline here to keep this a faithful 1:1 port of the MATLAB code):

           y_corr = y / sqrt(fit / fit[0]) + fit[0] * (1 - sqrt(fit / fit[0]))

    Parameters
    ----------
    trace_x : np.ndarray  bin left-edges (independent variable)
    trace_y : np.ndarray  photon counts per bin

    Returns
    -------
    trace_y_bleachcorr : np.ndarray  corrected trace
    trace_y_fit        : np.ndarray  fitted polynomial trend
    fitparam           : np.ndarray  polynomial coefficients (numpy
                                     polyfit convention: highest degree
                                     first, matching MATLAB's polyfit)
    bleachcorr_degree  : int         chosen polynomial degree
    """
    # MATLAB: trace_w = sqrt(trace_y); trace_w(trace_y==0) = max(trace_w);
    trace_w = np.sqrt(trace_y)
    if np.any(trace_y == 0):
        trace_w[trace_y == 0] = np.max(trace_w) if np.max(trace_w) > 0 else 1.0

    bleachcorr_degree = 1
    chi_sq = np.zeros(10)
    fitparams = [None] * 10

    while bleachcorr_degree <= 10:
        idx = bleachcorr_degree - 1  # 0-indexed storage

        # MATLAB: polyfit(trace_x, trace_y, degree) -- unweighted fit
        coeffs = np.polyfit(trace_x, trace_y, bleachcorr_degree)
        fitparams[idx] = coeffs
        trace_y_fit = np.polyval(coeffs, trace_x)

        # MATLAB: chi_sq(degree) = sum((trace_y - trace_y_fit./trace_w).^2)
        #                          / (length(trace_y) - degree)
        # NOTE: reproduced exactly as written in the MATLAB source,
        # including its (unusual) division of trace_y_fit by trace_w
        # rather than dividing the residual by trace_w.
        chi_sq[idx] = np.sum(
            (trace_y - trace_y_fit / trace_w) ** 2
        ) / (len(trace_y) - bleachcorr_degree)

        if bleachcorr_degree > 1:
            # MATLAB: f_crit = finv(0.95, degree-1, degree)
            #         f_val  = chi_sq(degree) / chi_sq(degree-1)
            f_crit = _stats.f.ppf(0.95, bleachcorr_degree - 1, bleachcorr_degree)
            f_val = chi_sq[idx] / chi_sq[idx - 1]

            if f_val <= f_crit:
                # insignificant improvement -- previous degree was optimal
                bleachcorr_degree -= 1
                break
            elif bleachcorr_degree == 10:
                break
            else:
                bleachcorr_degree += 1
        else:
            bleachcorr_degree += 1

    final_idx = bleachcorr_degree - 1
    fitparam = fitparams[final_idx]
    trace_y_fit = np.polyval(fitparam, trace_x)

    # MATLAB: trace_y_bleachcorr = real(
    #             trace_y ./ sqrt(trace_y_fit ./ trace_y_fit(1))
    #             + trace_y_fit(1) .* (1 - sqrt(trace_y_fit ./ trace_y_fit(1)))
    #         );
    ratio = trace_y_fit / trace_y_fit[0]
    sqrt_ratio = np.sqrt(np.clip(ratio, 0, None).astype(complex)).real
    with np.errstate(divide="ignore", invalid="ignore"):
        trace_y_bleachcorr = np.where(
            sqrt_ratio != 0,
            trace_y / np.where(sqrt_ratio != 0, sqrt_ratio, 1.0)
            + trace_y_fit[0] * (1.0 - sqrt_ratio),
            trace_y_fit[0],
        )

    return trace_y_bleachcorr, trace_y_fit, fitparam, bleachcorr_degree


def get_blcorr_weights(time_tags: np.ndarray, n_bins: int = 1000) -> np.ndarray:
    """
    Compute per-photon bleach-correction weights.

    Faithful port of get_blcorr_weights.m:
        1. Bin photons into a dense trace (1000 bins).
        2. Fit an optimal-degree polynomial bleaching trend to that
           trace (via _blcorr_trace / F-test degree selection).
        3. Evaluate the fitted polynomial AT EACH INDIVIDUAL PHOTON'S
           ARRIVAL TIME (not at the bin centres) to get a smooth,
           continuous per-photon bleaching estimate.
        4. Convert to a per-photon weight via:

               w = real( 1/sqrt(fit/fit[0]) + (1 - sqrt(fit/fit[0])) )

    Parameters
    ----------
    time_tags : np.ndarray  photon arrival times (sync-tick units)
    n_bins    : int         histogram bins for the dense trace (default 1000)

    Returns
    -------
    photon_weights : np.ndarray, same length as time_tags
    """
    trace_x, trace_y = _get_dense_trace(time_tags, n_bins)
    _, _, fitparam, _ = _blcorr_trace(trace_x, trace_y)

    trace_y_fit_photons = np.polyval(fitparam, time_tags.astype(np.float64))

    ratio = trace_y_fit_photons / trace_y_fit_photons[0]
    sqrt_ratio = np.sqrt(np.clip(ratio, 0, None).astype(complex)).real

    with np.errstate(divide="ignore", invalid="ignore"):
        photon_weights = np.where(
            sqrt_ratio != 0,
            1.0 / np.where(sqrt_ratio != 0, sqrt_ratio, 1.0) + (1.0 - sqrt_ratio),
            1.0,
        )

    return photon_weights.astype(np.float64)


# ═════════════════════════════════════════════════════════════════════════
# Section 3 ── Log2-spaced lag bin construction (faithful port of
#               generate_log2_lags, shared by both cross_corr variants)
# ═════════════════════════════════════════════════════════════════════════

def _generate_log2_lags(
    t_end: float,
    coarseness: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate log2-spaced multi-tau lag bin edges.

    Faithful port of generate_log2_lags.m.

    Parameters
    ----------
    t_end      : float  maximum lag time (in sync-tick units)
    coarseness : int    number of equally-spaced bins per cascade
                        (MATLAB: nper_cascade)

    Returns
    -------
    lag_bin_edges   : np.ndarray  cumulative bin edges (length n_edges)
    lags            : np.ndarray  bin centres (length n_edges - 1)
    division_factor : np.ndarray  per-bin normalisation divisor
                                  (length n_edges - 1)
    """
    cascade_end = int(np.floor(np.log2(t_end) - 2))
    nper_cascade = coarseness
    n_edges = cascade_end * nper_cascade

    lag_bin_edges = np.zeros(n_edges, dtype=np.float64)
    for j in range(1, n_edges + 1):
        if j == 1:
            lag_bin_edges[j - 1] = 1
        else:
            lag_bin_edges[j - 1] = lag_bin_edges[j - 2] + 2 ** np.floor((j - 1) / nper_cascade)
            # NOTE: MATLAB is 1-indexed: floor((j-1)/nper_cascade) with
            # j the 1-indexed loop variable. In 0-indexed Python with
            # j running 1..n_edges (kept 1-indexed here to mirror MATLAB
            # exactly), the exponent uses (j-1) as MATLAB does.

    lags = np.diff(lag_bin_edges) / 2 + lag_bin_edges[:-1]
    division_factor = np.repeat(
        2.0 ** np.arange(1, cascade_end + 1), nper_cascade
    )

    return lag_bin_edges, lags, division_factor


# ═════════════════════════════════════════════════════════════════════════
# Section 4 ── Photon-counting core  (faithful port of photons_in_bins,
#               both unweighted and weighted variants)
# ═════════════════════════════════════════════════════════════════════════

def _photons_in_bins_python(
    ch1: np.ndarray,
    ch2: np.ndarray,
    lag_bin_edges: np.ndarray,
    cascade_start: int,
    nper_cascade: int,
    offset_lag: float,
) -> np.ndarray:
    """
    Pure-Python (no numba) fallback implementation of photons_in_bins.m
    (unweighted variant). Faithful port of the Laurence et al. 2006
    algorithm with running low/high index pointers per cascade bin.

    See _photons_in_bins_numba for the accelerated, numba-JIT version
    with identical logic (used automatically when numba is available).
    """
    num_ch1 = len(ch1)
    n_edges = len(lag_bin_edges)
    n_bins = n_edges - 1

    low_inds = np.ones(n_bins, dtype=np.int64)
    low_inds[0] = 1  # MATLAB: low_inds(1) = 2 (1-indexed) -> index 1 (0-indexed)
    max_inds = np.zeros(n_bins, dtype=np.int64)
    acf = np.zeros(n_bins, dtype=np.float64)

    n_ch2 = len(ch2)
    k_start = cascade_start * nper_cascade

    for phot_ind in range(num_ch1):
        t1 = ch1[phot_ind]
        bin_edges = t1 + lag_bin_edges + offset_lag

        for k in range(k_start, n_bins):
            while low_inds[k] < n_ch2 and ch2[low_inds[k]] < bin_edges[k]:
                low_inds[k] += 1
            while max_inds[k] < n_ch2 and ch2[max_inds[k]] <= bin_edges[k + 1]:
                max_inds[k] += 1

            if k + 1 < n_bins:
                low_inds[k + 1] = max_inds[k]

            acf[k] += max_inds[k] - low_inds[k]

    return acf


def _photons_in_bins_weighted_python(
    ch1: np.ndarray,
    weights_1: np.ndarray,
    ch2: np.ndarray,
    weights_2: np.ndarray,
    lag_bin_edges: np.ndarray,
    cascade_start: int,
    nper_cascade: int,
    offset_lag: float,
) -> np.ndarray:
    """
    Pure-Python fallback for the weighted variant (cross_corr_weights.m's
    photons_in_bins). Faithful port; uses cumulative-sum lookups on
    weights_2 for O(1) amortised segment sums instead of MATLAB's
    sum(weights_2(low:max-1)), which is mathematically identical but
    faster for repeated overlapping-range sums.
    """
    num_ch1 = len(ch1)
    n_edges = len(lag_bin_edges)
    n_bins = n_edges - 1
    n_ch2 = len(ch2)

    # cumulative sum for O(1) range-sum queries on weights_2
    cumw2 = np.zeros(n_ch2 + 1, dtype=np.float64)
    cumw2[1:] = np.cumsum(weights_2)

    low_inds = np.ones(n_bins, dtype=np.int64)
    low_inds[0] = 1
    max_inds = np.zeros(n_bins, dtype=np.int64)
    acf = np.zeros(n_bins, dtype=np.float64)

    k_start = cascade_start * nper_cascade

    for phot_ind in range(num_ch1):
        t1 = ch1[phot_ind]
        w1 = weights_1[phot_ind]
        bin_edges = t1 + lag_bin_edges + offset_lag

        for k in range(k_start, n_bins):
            while low_inds[k] < n_ch2 and ch2[low_inds[k]] < bin_edges[k]:
                low_inds[k] += 1
            while max_inds[k] < n_ch2 and ch2[max_inds[k]] <= bin_edges[k + 1]:
                max_inds[k] += 1

            if k + 1 < n_bins:
                low_inds[k + 1] = max_inds[k]

            w2_sum = cumw2[max_inds[k]] - cumw2[low_inds[k]]
            acf[k] += w1 * w2_sum

    return acf


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _photons_in_bins_numba(ch1, ch2, lag_bin_edges, cascade_start,
                              nper_cascade, offset_lag):
        num_ch1 = ch1.shape[0]
        n_edges = lag_bin_edges.shape[0]
        n_bins = n_edges - 1
        n_ch2 = ch2.shape[0]

        low_inds = np.ones(n_bins, dtype=np.int64)
        low_inds[0] = 1
        max_inds = np.zeros(n_bins, dtype=np.int64)
        acf = np.zeros(n_bins, dtype=np.float64)

        k_start = cascade_start * nper_cascade

        for phot_ind in range(num_ch1):
            t1 = ch1[phot_ind]

            for k in range(k_start, n_bins):
                edge_lo = t1 + lag_bin_edges[k] + offset_lag
                edge_hi = t1 + lag_bin_edges[k + 1] + offset_lag

                while low_inds[k] < n_ch2 and ch2[low_inds[k]] < edge_lo:
                    low_inds[k] += 1
                while max_inds[k] < n_ch2 and ch2[max_inds[k]] <= edge_hi:
                    max_inds[k] += 1

                if k + 1 < n_bins:
                    low_inds[k + 1] = max_inds[k]

                acf[k] += max_inds[k] - low_inds[k]

        return acf

    @njit(cache=True, fastmath=True)
    def _photons_in_bins_weighted_numba(ch1, weights_1, ch2, weights_2,
                                        lag_bin_edges, cascade_start,
                                        nper_cascade, offset_lag):
        num_ch1 = ch1.shape[0]
        n_edges = lag_bin_edges.shape[0]
        n_bins = n_edges - 1
        n_ch2 = ch2.shape[0]

        cumw2 = np.zeros(n_ch2 + 1, dtype=np.float64)
        for i in range(n_ch2):
            cumw2[i + 1] = cumw2[i] + weights_2[i]

        low_inds = np.ones(n_bins, dtype=np.int64)
        low_inds[0] = 1
        max_inds = np.zeros(n_bins, dtype=np.int64)
        acf = np.zeros(n_bins, dtype=np.float64)

        k_start = cascade_start * nper_cascade

        for phot_ind in range(num_ch1):
            t1 = ch1[phot_ind]
            w1 = weights_1[phot_ind]

            for k in range(k_start, n_bins):
                edge_lo = t1 + lag_bin_edges[k] + offset_lag
                edge_hi = t1 + lag_bin_edges[k + 1] + offset_lag

                while low_inds[k] < n_ch2 and ch2[low_inds[k]] < edge_lo:
                    low_inds[k] += 1
                while max_inds[k] < n_ch2 and ch2[max_inds[k]] <= edge_hi:
                    max_inds[k] += 1

                if k + 1 < n_bins:
                    low_inds[k + 1] = max_inds[k]

                w2_sum = cumw2[max_inds[k]] - cumw2[low_inds[k]]
                acf[k] += w1 * w2_sum

        return acf


def _photons_in_bins(ch1, ch2, lag_bin_edges, cascade_start, nper_cascade, offset_lag):
    """Dispatch to numba-accelerated or pure-Python implementation."""
    if NUMBA_AVAILABLE:
        return _photons_in_bins_numba(
            ch1.astype(np.float64), ch2.astype(np.float64),
            lag_bin_edges.astype(np.float64),
            cascade_start, nper_cascade, offset_lag,
        )
    return _photons_in_bins_python(
        ch1, ch2, lag_bin_edges, cascade_start, nper_cascade, offset_lag
    )


def _photons_in_bins_weighted(ch1, weights_1, ch2, weights_2,
                              lag_bin_edges, cascade_start, nper_cascade, offset_lag):
    """Dispatch to numba-accelerated or pure-Python implementation."""
    if NUMBA_AVAILABLE:
        return _photons_in_bins_weighted_numba(
            ch1.astype(np.float64), weights_1.astype(np.float64),
            ch2.astype(np.float64), weights_2.astype(np.float64),
            lag_bin_edges.astype(np.float64),
            cascade_start, nper_cascade, offset_lag,
        )
    return _photons_in_bins_weighted_python(
        ch1, weights_1, ch2, weights_2,
        lag_bin_edges, cascade_start, nper_cascade, offset_lag
    )


# ═════════════════════════════════════════════════════════════════════════
# Section 5 ── cross_corr / cross_corr_weights  (faithful ports)
# ═════════════════════════════════════════════════════════════════════════

def cross_corr(
    ch1: np.ndarray,
    ch2: np.ndarray,
    start_time: float,
    stop_time: float,
    coarseness: int,
    offset_lag: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Faithful port of cross_corr.m — Laurence et al. 2006 fast
    photon-counting cross-correlation (unweighted).

    Parameters
    ----------
    ch1, ch2   : np.ndarray  photon arrival times (sync-tick units,
                             ascending order)
    start_time : float       minimum lag time to analyse (sync ticks)
    stop_time  : float       maximum lag time to analyse (sync ticks)
    coarseness : int         points per log2 cascade (MATLAB: Sampling)
    offset_lag : float       channel temporal offset (sync ticks)

    Returns
    -------
    corr_norm : np.ndarray  normalised correlation, G(0)+1 convention
                            (NOT yet offset-subtracted -- caller
                            subtracts 1 as in the original script)
    lags      : np.ndarray  lag times in sync-tick units
    """
    cascade_start = int(np.floor(np.log2(start_time) - 2))
    lag_bin_edges, lags, division_factor = _generate_log2_lags(stop_time, coarseness)

    corr = _photons_in_bins(ch1, ch2, lag_bin_edges, cascade_start, coarseness, offset_lag)

    num_ch1 = len(ch1)
    num_ch2 = len(ch2)
    ch1_max = ch1[-1]
    ch2_max = ch2[-1]
    tcor = np.minimum(ch1_max, ch2_max - lags)

    skip_lags = cascade_start * coarseness

    corr_div = corr / division_factor[1:]
    corr_norm = 2.0 * (corr_div / tcor) / ((num_ch1 / ch1_max) * (num_ch2 / ch2_max))

    corr_norm = corr_norm[skip_lags:]
    lags = lags[skip_lags:]

    return corr_norm, lags


def cross_corr_weights(
    ch1: np.ndarray,
    weights_1: np.ndarray,
    ch2: np.ndarray,
    weights_2: np.ndarray,
    start_time: float,
    stop_time: float,
    coarseness: int,
    offset_lag: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Faithful port of cross_corr_weights.m — weighted variant for bleach
    correction, using per-photon weights instead of raw photon counts.
    """
    cascade_start = int(np.floor(np.log2(start_time) - 2))
    lag_bin_edges, lags, division_factor = _generate_log2_lags(stop_time, coarseness)

    corr = _photons_in_bins_weighted(
        ch1, weights_1, ch2, weights_2,
        lag_bin_edges, cascade_start, coarseness, offset_lag
    )

    sum_ch1 = np.sum(weights_1)
    sum_ch2 = np.sum(weights_2)
    ch1_max = ch1[-1]
    ch2_max = ch2[-1]
    tcor = np.minimum(ch1_max, ch2_max - lags)

    skip_lags = cascade_start * coarseness

    corr_div = corr / division_factor[1:]
    corr_norm = 2.0 * (corr_div / tcor) / ((sum_ch1 / ch1_max) * (sum_ch2 / ch2_max))

    corr_norm = corr_norm[skip_lags:]
    lags = lags[skip_lags:]

    return corr_norm, lags


# ═════════════════════════════════════════════════════════════════════════
# Section 6 ── Afterpulsing calibration  (LSM980 4-parameter biexponential)
# ═════════════════════════════════════════════════════════════════════════

def load_lsm980_afterpulsing_params(filepath: str) -> np.ndarray:
    """
    Load LSM980 afterpulsing calibration parameters.

    The MATLAB script loads 'detectorsD118_LSM980.mat', a variable
    `detectors` of shape (n_channels, 4) with columns
    [A1, tau1, A2, tau2] (matching the format used throughout FCSFixer).

    This Python port expects the same layout supplied as a plain CSV
    (no header), one row per channel:
        A1, tau1, A2, tau2

    Parameters
    ----------
    filepath : str  path to calibration CSV

    Returns
    -------
    np.ndarray shape (n_channels, 4)
    """
    params = np.genfromtxt(filepath, delimiter=",", skip_header=0)
    if params.ndim == 1:
        params = params[np.newaxis, :]
    return params.astype(np.float64)


def afterpulse_correction_raw(
    lags: np.ndarray,
    ap_char: np.ndarray,
    cntrate_hz: float,
) -> np.ndarray:
    """
    Afterpulsing correlation term, faithful port of the MATLAB
    anonymous function:

        G_afterpulse = @(Lags, AP_char, Cntrate_Hz) ...
            (AP_char(1)*exp(-Lags/AP_char(2)) + AP_char(3)*exp(-Lags/AP_char(4)))
            ./ (1 + AP_char(1)*AP_char(2) + AP_char(3)*AP_char(4)) ./ Cntrate_Hz;

    Parameters
    ----------
    lags       : np.ndarray  lag times in SECONDS
    ap_char    : np.ndarray  [A1, tau1, A2, tau2] for this channel
    cntrate_hz : float       photon count rate in Hz

    Returns
    -------
    G_ap : np.ndarray
    """
    A1, tau1, A2, tau2 = ap_char
    numerator = A1 * np.exp(-lags / tau1) + A2 * np.exp(-lags / tau2)
    denom = (1.0 + A1 * tau1 + A2 * tau2) * cntrate_hz
    return numerator / denom


# ═════════════════════════════════════════════════════════════════════════
# Section 7 ── Segment-based Wohland SD  (faithful port of the main
#               script's inline segment-correlation + amplitude-matching
#               + std/sqrt(N) block)
# ═════════════════════════════════════════════════════════════════════════

def _amplitude_match_scale(segment_cc: np.ndarray, target_cc: np.ndarray) -> float:
    """
    Find the scalar x minimizing sum((x*segment_cc - target_cc)^2),
    faithful port of the MATLAB fminsearch amplitude-matching step.

    This has a closed-form least-squares solution (no need for
    fminsearch's iterative Nelder-Mead in Python):

        x* = sum(segment_cc * target_cc) / sum(segment_cc^2)
    """
    denom = np.sum(segment_cc ** 2)
    if denom <= 0:
        return 1.0
    return float(np.sum(segment_cc * target_cc) / denom)


def _segment_correlate(
    ch1_seg: np.ndarray,
    w1_seg,
    ch2_seg: np.ndarray,
    w2_seg,
    lagmin_sync: float,
    lagmax_sync: float,
    sampling: int,
    offset_sync: float,
    cross_corr_symm: bool,
    same_channel: bool,
    correct_bleaching: bool,
) -> np.ndarray | None:
    """
    Correlate one time segment, applying the same forward/backward
    symmetric-averaging logic as the main loop, but WITHOUT afterpulsing
    subtraction (afterpulsing subtraction is only applied to the
    full-length curve in the MATLAB script, not to segments).

    Returns None if the segment is too short/empty to correlate.
    """
    if len(ch1_seg) < 2 or len(ch2_seg) < 2:
        return None

    try:
        if not correct_bleaching:
            segment_cc, _ = cross_corr(ch1_seg, ch2_seg, lagmin_sync, lagmax_sync, sampling, offset_sync)
        else:
            segment_cc, _ = cross_corr_weights(
                ch1_seg, w1_seg, ch2_seg, w2_seg, lagmin_sync, lagmax_sync, sampling, offset_sync
            )

        if cross_corr_symm and not same_channel:
            if not correct_bleaching:
                segment_cc2, _ = cross_corr(ch2_seg, ch1_seg, lagmin_sync, lagmax_sync, sampling, offset_sync)
            else:
                segment_cc2, _ = cross_corr_weights(
                    ch2_seg, w2_seg, ch1_seg, w1_seg, lagmin_sync, lagmax_sync, sampling, offset_sync
                )
            segment_cc = (segment_cc + segment_cc2) / 2.0

        return segment_cc
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════
# Section 8 ── Full single-file / single-channel-pair pipeline
# ═════════════════════════════════════════════════════════════════════════

# LSM980 channel name suffix lookup (from the MATLAB switch statement)
_LSM980_CHANNEL_SUFFIX = {
    1: "_ChS1",
    2: "_ChS2",
    3: "_Ch2",
    4: "_GaAsP1",
}


def _load_channel_data(base_readname: str, channel_index: int) -> dict:
    """
    Load one channel's .raw data for a given LSM980 channel index (1-4),
    using the naming convention <base_readname><suffix>.raw.
    """
    suffix = _LSM980_CHANNEL_SUFFIX.get(channel_index)
    if suffix is None:
        raise ValueError(
            f"Undefined channel index {channel_index}. "
            f"Allowed: 1->ChS1, 2->ChS2, 3->Ch2, 4->GaAsP1"
        )
    return read_confocor3_raw(base_readname + suffix + ".raw")


def run_fcs_export_raw(
    filepath: str,
    channels_pairs: list[tuple[int, int]] | None = None,
    channel: int = 1,
    tau_min_s: float = 1e-6,
    tau_max_s: float = 1.0,
    n_bins: int = 12,
    offset_s: float = 0.0,
    symmetric_cc: bool = True,
    n_segments: int = 6,
    use_afterpulsing: bool = False,
    ap_params_path: str = "",
    correct_bleaching: bool = False,
    out_dir_override: str = "",
    progress_callback=None,
    **_ignored_kwargs,
) -> dict:
    """
    Full FCS correlation export pipeline for one Zeiss ConfoCor3/LSM980
    .raw measurement, faithfully following the MATLAB batch script's
    per-file processing loop.

    File naming convention (matching the MATLAB script)
    ─────────────────────────────────────────────────────
    Given `filepath` pointing at e.g.
        ".../Measurement_R1_P0_K1_ChS1.raw"
    the "base readname" (everything before the channel suffix) is
    extracted automatically, and any channel listed in `channels_pairs`
    is loaded from "<base_readname><suffix>.raw" where suffix follows
    the LSM980 lookup table (1->_ChS1, 2->_ChS2, 3->_Ch2, 4->_GaAsP1).

    Parameters
    ----------
    filepath          : str
        Path to ONE channel's .raw file for the measurement (its
        channel suffix is stripped to determine the base readname;
        other channels needed for cross-correlation are loaded
        automatically from sibling files).
    channels_pairs    : list[(int, int)] | None
        List of (channel_1, channel_2) pairs to correlate, using the
        LSM980 1-4 indexing. If None, defaults to [(channel, channel)]
        (autocorrelation of the single channel implied by `channel`).
    channel           : int
        Fallback channel index (1-4) used when channels_pairs is None.
    tau_min_s         : float  minimum lag time in seconds
    tau_max_s         : float  maximum lag time in seconds
    n_bins            : int    points per log2 cascade (MATLAB: Sampling)
    offset_s          : float  channel temporal offset in seconds
    symmetric_cc      : bool   average forward+backward CCF (MATLAB:
                               crossCorrSymm)
    n_segments        : int    number of segments for Wohland SD
                               (MATLAB: nSegments; 1 disables SD calc)
    use_afterpulsing  : bool   subtract calibrated AP pattern (ACF only)
    ap_params_path    : str    path to LSM980 afterpulsing calibration
                               CSV (columns A1, tau1, A2, tau2 per
                               channel row, 1-indexed by channel number)
    correct_bleaching : bool   apply get_blcorr_weights() bleach
                               correction before correlating
    out_dir_override  : str    output directory override (default: a
                               subfolder named after the measurement's
                               base filename, next to the input file)
    progress_callback : callable(float) | None

    Returns
    -------
    dict with keys:
        results          : list[dict]  one entry per channel pair
        fret             : None        (not applicable to .raw / PIE-less data)
        info             : dict        file metadata
        intensity_traces : dict        per-channel intensity traces
        out_dir          : str
    """
    def _prog(pct):
        if progress_callback is not None:
            progress_callback(float(pct))

    _prog(0.0)

    if channels_pairs is None:
        channels_pairs = [(channel, channel)]

    # ── determine base readname by stripping the channel suffix ────────
    filepath = str(filepath)
    stem_with_suffix = os.path.splitext(os.path.basename(filepath))[0]
    base_readname = None
    matched_suffix = None
    for ch_idx, suffix in _LSM980_CHANNEL_SUFFIX.items():
        if stem_with_suffix.endswith(suffix):
            base_readname = stem_with_suffix[: -len(suffix)]
            matched_suffix = suffix
            break
    if base_readname is None:
        raise ValueError(
            f"Could not determine LSM980 channel suffix from filename "
            f"'{stem_with_suffix}'. Expected one of: "
            f"{list(_LSM980_CHANNEL_SUFFIX.values())}"
        )

    in_dir = os.path.dirname(filepath)
    base_path = os.path.join(in_dir, base_readname)

    out_base = out_dir_override if out_dir_override else os.path.join(in_dir, base_readname)
    os.makedirs(out_base, exist_ok=True)

    # ── load all needed channels ─────────────────────────────────────────
    channels_needed = sorted(set(c for pair in channels_pairs for c in pair))
    channel_data = {}
    for ch_idx in channels_needed:
        channel_data[ch_idx] = _load_channel_data(base_path, ch_idx)

    _prog(10.0)

    sync_rate = channel_data[channels_needed[0]]["TTResult_SyncRate"]
    acq_time_ms = channel_data[channels_needed[0]]["MeasDesc_AcquisitionTime"]

    lagmin_sync = tau_min_s * sync_rate
    lagmax_sync = tau_max_s * sync_rate
    offset_sync = offset_s * sync_rate

    # ── optional bleach-correction weights per channel ───────────────────
    channel_weights = {}
    if correct_bleaching:
        for ch_idx in channels_needed:
            channel_weights[ch_idx] = get_blcorr_weights(
                channel_data[ch_idx]["ph_sync"].astype(np.float64)
            )

    # ── afterpulsing calibration (optional) ──────────────────────────────
    ap_params = None
    if use_afterpulsing and os.path.isfile(ap_params_path):
        ap_params = load_lsm980_afterpulsing_params(ap_params_path)

    n_pairs = len(channels_pairs)
    results = []

    for i_pair, (ch1_idx, ch2_idx) in enumerate(channels_pairs):
        _prog(10.0 + 80.0 * i_pair / n_pairs)

        same_channel = (ch1_idx == ch2_idx)
        arr1 = channel_data[ch1_idx]["ph_sync"].astype(np.float64)
        arr2 = channel_data[ch2_idx]["ph_sync"].astype(np.float64)
        w1 = channel_weights.get(ch1_idx) if correct_bleaching else None
        w2 = channel_weights.get(ch2_idx) if correct_bleaching else None

        try:
            if not correct_bleaching:
                cc_raw, lags = cross_corr(arr1, arr2, lagmin_sync, lagmax_sync, n_bins, offset_sync)
            else:
                cc_raw, lags = cross_corr_weights(
                    arr1, w1, arr2, w2, lagmin_sync, lagmax_sync, n_bins, offset_sync
                )

            lags_s = lags / sync_rate

            # ── offset subtraction + afterpulsing / symmetry handling ────
            ap_used = False
            if (
                use_afterpulsing and ap_params is not None and same_channel
                and (ap_params[ch1_idx - 1, 0] != 0 or ap_params[ch1_idx - 1, 2] != 0)
            ):
                cntrate_hz = (
                    (np.sum(w1) if correct_bleaching else len(arr1))
                    / acq_time_ms * 1e3
                )
                G_ap = afterpulse_correction_raw(lags_s, ap_params[ch1_idx - 1], cntrate_hz)
                cc = cc_raw - 1.0 - G_ap
                ap_used = True

            elif symmetric_cc and not same_channel:
                if not correct_bleaching:
                    cc_raw2, _ = cross_corr(arr2, arr1, lagmin_sync, lagmax_sync, n_bins, offset_sync)
                else:
                    cc_raw2, _ = cross_corr_weights(
                        arr2, w2, arr1, w1, lagmin_sync, lagmax_sync, n_bins, offset_sync
                    )
                cc = (cc_raw + cc_raw2) / 2.0 - 1.0

            else:
                cc = cc_raw - 1.0

            # ── segment-based Wohland SD ──────────────────────────────────
            sd_cc = None
            if n_segments > 1:
                segment_length = acq_time_ms / 1e3 / n_segments * sync_rate
                segment_ccs = []

                for m in range(n_segments):
                    lo = segment_length * m
                    hi = segment_length * (m + 1)

                    mask1 = (arr1 >= lo) & (arr1 < hi)
                    mask2 = (arr2 >= lo) & (arr2 < hi)
                    ch1_seg = arr1[mask1] - lo
                    ch2_seg = arr2[mask2] - lo
                    w1_seg = w1[mask1] if correct_bleaching else None
                    w2_seg = w2[mask2] if correct_bleaching else None

                    seg_cc = _segment_correlate(
                        ch1_seg, w1_seg, ch2_seg, w2_seg,
                        lagmin_sync, lagmax_sync, n_bins, offset_sync,
                        symmetric_cc, same_channel, correct_bleaching,
                    )
                    if seg_cc is not None and len(seg_cc) == len(cc_raw):
                        scale = _amplitude_match_scale(seg_cc, cc_raw)
                        segment_ccs.append(seg_cc * scale)

                if len(segment_ccs) >= 2:
                    seg_arr = np.stack(segment_ccs, axis=1)
                    sd_cc = np.std(seg_arr, axis=1, ddof=0) / np.sqrt(len(segment_ccs))

            if sd_cc is None:
                sd_cc = np.zeros_like(cc)

            # ── count rates ────────────────────────────────────────────────
            if symmetric_cc and not same_channel:
                cntrate = (
                    (np.sum(w1) + np.sum(w2)) if correct_bleaching
                    else (len(arr1) + len(arr2))
                ) / acq_time_ms * 1e3
                cntrate1 = cntrate2 = cntrate
            else:
                cntrate1 = (np.sum(w1) if correct_bleaching else len(arr1)) / acq_time_ms * 1e3
                cntrate2 = (np.sum(w2) if correct_bleaching else len(arr2)) / acq_time_ms * 1e3

            # ── export CSV (MATLAB layout: skip first lag point [1:end]) ───
            label = f"ch{ch1_idx}ch{ch2_idx}"
            tags = ""
            if correct_bleaching:
                tags += "_bl"
            if ap_used:
                tags += "_ap"
            tags += "_corr"

            csv_path = os.path.join(out_base, f"{label}{tags}.csv")

            acr_col = np.zeros(len(lags_s) - 1)
            acr_col[0] = cntrate1

            pd.DataFrame({
                "tau_s":   lags_s[1:],
                "G":       cc[1:],
                "acr_Hz":  acr_col,
                "sigma_G": sd_cc[1:],
            }).to_csv(csv_path, index=False, header=False)

            results.append({
                "label":         label,
                "csv_path":      csv_path,
                "lag_s":         lags_s[1:],
                "G":             cc[1:],
                "sigma_G":       sd_cc[1:],
                "acr1_Hz":       cntrate1,
                "acr2_Hz":       cntrate2,
                "is_acf":        same_channel,
                "flcs_used":     False,   # never available for .raw
                "ap_used":       ap_used,
                "tcspc_csv":     None,    # no TCSPC info for .raw
                "tcspc_csv_cs2": None,
            })

        except Exception:
            import traceback as _tb
            results.append({
                "label":    f"ch{ch1_idx}ch{ch2_idx}",
                "error":    _tb.format_exc(),
                "csv_path": None,
                "lag_s":    np.array([]),
                "G":        np.array([]),
                "sigma_G":  np.array([]),
                "acr1_Hz":  0.0,
                "acr2_Hz":  0.0,
                "is_acf":   same_channel,
                "flcs_used": False,
                "ap_used":  False,
                "tcspc_csv": None,
                "tcspc_csv_cs2": None,
            })

    _prog(95.0)

    # ── intensity traces for display ─────────────────────────────────────
    intensity_traces = {}
    for ch_idx in channels_needed:
        arr = channel_data[ch_idx]["ph_sync"].astype(np.float64)
        if len(arr) == 0:
            continue
        acq_s = float(arr[-1]) / sync_rate
        n_trace_bins = min(500, max(1, int(acq_s * 10)))
        bin_edges = np.linspace(0, arr[-1], n_trace_bins + 1)
        counts, _ = np.histogram(arr, bins=bin_edges)
        bin_width_s = (bin_edges[1] - bin_edges[0]) / sync_rate
        t_s = (bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2.0) / sync_rate
        cps = counts / bin_width_s if bin_width_s > 0 else counts
        intensity_traces[f"ch{ch_idx}"] = {
            "t_s": t_s, "cps": cps, "channel": ch_idx
        }

    info = {
        "n_photons":          {ch: len(channel_data[ch]["ph_sync"]) for ch in channels_needed},
        "sync_rate_hz":       sync_rate,
        "acquisition_time_s": acq_time_ms / 1e3,
        "routing_channels":   channels_needed,
        "base_readname":      base_readname,
    }

    _prog(100.0)

    return {
        "results":          results,
        "fret":             None,
        "info":             info,
        "intensity_traces": intensity_traces,
        "out_dir":          out_base,
    }


# ═════════════════════════════════════════════════════════════════════════
# Section 9 ── batch helper
# ═════════════════════════════════════════════════════════════════════════

def run_fcs_export_raw_batch(
    filepaths: list,
    progress_queue=None,
    cancel_event=None,
    cpu_n: int = 1,
    **kwargs,
) -> dict:
    """
    Run run_fcs_export_raw() over multiple .raw files, sequentially or
    in parallel (cpu_n > 1), mirroring the ptu_correlate batch runner.
    """
    n_total = len(filepaths)
    n_ok = 0
    failed = []
    last_res = None

    if progress_queue is not None:
        progress_queue.put(("progress", 0.0))

    if n_total == 0:
        return {"n_total": 0, "n_ok": 0, "n_failed": 0, "failed": [], "last_res": None}

    cpu_n = max(1, int(cpu_n))

    if cpu_n <= 1 or n_total <= 1:
        for i, fp in enumerate(filepaths):
            if cancel_event is not None and cancel_event.is_set():
                break
            try:
                def _cb(pct, _i=i, _n=n_total):
                    if progress_queue is not None:
                        progress_queue.put(("progress", 100.0 * (_i + pct / 100.0) / _n))
                res = run_fcs_export_raw(fp, progress_callback=_cb, **kwargs)
                last_res = res
                n_ok += 1
                if progress_queue is not None:
                    progress_queue.put(("file_done", res))
            except Exception:
                import traceback as _tb
                failed.append(fp)
                if progress_queue is not None:
                    progress_queue.put(("file_error", f"FAILED: {os.path.basename(fp)}\n{_tb.format_exc()}"))
            if progress_queue is not None:
                progress_queue.put(("progress", 100.0 * (i + 1) / max(1, n_total)))

        return {"n_total": n_total, "n_ok": n_ok, "n_failed": len(failed), "failed": failed, "last_res": last_res}

    import multiprocessing as mp

    def _wrapper(args):
        fp, kw = args
        try:
            return (fp, run_fcs_export_raw(fp, progress_callback=None, **kw), None)
        except Exception:
            import traceback as _tb
            return (fp, None, _tb.format_exc())

    cpu_n = min(cpu_n, n_total, mp.cpu_count())
    completed = 0
    with mp.Pool(processes=cpu_n) as pool:
        for fp, res, err in pool.imap(_wrapper, [(fp, kwargs) for fp in filepaths], chunksize=1):
            if cancel_event is not None and cancel_event.is_set():
                pool.terminate()
                pool.join()
                break
            if err is None:
                last_res = res
                n_ok += 1
                if progress_queue is not None:
                    progress_queue.put(("file_done", res))
            else:
                failed.append(fp)
                if progress_queue is not None:
                    progress_queue.put(("file_error", f"FAILED: {os.path.basename(fp)}\n{err}"))
            completed += 1
            if progress_queue is not None:
                progress_queue.put(("progress", 100.0 * completed / max(1, n_total)))

    return {"n_total": n_total, "n_ok": n_ok, "n_failed": len(failed), "failed": failed, "last_res": last_res}   