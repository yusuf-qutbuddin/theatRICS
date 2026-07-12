"""
theatrics/fcsfit/ptu_correlate.py

Thin wrapper around fcs_fixer_core.FCS_Fixer, exposing the API the
GUI/worker expect (run_fcs_export, run_fcs_export_batch).

All correlation, afterpulsing subtraction, and FLCS background
correction math is delegated entirely to FCS_Fixer — this file only:

  - loads the PTU file into a tttrlib.TTTR object
  - builds channels_spec objects for PIE (donor/acceptor prompt/delay)
    using FCS_Fixer.build_channels_spec() / FCS_Fixer.check_channels_spec()
  - drives the sequence of calls per correlation pair:
        (optional FLCS filter setup) -> get_correlation_uncertainty()
  - computes FRET photon counts (not part of FCS_Fixer; specific to
    this tool) via FCS_Fixer.select_photons()
  - builds intensity traces for display (not part of FCS_Fixer either)
  - writes CSV / summary output

See fcs_fixer_core.py's module docstring for the exact list of
tttrlib-version-compatibility changes made relative to the original
FCS_Fixer implementation. Nothing in that file's actual algorithms
(select_photons, correlation_apply_filters, afterpulse_correlation,
get_Wohland_SD, get_bootstrap_SD, get_tcspc_histogram,
get_background_tail_fit, get_flcs_background_filter,
get_flcs_filters) was changed beyond those compatibility fixes.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from theatrics.fcsfit.fcs_fixer_core import FCS_Fixer, TTTRLIB_AVAILABLE
from theatrics.fcsfit.zeiss_raw_correlate import is_raw_file, run_fcs_export_raw

try:
    import tttrlib
except ImportError:
    pass


# ═════════════════════════════════════════════════════════════════════════
# Loading
# ═════════════════════════════════════════════════════════════════════════

def load_ptu(filepath: str):
    """
    Load a PicoQuant PTU file into a tttrlib.TTTR object.
    """
    if not TTTRLIB_AVAILABLE:
        raise ImportError(
            "tttrlib is not installed. Install it with: pip install tttrlib"
        )
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PTU file not found: {filepath}")
    return tttrlib.TTTR(str(filepath), "PTU")


def get_ptu_info(tttr_data) -> dict:
    """
    Extract key metadata from a loaded TTTR object.

    NOTE: tttrlib's header.macro_time_resolution / micro_time_resolution
    are returned in SECONDS in the current version (confirmed empirically
    -- see fcs_fixer_core.py FCS_Fixer.__init__ for the corresponding fix
    applied there). Converted to ns here for consistent internal units
    and human-readable display.
    """
    macro_res_ns = float(tttr_data.header.macro_time_resolution) * 1e9   # CHANGED: was missing * 1e9
    micro_res_ns = float(tttr_data.header.micro_time_resolution) * 1e9   # CHANGED: was missing * 1e9
    n_mt_bins    = int(tttr_data.get_number_of_micro_time_channels())
    macro_times  = tttr_data.macro_times
    channels     = sorted(int(c) for c in np.unique(tttr_data.routing_channels))
    acq_time_s   = float(np.max(macro_times)) * macro_res_ns * 1e-9   # macro_res_ns is now correctly in ns

    photons_per_ch = {
        ch: int(tttr_data.get_selection_by_channel([ch]).shape[0])
        for ch in channels
    }

    return {
        "n_photons":           int(macro_times.shape[0]),
        "macro_time_res_ns":   macro_res_ns,
        "micro_time_res_ns":   micro_res_ns,
        "n_micro_time_bins":   n_mt_bins,
        "acquisition_time_s":  acq_time_s,
        "routing_channels":    channels,
        "photons_per_channel": photons_per_ch,
    }


# ═════════════════════════════════════════════════════════════════════════
# PIE channel construction — via FCS_Fixer's OWN build_channels_spec
# ═════════════════════════════════════════════════════════════════════════

def build_pie_channels(
    donor_channel:    int,
    acceptor_channel: int,
    prompt_gate:      tuple = (0.0, 0.5),
    delay_gate:       tuple = (0.5, 1.0),
) -> dict:
    """
    Build the six channels_spec objects needed for a two-colour PIE
    experiment, using FCS_Fixer.build_channels_spec() exactly as the
    original code would.

    FCS_Fixer.build_channels_spec(channel, micro_time_gates) expects
    micro_time_gates as a FLAT list [start_0, stop_0, start_1, stop_1, ...],
    NOT a list of tuples — reproduced exactly here.

    Returns
    -------
    dict with keys:
        donor_prompt, donor_delay, donor_all,
        acceptor_prompt, acceptor_delay, acceptor_all
    """
    pg = [float(prompt_gate[0]), float(prompt_gate[1])]
    dg = [float(delay_gate[0]),  float(delay_gate[1])]

    return {
        "donor_prompt":    FCS_Fixer.build_channels_spec(donor_channel,    pg),
        "donor_delay":     FCS_Fixer.build_channels_spec(donor_channel,    dg),
        "donor_all":       FCS_Fixer.build_channels_spec(donor_channel),
        "acceptor_prompt": FCS_Fixer.build_channels_spec(acceptor_channel, pg),
        "acceptor_delay":  FCS_Fixer.build_channels_spec(acceptor_channel, dg),
        "acceptor_all":    FCS_Fixer.build_channels_spec(acceptor_channel),
    }


# ═════════════════════════════════════════════════════════════════════════
# FLCS setup helper
#
# Reproduces exactly the sequence FCS_Fixer.run_standard_pipeline() uses
# internally for its FLCS background-correction block:
#
#     tcspc_x, tcspc_y = get_tcspc_histogram(channels_spec, ...)
#     peak_position = argmax(tcspc_y)
#     fit_start = peak_position + ceil(2 ns / micro_time_resolution)
#     flat_background, _ = get_background_tail_fit(...)
#     patterns_norm_full, flcs_weights_full = get_flcs_background_filter(...)
#
# ONE deviation from the original (documented in fcs_fixer_core.py's
# module docstring): the original FCS_Fixer.run_standard_pipeline()
# determines peak_position the same way (np.argmax(tcspc_y) of the SAME
# channel's own TCSPC histogram) — so this is actually NOT a deviation
# at all; run_standard_pipeline() never calls find_IRF_position() for
# the background-correction step, only its (unused-here) IRF-fitting
# convenience method does. So this function is a faithful reproduction
# of run_standard_pipeline()'s own FLCS block.
# ═════════════════════════════════════════════════════════════════════════

def _ungated_channels_spec(channels_spec):
    """
    Strip any micro-time gate from a channels_spec, keeping only the
    routing channel selection.

    Needed because: computing the FLCS background filter requires an
    exponential tail fit that needs enough TCSPC bins AFTER the peak to
    converge. If channels_spec carries a PIE gate (e.g. the "prompt"
    half of the period), the peak can legitimately fall close to the
    END of that gated window, leaving zero bins for the tail fit within
    the gate alone -- this is exactly what caused the
    "zero-size array to reduction operation minimum" crash.

    This situation never arises in the original FCS_Fixer, because its
    run_standard_pipeline() always calls the FLCS block with a plain,
    ungated channels_spec. Combining PIE micro-time gating with FLCS
    background correction on the SAME gated spec is a new combination
    specific to this PIE-FCCS tool, so this adaptation is required.
    """
    cs_norm  = FCS_Fixer.check_channels_spec(channels_spec)
    channels = cs_norm[0]
    return FCS_Fixer.build_channels_spec(list(channels))

def plot_flcs_diagnostic(
    patterns_norm_full: np.ndarray,
    flcs_weights_full:  np.ndarray,
    title:              str = "",
    filter_ylim:        tuple = (-3.0, 4.0),
) -> "plt.Figure":
    """
    Reproduce FCS_Fixer's own two-panel FLCS diagnostic figure:

        top panel    : normalized micro-time patterns (signal vs
                       background), log y-scale, scatter points
        bottom panel : the resulting FLCS filter functions (signal vs
                       background), linear y-scale, line plot

    Both panels share the x-axis (TCSPC bin index).

    Parameters
    ----------
    patterns_norm_full : np.ndarray, shape (n_micro_time_bins, 2)
        Column 0 = normalized signal pattern, column 1 = normalized
        background pattern. Bins not used in the filter fit are zero
        and are simply not drawn (log scale cannot show zero anyway).
    flcs_weights_full : np.ndarray, shape (n_micro_time_bins, 2)
        Column 0 = signal filter weight, column 1 = background filter
        weight, per TCSPC bin.
    title : str
        Optional suffix appended to both panel titles (e.g. channel or
        ACF/CCF label), so multiple diagnostic plots stay distinguishable.
    filter_ylim : (float, float)
        Y-axis display range for the bottom (filter functions) panel.
        Matches FCS_Fixer's own reference figure convention: filter
        weights are shown clipped to a fixed visual range rather than
        auto-scaled, since a small number of noisy bins can otherwise
        dominate the axis scaling.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n_bins = patterns_norm_full.shape[0]
    x = np.arange(n_bins)

    signal_pattern     = patterns_norm_full[:, 0]
    background_pattern = patterns_norm_full[:, 1]
    signal_filter       = flcs_weights_full[:, 0]
    background_filter   = flcs_weights_full[:, 1]

    # log-scale scatter cannot show exact zeros -- mask them out so
    # unused bins simply leave a gap, matching the reference figure
    sig_mask = signal_pattern > 0
    bg_mask  = background_pattern > 0

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # ── top panel: normalized micro-time patterns ─────────────────────
    ax_top.semilogy(
        x[sig_mask], signal_pattern[sig_mask],
        "o", color="black", markersize=3, label="Signal"
    )
    ax_top.semilogy(
        x[bg_mask], background_pattern[bg_mask],
        "o", color="gray", markersize=3, alpha=0.8, label="Background"
    )
    ax_top.set_title(f"Normalized micro time patterns{(' — ' + title) if title else ''}")
    ax_top.legend(loc="upper right", frameon=True)
    ax_top.tick_params(axis="x", labelbottom=False)

    # ── bottom panel: FLCS filter functions ───────────────────────────
    ax_bot.plot(x, signal_filter, "-", color="black", linewidth=1.0)
    ax_bot.plot(x, background_filter, "-", color="gray", linewidth=1.0)
    ax_bot.axhline(0, color="lightgray", linewidth=0.6, zorder=0)
    ax_bot.set_title(f"FLCS filter functions{(' — ' + title) if title else ''}")
    ax_bot.set_xlabel("TCSPC bins")
    ax_bot.set_ylim(*filter_ylim)
    ax_bot.set_xlim(0, n_bins)

    fig.tight_layout()
    return fig

def _apply_burst_removal_for_channel(
    fixer: FCS_Fixer, channels_spec, threshold_alpha: float = 0.02,
) -> dict:
    """
    Compute and apply burst-removal weights for one channels_spec
    (normally a FULL, ungated channel — see run_fcs_export for why),
    using FCS_Fixer's auto-thresholding and photon-level burst masking.

    Also reconstructs the burst time intervals (in seconds) for GUI
    display purposes, by grouping contiguous True runs in the
    thresholded trace into (start_s, stop_s) pairs.

    Returns
    -------
    dict with keys:
        n_burst_bins       : int    number of trace bins flagged as burst
        n_total_bins       : int    total number of trace bins
        burst_intervals_s  : list[(float, float)]  burst time windows, in
                             seconds, relative to the first photon of
                             this channel
        threshold_counts   : int | None  threshold actually used (None
                             if burst removal could not run at all)
        sampling_s         : float | None  bin width used, in seconds
    """
    empty_result = {
        "n_burst_bins": 0, "n_total_bins": 0,
        "burst_intervals_s": [], "threshold_counts": None, "sampling_s": None,
    }

    macro_times, _, _, _ = fixer.select_photons(channels_spec)
    if len(macro_times) < 2:
        return empty_result

    acq_time_ns = float(macro_times[-1] - macro_times[0]) * fixer._macro_time_resolution
    if acq_time_ns <= 0:
        return empty_result

    sampling_s = fixer.get_trace_time_scale(channels_spec)

    min_bins_required = 5
    if sampling_s * 1e9 * min_bins_required > acq_time_ns:
        sampling_s = (acq_time_ns / min_bins_required) * 1e-9

    trace, bin_centers_s = fixer.get_time_trace(channels_spec, sampling_s)
    if len(trace) < 3:
        return empty_result

    try:
        burst_bins, _, threshold_counts = fixer.run_burst_removal(
            trace, sampling_s, threshold_alpha=threshold_alpha,
        )
    except Exception:
        return empty_result

    burst_intervals_s = []
    if np.any(burst_bins):
        idx = np.where(burst_bins)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        groups = np.split(idx, splits + 1)
        for g in groups:
            start_s = float(bin_centers_s[g[0]] - sampling_s / 2.0)
            stop_s  = float(bin_centers_s[g[-1]] + sampling_s / 2.0)
            burst_intervals_s.append((start_s, stop_s))

    return {
        "n_burst_bins":      int(np.sum(burst_bins)),
        "n_total_bins":      int(len(burst_bins)),
        "burst_intervals_s": burst_intervals_s,
        "threshold_counts":  threshold_counts,
        "sampling_s":        sampling_s,
    }
def _apply_undrift_for_channel(fixer: FCS_Fixer, channels_spec) -> int:
    """
    Compute and apply the bleaching/drift-correction weights for one
    channels_spec, using FCS_Fixer's polynomial_undrifting_rss with
    auto time-scale (get_trace_time_scale) and auto polynomial-degree
    selection.

    Writes results directly into fixer._weights_undrift (via
    polynomial_undrifting_rss's update_undrift_weights=True default).

    Returns
    -------
    int  the polynomial degree actually chosen (for logging/diagnostics),
        or 0 if there was too little data to compute a meaningful trend
        (weights are left at their default of 1.0 in that case, i.e.
        bleach correction is effectively skipped for this channel).
    """
    # FIXED: get_trace_time_scale() can pick a sampling interval so
    # coarse (relative to a short/dim/empty acquisition, e.g. an
    # unused or near-empty channel) that the resulting time trace has
    # only 1 bin -- np.arange(0, acq_time, sampling) then produces a
    # bin-edges array of length 1, and get_time_trace()'s
    # time_trace_bins[1] indexing crashes with an IndexError. Detect
    # this up front using the SAME channels_spec/photon selection that
    # get_time_trace() itself will use, and fall back to a coarser but
    # bin-count-safe sampling interval if needed.
    macro_times, _, _, _ = fixer.select_photons(channels_spec)
    if len(macro_times) < 2:
        # no usable photons at all for this channel -- nothing to correct
        return 0

    acq_time_ns = float(macro_times[-1] - macro_times[0]) * fixer._macro_time_resolution
    if acq_time_ns <= 0:
        return 0

    sampling_s = fixer.get_trace_time_scale(channels_spec)

    min_bins_required = 5
    if sampling_s * 1e9 * min_bins_required > acq_time_ns:
        sampling_s = (acq_time_ns / min_bins_required) * 1e-9

    trace, bin_centers_s = fixer.get_time_trace(channels_spec, sampling_s)

    if len(trace) < 3:
        # still too little data to fit a meaningful trend -- leave
        # weights at their default of 1.0 (no correction applied)
        return 0

    _, degree_used = fixer.polynomial_undrifting_rss(
        trace, bin_centers_s, channels_spec
    )
    return degree_used
def _apply_flcs_for_channel(fixer: FCS_Fixer, channels_spec, use_drift_correction: bool = False, use_burst_removal: bool = False,) -> dict:
    full_spec = _ungated_channels_spec(channels_spec)

    tcspc_x, tcspc_y = fixer.get_tcspc_histogram(full_spec, use_drift_correction=use_drift_correction, use_burst_removal=use_burst_removal,)

    peak_position = int(tcspc_x[np.argmax(tcspc_y)])
    fit_start = np.uint64(
        peak_position + np.ceil(2.0 / fixer._micro_time_resolution)
    )
    if fit_start > tcspc_x.max():
        fit_start = np.uint64(min(peak_position + 1, int(tcspc_x.max())))

    flat_background, tail_fit, _, _ = fixer.get_background_tail_fit(
        full_spec, peak_position, fit_start, use_drift_correction=use_drift_correction, use_burst_removal=use_burst_removal,
    )

    patterns_norm_full, flcs_weights_full = fixer.get_flcs_background_filter(
        tcspc_x, tcspc_y, flat_background, channels_spec,
        handle_outside="zero", update_weights=True,
    )

    gate_mask    = fixer._get_micro_time_mask(channels_spec)
    display_mask = gate_mask[tcspc_x]

    return {
        "tcspc_x":              tcspc_x[display_mask],
        "tcspc_y":              tcspc_y[display_mask],
        "peak_position":        peak_position,
        "flat_background":      flat_background,
        "pattern_signal":       patterns_norm_full[tcspc_x[display_mask], 0],      # NEW
        "pattern_background":  patterns_norm_full[tcspc_x[display_mask], 1],      # NEW
        "filter_signal":        flcs_weights_full[tcspc_x[display_mask], 0],
        "filter_background":   flcs_weights_full[tcspc_x[display_mask], 1],       # NEW
        "patterns_norm_full":   patterns_norm_full,
        "flcs_weights_full":    flcs_weights_full,
        "n_micro_time_bins":    fixer._n_micro_time_bins,
    }


# ═════════════════════════════════════════════════════════════════════════
# Intensity trace (display convenience — not part of FCS_Fixer's API)
# ═════════════════════════════════════════════════════════════════════════

def get_intensity_trace(
    fixer:            FCS_Fixer,
    channels_spec,
    n_points_target:      int  = 500,
    use_drift_correction: bool = False,   # NEW
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin photon macro times into a count-rate-vs-time trace for display,
    using FCS_Fixer.select_photons() for consistent channel/gate handling.

    Parameters
    ----------
    use_drift_correction : bool
        If True, weights each photon by fixer._weights_undrift (the
        per-photon bleach/drift correction weights computed by
        _apply_undrift_for_channel() / polynomial_undrifting_rss()).
        If bleach correction was never run for this channel, the
        weights default to 1.0 and this simply reproduces the raw
        trace.

    Returns
    -------
    t_s  : np.ndarray  bin-centre times in seconds
    cps  : np.ndarray  count rate per bin in Hz
    """
    macro_times, _, weights, _ = fixer.select_photons(
        channels_spec, use_drift_correction=use_drift_correction
    )

    if len(macro_times) == 0:
        return np.array([]), np.array([])

    t0 = int(macro_times[0])
    mt_shifted_ns = (macro_times - t0).astype(np.float64) * fixer._macro_time_resolution
    acq_s = float(mt_shifted_ns[-1]) * 1e-9

    if acq_s <= 0:
        return np.array([]), np.array([])

    bin_width_s  = max(acq_s / n_points_target, 1e-3)
    bin_width_ns = bin_width_s * 1e9
    n_bins       = max(1, int(np.ceil(mt_shifted_ns[-1] / bin_width_ns)))
    bin_edges_ns = np.arange(0, (n_bins + 1) * bin_width_ns, bin_width_ns)

    counts, _ = np.histogram(mt_shifted_ns, bins=bin_edges_ns, weights=weights)

    t_s = (bin_edges_ns[:-1] + bin_width_ns / 2.0) * 1e-9
    cps = counts / bin_width_s

    return t_s, cps


# ═════════════════════════════════════════════════════════════════════════
# PIE FRET (photon counts only, via FCS_Fixer.select_photons)
# ═════════════════════════════════════════════════════════════════════════

def compute_pie_fret(
    fixer:             FCS_Fixer,
    pie_channels:      dict,
    gamma:             float = 1.0,
    crosstalk:         float = 0.0,
    direct_excitation: float = 0.0,
) -> dict:
    """
    Compute PIE-corrected FRET efficiency and stoichiometry from photon
    counts in the three relevant PIE windows.

    Based on:
        Lee et al., Biophys J 2005, DOI: 10.1529/biophysj.104.053017

    F_DD : donor channel,   prompt window
    F_DA : acceptor channel, prompt window
    F_AA : acceptor channel, delay window

    This is NOT part of FCS_Fixer — it is specific to this tool's PIE
    workflow, built on top of FCS_Fixer.select_photons() for consistent
    channel/gate handling.
    """
    def _count(key: str) -> float:
        _, _, w, _ = fixer.select_photons(pie_channels[key])
        return float(np.sum(w))

    def _acr(key: str) -> float:
        mt, _, w, _ = fixer.select_photons(pie_channels[key])
        if len(mt) == 0:
            return 0.0
        span_ns = float(np.max(mt) - np.min(mt)) * fixer._macro_time_resolution
        if span_ns <= 0:
            return 0.0
        return float(np.sum(w)) / (span_ns * 1e-9)

    F_DD = _count("donor_prompt")
    F_DA = _count("acceptor_prompt")
    F_AA = _count("acceptor_delay")

    F_DA_corr = max(F_DA - crosstalk * F_DD - direct_excitation * F_AA, 0.0)

    denom_E  = F_DA_corr + gamma * F_DD
    E        = F_DA_corr / denom_E if denom_E > 0 else float("nan")

    denom_PR = F_DA + F_DD
    PR       = F_DA / denom_PR if denom_PR > 0 else float("nan")

    denom_S  = F_DA_corr + gamma * F_DD + F_AA
    S        = (F_DA_corr + gamma * F_DD) / denom_S if denom_S > 0 else float("nan")

    return {
        "F_DD":             F_DD,
        "F_DA":             F_DA,
        "F_AA":             F_AA,
        "F_DA_corrected":   F_DA_corr,
        "proximity_ratio":  PR,
        "FRET_efficiency":  E,
        "stoichiometry":    S,
        "donor_acr_Hz":     _acr("donor_prompt"),
        "acceptor_acr_Hz":  _acr("acceptor_delay"),
    }


# ═════════════════════════════════════════════════════════════════════════
# helper: are two channels_spec objects referring to the same routing
# channel(s)? Used to decide whether afterpulsing subtraction actually
# applies to a given pair (it does whenever channel_1 == channel_2,
# regardless of whether the micro-time gates differ — see
# correlation_apply_filters branches 2/3/4 in fcs_fixer_core.py).
# ═════════════════════════════════════════════════════════════════════════

def _same_channels(cs1, cs2) -> bool:
    cs1n = FCS_Fixer.check_channels_spec(cs1)
    cs2n = FCS_Fixer.check_channels_spec(cs2)
    return cs1n[0] == cs2n[0]


# ═════════════════════════════════════════════════════════════════════════
# Full single-file export pipeline
# ═════════════════════════════════════════════════════════════════════════


def run_fcs_export(
    filepath:           str,
    cs1                = None,
    cs2                = None,
    channel:            int = 0,
    use_pie:            bool  = False,
    donor_channel:      int   = 0,
    acceptor_channel:   int   = 1,
    prompt_gate:        tuple = (0.0, 0.5),
    delay_gate:         tuple = (0.5, 1.0),
    symmetric_cc:       bool  = True,
    gamma:              float = 1.0,
    crosstalk:          float = 0.0,
    direct_excitation:  float = 0.0,
    tau_min_s:          float = 1e-6,
    tau_max_s:          float = 1.0,
    n_bins:             int   = 9,
    use_afterpulsing:   bool  = False,
    ap_params_path:     str   = "",
    use_flcs_bg:        bool  = False,
    correct_bleaching:  bool  = False,   
    use_burst_removal:  bool  = False,      
    burst_threshold_alpha: float = 0.02,     
    compute_dd:         bool  = True,    #  channel-1 autocorrelation
    compute_aa:         bool  = True,    #  channel-2 autocorrelation
    compute_da:         bool  = True,
    save_flcs_diagnostic: bool = False,
    wohland_window_s          = None,
    n_bootstrap:        int   = 20,
    out_dir_override:   str   = "",
    progress_callback         = None,
    **_ignored_kwargs,
) -> dict:
    """
    Full FCS / PIE-FCCS export pipeline for one PTU file, built entirely
    on top of fcs_fixer_core.FCS_Fixer.

    Output location
    ────────────────
    Unless out_dir_override is given, all output files are written into
    a NEW SUBFOLDER placed right next to the input file, named after the
    input file's own stem (filename without extension):

        /path/to/data/Measurement_1.ptu
        /path/to/data/Measurement_1/            <- created automatically
            donor_ACF.csv
            acceptor_ACF.csv
            PIE_CCF.csv
            donor_ACF_tcspc.csv
            acceptor_ACF_tcspc.csv
            donor_ACF_flcs_diagnostic.png
            acceptor_ACF_flcs_diagnostic.png
            PIE_CCF_tcspc.csv
            PIE_CCF_tcspc_cs2.csv
            PIE_CCF_flcs_diagnostic.png
            PIE_CCF_flcs_diagnostic_cs2.png
            FRET_summary.csv
            overview.svg

    Naming convention
    ─────────────────
    Every per-pair output file starts with the correlation pair's own
    label (donor_ACF / acceptor_ACF / PIE_CCF / ACF / CCF), optionally
    followed by correction tags (_apcorr if afterpulsing subtraction
    was applied, _flcsbg if FLCS background correction was applied),
    then a fixed suffix describing the file's content:
        <label>[_apcorr][_flcsbg].csv        main correlation curve
        <label>_tcspc.csv                    TCSPC histogram + FLCS filter (cs1)
        <label>_tcspc_cs2.csv                same, for cs2 (cross-corr only)
        <label>_flcs_diagnostic.png          2-panel FLCS diagnostic (cs1)
        <label>_flcs_diagnostic_cs2.png      same, for cs2 (cross-corr only)
    File-level outputs (not tied to one correlation pair):
        FRET_summary.csv                     PIE mode only
        overview.svg                         combined display figure

    Returns
    -------
    dict with keys:
        results          : list[dict]  one entry per correlation pair
        fret             : dict | None  FRET metrics (PIE mode only)
        info             : dict         file metadata
        intensity_traces : dict         per-channel intensity traces
        out_dir          : str          the folder all output was written to
    """
    def _prog(pct):
        if progress_callback is not None:
            progress_callback(float(pct))

    _prog(0.0)

    tttr_data = load_ptu(filepath)
    info      = get_ptu_info(tttr_data)

    stem = os.path.splitext(os.path.basename(filepath))[0]

    # CHANGED: output folder is always <input_dir>/<stem>/ unless
    # explicitly overridden.
    if out_dir_override:
        out_base = out_dir_override
    else:
        out_base = os.path.join(os.path.dirname(filepath), stem)
    os.makedirs(out_base, exist_ok=True)

    fixer = FCS_Fixer(
        photon_data              = tttr_data,
        tau_min                  = tau_min_s,
        tau_max                  = tau_max_s,
        sampling                 = n_bins,
        cross_corr_symm          = False,
        correlation_method       = "wahl",
        subtract_afterpulsing    = use_afterpulsing,
        afterpulsing_params_path = ap_params_path if use_afterpulsing else "",
    )
    fixer.update_params()

    _prog(5.0)
    

    # ── pre-flight check of which routing channels actually exist ──────────
    available_channels = set(int(c) for c in np.unique(tttr_data.routing_channels))
    pipeline_warnings = []

    def _channel_exists(ch: int) -> bool:
        return ch in available_channels
    pie_actually_used = False
    if use_pie:
        if not (_channel_exists(donor_channel) and _channel_exists(acceptor_channel)):
            pipeline_warnings.append(
                f"PIE mode requested but donor channel {donor_channel} and/or "
                f"acceptor channel {acceptor_channel} are not present in this "
                f"file (available: {sorted(available_channels)}). Falling back "
                f"to plain autocorrelation of whichever channel IS available."
            )
            fallback_channel = (
                donor_channel if _channel_exists(donor_channel)
                else (acceptor_channel if _channel_exists(acceptor_channel)
                      else sorted(available_channels)[0])
            )
            _cs = FCS_Fixer.build_channels_spec(fallback_channel)
            all_pairs = [(_cs, _cs, "ACF", True, False)]
        else:
            pie_actually_used = True
            pie_chs = build_pie_channels(
                donor_channel, acceptor_channel, prompt_gate, delay_gate,
            )
            # NEW: build the full candidate list, then filter below by
            # compute_dd/compute_aa/compute_da -- DD = donor ACF,
            # AA = acceptor ACF, DA = donor-acceptor CCF
            all_pairs = []
            if compute_dd:
                all_pairs.append((
                    pie_chs["donor_prompt"], pie_chs["donor_prompt"],
                    "donor_ACF", True, False,
                ))
            if compute_aa:
                all_pairs.append((
                    pie_chs["acceptor_delay"], pie_chs["acceptor_delay"],
                    "acceptor_ACF", True, False,
                ))
            if compute_da:
                all_pairs.append((
                    pie_chs["donor_prompt"], pie_chs["acceptor_prompt"],
                    "PIE_CCF", False, symmetric_cc,
                ))

            if not all_pairs:
                pipeline_warnings.append(
                    "No correlation pairs selected (DD/AA/DA all unchecked) "
                    "in PIE mode. Defaulting to donor ACF only."
                )
                all_pairs = [(
                    pie_chs["donor_prompt"], pie_chs["donor_prompt"],
                    "donor_ACF", True, False,
                )]
    else:
        if cs1 is None:
            cs1 = FCS_Fixer.build_channels_spec(channel)
        if cs2 is None:
            cs2 = cs1

        cs1_norm = FCS_Fixer.check_channels_spec(cs1)
        cs2_norm = FCS_Fixer.check_channels_spec(cs2)
        cs1_channels = set(cs1_norm[0])
        cs2_channels = set(cs2_norm[0])

        if not cs1_channels.issubset(available_channels):
            pipeline_warnings.append(
                f"Requested channel(s) {sorted(cs1_channels)} not present "
                f"in this file (available: {sorted(available_channels)}). "
                f"Falling back to channel {sorted(available_channels)[0]}."
            )
            cs1 = FCS_Fixer.build_channels_spec(sorted(available_channels)[0])
        if not cs2_channels.issubset(available_channels):
            if cs1_channels != cs2_channels:
                pipeline_warnings.append(
                    f"Requested channel(s) {sorted(cs2_channels)} not present "
                    f"in this file (available: {sorted(available_channels)}). "
                    f"Falling back to autocorrelation (cs2 = cs1)."
                )
            cs2 = cs1

        same_channel_only = (cs1 == cs2)

        # NEW: in non-PIE two-channel mode, DD = ch1 ACF, AA = ch2 ACF,
        # DA = ch1-ch2 CCF. If cs1 == cs2 (i.e. the user only ever
        # specified one channel to begin with), there is only ever one
        # meaningful pair -- the plain autocorrelation -- and the
        # DD/AA/DA checkboxes are not applicable; we always compute it
        # regardless of checkbox state so single-channel workflows keep
        # working exactly as before.
        all_pairs = []
        if same_channel_only:
            all_pairs.append((cs1, cs2, "ACF", True, False))
        else:
            if compute_dd:
                all_pairs.append((cs1, cs1, "DD_ACF", True, False))
            if compute_aa:
                all_pairs.append((cs2, cs2, "AA_ACF", True, False))
            if compute_da:
                all_pairs.append((cs1, cs2, "DA_CCF", False, symmetric_cc))

            if not all_pairs:
                pipeline_warnings.append(
                    "No correlation pairs selected (DD/AA/DA all unchecked) "
                    "for a two-channel measurement. Defaulting to channel-1 "
                    "autocorrelation only."
                )
                all_pairs = [(cs1, cs1, "DD_ACF", True, False)]

    pairs = all_pairs

    # ── derive the FULL (ungated) channels involved, used both for burst
    # removal below and for the intensity-trace display later. Moved up
    # here (previously computed only at the very end of the function)
    # so burst removal can run before the per-pair correlation loop. ──────
    if use_pie:
        trace_channels = [("donor", donor_channel), ("acceptor", acceptor_channel)]
    else:
        cs1_norm = FCS_Fixer.check_channels_spec(cs1)
        cs2_norm = FCS_Fixer.check_channels_spec(cs2)
        chans_involved = sorted(set(cs1_norm[0]) | set(cs2_norm[0]))
        trace_channels = [(f"ch{ch}", ch) for ch in chans_involved]

    # ── bleach/drift correction -- ONLY for channels actually used by the
    # selected pairs (derived from `pairs` itself, so DD/AA/DA checkbox
    # selections are respected and unused/empty channels are never
    # needlessly -- and riskily -- bleach-corrected) ────────────────────────
    undrift_degrees = {}
    if correct_bleaching:
        channels_to_correct = []
        for p_cs1, p_cs2, _label, _is_acf, _sym in pairs:
            channels_to_correct.append(p_cs1)
            channels_to_correct.append(p_cs2)

        seen = set()
        for cs in channels_to_correct:
            cs_key = FCS_Fixer.check_channels_spec(cs)
            if cs_key in seen:
                continue
            seen.add(cs_key)
            degree = _apply_undrift_for_channel(fixer, cs)
            undrift_degrees[cs_key] = degree

    # ── burst removal -- operates on FULL, ungated channels (trace_channels)
    # rather than on individually-gated pairs, so that the resulting
    # per-photon weights (self._weights_burst_removal) apply consistently
    # regardless of any micro-time gating a given correlation pair may
    # additionally use (select_photons() intersects both masks). ──────────
    burst_removal_info = {}
    if use_burst_removal:
        for name, ch in trace_channels:
            cs_full = FCS_Fixer.build_channels_spec(ch)
            info = _apply_burst_removal_for_channel(
                fixer, cs_full, threshold_alpha=burst_threshold_alpha,
            )
            burst_removal_info[name] = info

    n_pairs  = len(pairs)
    results  = []
    fret_res = None

    for i_pair, (p_cs1, p_cs2, label, is_acf, sym) in enumerate(pairs):

        _prog(5.0 + 80.0 * i_pair / n_pairs)

        fixer.cross_corr_symm = sym

        tcspc_csv   = None
        tcspc_csv_2 = None

        if use_flcs_bg:
            flcs1 = _apply_flcs_for_channel(fixer, p_cs1, use_drift_correction=correct_bleaching, use_burst_removal=use_burst_removal,)
            # CHANGED: uniform naming, no stem prefix (folder already
            # named after stem)
            tcspc_csv = os.path.join(out_base, f"{label}_tcspc.csv")
            pd.DataFrame({
                "time_ns":             flcs1["tcspc_x"] * fixer._micro_time_resolution,
                "counts":              flcs1["tcspc_y"],
                "pattern_signal":      flcs1["pattern_signal"],
                "pattern_background":  flcs1["pattern_background"],
                "filter_signal":       flcs1["filter_signal"],
                "filter_background":   flcs1["filter_background"],
            }).to_csv(tcspc_csv, index=False)

            if save_flcs_diagnostic:
                try:
                    fig1 = plot_flcs_diagnostic(
                        flcs1["patterns_norm_full"], flcs1["flcs_weights_full"],
                        title=label,
                    )
                    fig1_path = os.path.join(
                        out_base, f"{label}_flcs_diagnostic.png"
                    )
                    fig1.savefig(fig1_path, dpi=150)
                    import matplotlib.pyplot as plt
                    plt.close(fig1)
                except Exception:
                    pass

            if not _same_channels(p_cs1, p_cs2):
                flcs2 = _apply_flcs_for_channel(fixer, p_cs2, use_drift_correction=correct_bleaching, use_burst_removal=use_burst_removal,)
                tcspc_csv_2 = os.path.join(out_base, f"{label}_tcspc_cs2.csv")
                pd.DataFrame({
                    "time_ns":             flcs2["tcspc_x"] * fixer._micro_time_resolution,
                    "counts":              flcs2["tcspc_y"],
                    "pattern_signal":      flcs2["pattern_signal"],
                    "pattern_background":  flcs2["pattern_background"],
                    "filter_signal":       flcs2["filter_signal"],
                    "filter_background":   flcs2["filter_background"],
                }).to_csv(tcspc_csv_2, index=False)

                if save_flcs_diagnostic:
                    try:
                        fig2 = plot_flcs_diagnostic(
                            flcs2["patterns_norm_full"], flcs2["flcs_weights_full"],
                            title=f"{label} (cs2)",
                        )
                        fig2_path = os.path.join(
                            out_base, f"{label}_flcs_diagnostic_cs2.png"
                        )
                        fig2.savefig(fig2_path, dpi=150)
                        import matplotlib.pyplot as plt
                        plt.close(fig2)
                    except Exception:
                        pass

        lag_ns, cc, sd_cc, acr1_ghz, acr2_ghz = fixer.get_correlation_uncertainty(
            p_cs1, p_cs2,
            default_uncertainty_method="Wohland",
            minimum_window_length=([] if wohland_window_s is None else wohland_window_s),
            n_bootstrap_reps=n_bootstrap,
            use_flcs_bg_corr=use_flcs_bg, use_drift_correction=correct_bleaching, use_burst_removal=use_burst_removal,
        )

        lag_s = np.asarray(lag_ns, dtype=float) * 1e-9
        acr1_hz = float(acr1_ghz) * 1e9
        acr2_hz = float(acr2_ghz) * 1e9

        ap_used = bool(use_afterpulsing) and _same_channels(p_cs1, p_cs2)

        # CHANGED: uniform, descriptive correction tags and no stem prefix
        tags = ""
        if use_burst_removal:
            tags+="_br"
        if correct_bleaching:
            tags+="_bl"
        if ap_used:
            tags += "_apcorr"
        if use_flcs_bg:
            tags += "_flcsbg"
        csv_path = os.path.join(out_base, f"{label}{tags}.csv")

        acr_col      = np.zeros(len(lag_s))
        acr_col[0]   = acr1_hz

        pd.DataFrame({
            "tau_s":   lag_s,
            "G":       cc,
            "acr_Hz":  acr_col,
            "sigma_G": sd_cc,
        }).to_csv(csv_path, index=False, header=False)

        results.append({
            "label":         label,
            "csv_path":      csv_path,
            "lag_s":         lag_s,
            "G":             np.asarray(cc, dtype=float),
            "sigma_G":       np.asarray(sd_cc, dtype=float),
            "acr1_Hz":       acr1_hz,
            "acr2_Hz":       acr2_hz,
            "is_acf":        is_acf,
            "flcs_used":     use_flcs_bg,
            "ap_used":       ap_used,
            "bleach_corrected":   correct_bleaching,
            "burst_removed":      use_burst_removal,
            "tcspc_csv":     tcspc_csv,
            "tcspc_csv_cs2": tcspc_csv_2,
        })

    _prog(90.0)
    fret_res = None
    if pie_actually_used:
        fret_res = compute_pie_fret(
            fixer, pie_chs, gamma, crosstalk, direct_excitation
        )
        # CHANGED: uniform naming, no stem prefix
        fret_csv = os.path.join(out_base, "FRET_summary.csv")
        pd.DataFrame([fret_res]).to_csv(fret_csv, index=False)
        fret_res["csv_path"] = fret_csv
    elif use_pie:
        pipeline_warnings.append(
            "FRET metrics were not computed because PIE mode could not "
            "run (see channel-availability warning above) -- FRET "
            "requires genuine donor and acceptor channels."
        )

    

    intensity_traces = {}
    for name, ch in trace_channels:
        cs_all   = FCS_Fixer.build_channels_spec(ch)
        t_s, cps = get_intensity_trace(fixer, cs_all)
        entry = {"t_s": t_s, "cps": cps, "channel": ch}

        # NEW: also compute the bleach-corrected trace for the same
        # channel/binning, using the drift-correction weights already
        # written into fixer._weights_undrift by the bleach-correction
        # step above (this block runs after that step, so the weights
        # are already available). If bleach correction was never
        # applied for this particular channel (e.g. it wasn't part of
        # the selected DD/AA/DA pairs), the weights default to 1.0 and
        # this simply reproduces the raw trace -- harmless, just not
        # informative, so we still store it for consistency.
        if correct_bleaching:
            t_s_corr, cps_corr = get_intensity_trace(
                fixer, cs_all, use_drift_correction=True
            )
            entry["t_s_corrected"] = t_s_corr
            entry["cps_corrected"] = cps_corr
        if use_burst_removal and name in burst_removal_info:
            entry["burst_intervals_s"]     = burst_removal_info[name]["burst_intervals_s"]
            entry["burst_threshold_counts"] = burst_removal_info[name]["threshold_counts"]
            entry["n_burst_bins"]          = burst_removal_info[name]["n_burst_bins"]
            entry["n_total_bins"]          = burst_removal_info[name]["n_total_bins"]
        intensity_traces[name] = entry

    _prog(100.0)

    return {
        "results":          results,
        "fret":             fret_res,
        "info":             info,
        "intensity_traces": intensity_traces,
        "out_dir":          out_base,   # NEW: reported so the GUI/log can show it
        "warnings":         pipeline_warnings,
        "burst_removal_info": burst_removal_info,
    }
# ═════════════════════════════════════════════════════════════════════════
# File-type dispatch — routes .ptu files to run_fcs_export() and .raw
# files to the (currently stubbed) run_fcs_export_raw().
# ═════════════════════════════════════════════════════════════════════════

def run_fcs_export_dispatch(filepath: str, progress_callback=None, **kwargs) -> dict:
    """
    Dispatch a single file to the correct export pipeline based on its
    extension.

    .ptu -> run_fcs_export()      (full PTU/PIE/FLCS pipeline)
    .raw -> run_fcs_export_raw()  (Zeiss raw — currently NotImplementedError)

    This lets a single batch folder safely contain a mix of file types:
    each file is routed independently, and a failure on one .raw file
    (NotImplementedError) is caught per-file by the batch runner rather
    than aborting the whole batch.
    """
    if is_raw_file(filepath):
        return run_fcs_export_raw(filepath, progress_callback=progress_callback, **kwargs)
    return run_fcs_export(filepath, progress_callback=progress_callback, **kwargs)
    
# ═════════════════════════════════════════════════════════════════════════
# Batch helper — sequential or parallel
# ═════════════════════════════════════════════════════════════════════════

def _batch_worker_wrapper(args):
    """
    Picklable module-level wrapper for multiprocessing.Pool.imap().

    Parameters
    ----------
    args : tuple (filepath, kwargs_dict)

    Returns
    -------
    (filepath, result_dict_or_None, traceback_str_or_None)
    """
    filepath, kwargs = args
    try:
        res = run_fcs_export_dispatch(filepath, progress_callback=None, **kwargs)
        return (filepath, res, None)
    except Exception:
        import traceback as _tb
        return (filepath, None, _tb.format_exc())


def run_fcs_export_batch(
    filepaths:      list,
    progress_queue = None,
    cancel_event   = None,
    cpu_n:          int = 1,
    **kwargs,
) -> dict:
    """
    Run the FCS export pipeline on multiple files, sequentially or in
    parallel depending on cpu_n.

    Each file is routed through run_fcs_export_dispatch(), so a batch
    folder may safely contain a mix of .ptu and .raw files — .raw files
    will simply fail per-file with a clear NotImplementedError until
    that pipeline is implemented, without aborting the rest of the batch.

    Parameters
    ----------
    filepaths      : list[str]  paths to input files (.ptu and/or .raw)
    progress_queue : multiprocessing.Queue | None
        Receives ("progress", float), ("file_done", result_dict),
        ("file_error", str) messages.
    cancel_event   : multiprocessing.Event | None
    cpu_n          : int
        Number of parallel worker processes to use. 1 = sequential
        (identical behaviour to the original implementation, with
        per-file progress_callback forwarding). >1 = parallel via
        multiprocessing.Pool (per-file progress_callback is not
        available in parallel mode — only per-file completion is
        reported, matching the granularity already used elsewhere in
        theatRICS, e.g. sim_worker.py / diffmap_worker.py).
    **kwargs       : forwarded to run_fcs_export_dispatch() / run_fcs_export()

    Returns
    -------
    dict: n_total, n_ok, n_failed, failed, last_res
    """
    n_total  = len(filepaths)
    n_ok     = 0
    failed   = []
    last_res = None

    if progress_queue is not None:
        progress_queue.put(("progress", 0.0))

    if n_total == 0:
        return {"n_total": 0, "n_ok": 0, "n_failed": 0, "failed": [], "last_res": None}

    cpu_n = max(1, int(cpu_n))

    # ── sequential path (cpu_n == 1, or only one file) ──────────────────
    if cpu_n <= 1 or n_total <= 1:
        for i, fp in enumerate(filepaths):
            if cancel_event is not None and cancel_event.is_set():
                break

            try:
                def _cb(pct, _i=i, _n=n_total):
                    if progress_queue is not None:
                        progress_queue.put((
                            "progress",
                            100.0 * (_i + pct / 100.0) / _n,
                        ))

                res      = run_fcs_export_dispatch(fp, progress_callback=_cb, **kwargs)
                last_res = res
                n_ok    += 1

                if progress_queue is not None:
                    progress_queue.put(("file_done", res))

            except Exception:
                import traceback as _tb
                failed.append(fp)
                if progress_queue is not None:
                    progress_queue.put((
                        "file_error",
                        f"FAILED: {os.path.basename(fp)}\n{_tb.format_exc()}"
                    ))

            if progress_queue is not None:
                progress_queue.put(
                    ("progress", 100.0 * (i + 1) / max(1, n_total))
                )

        return {
            "n_total":  n_total,
            "n_ok":     n_ok,
            "n_failed": len(failed),
            "failed":   failed,
            "last_res": last_res,
        }

    # ── parallel path (cpu_n > 1) ───────────────────────────────────────
    import multiprocessing as mp

    cpu_n = min(cpu_n, n_total, mp.cpu_count())
    args_list = [(fp, kwargs) for fp in filepaths]
    completed = 0

    with mp.Pool(processes=cpu_n) as pool:
        for filepath, res, err in pool.imap(_batch_worker_wrapper, args_list, chunksize=1):

            if cancel_event is not None and cancel_event.is_set():
                pool.terminate()
                pool.join()
                break

            if err is None:
                last_res = res
                n_ok    += 1
                if progress_queue is not None:
                    progress_queue.put(("file_done", res))
            else:
                failed.append(filepath)
                if progress_queue is not None:
                    progress_queue.put((
                        "file_error",
                        f"FAILED: {os.path.basename(filepath)}\n{err}"
                    ))

            completed += 1
            if progress_queue is not None:
                progress_queue.put(
                    ("progress", 100.0 * completed / max(1, n_total))
                )

    return {
        "n_total":  n_total,
        "n_ok":     n_ok,
        "n_failed": len(failed),
        "failed":   failed,
        "last_res": last_res,
    }