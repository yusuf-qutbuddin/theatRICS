"""
theatrics/workers/ptu_correlate_worker.py

Worker entry point for the FCS Export tab (PTU + Zeiss .raw, PIE-FCCS).
"""
from __future__ import annotations
import traceback
from theatrics.fcsfit.ptu_correlate import (
    run_fcs_export_dispatch,
    run_fcs_export_batch,
)


def ptu_correlate_worker_main(
    params:      dict,
    out_queue,
    cancel_event,
) -> None:
    """
    Worker entry point.

    params keys (all optional unless noted)
    ───────────────────────────────────────
    mode              : "single" | "batch"   (required)
    filepath          : str                  (single mode)
    filepaths         : list[str]            (batch mode)
    cpu_n             : int                  (batch mode; parallel workers)

    # channels_spec objects (built by build_channels_spec in the GUI)
    cs1               : tuple | None
    cs2               : tuple | None
    channel           : int               fallback when cs1/cs2 are None

    # PIE
    use_pie           : bool
    donor_channel     : int
    acceptor_channel  : int
    prompt_gate       : [start, stop]     stored as list, converted to tuple
    delay_gate        : [start, stop]
    symmetric_cc      : bool
    gamma             : float
    crosstalk         : float
    direct_excitation : float

    # correlation
    tau_min_s         : float
    tau_max_s         : float
    n_bins            : int

    # corrections
    use_afterpulsing  : bool
    ap_params_path    : str
    use_flcs_bg       : bool

    # SD
    wohland_window_s  : float | None
    n_bootstrap       : int
    """
    try:
        out_queue.put(("progress", 0.0))

        if cancel_event.is_set():
            out_queue.put(("cancelled", None))
            return

        mode = params.get("mode", "single")

        # ── convert list gates to tuples (JSON serialises tuples as lists) ──
        pg = params.get("prompt_gate", [0.0, 0.5])
        dg = params.get("delay_gate",  [0.5, 1.0])
        prompt_gate = (float(pg[0]), float(pg[1]))
        delay_gate  = (float(dg[0]), float(dg[1]))

        export_kwargs = dict(
            cs1               = params.get("cs1",               None),
            cs2               = params.get("cs2",               None),
            channel           = params.get("channel",           0),
            use_pie           = params.get("use_pie",           False),
            donor_channel     = params.get("donor_channel",     0),
            acceptor_channel  = params.get("acceptor_channel",  1),
            prompt_gate       = prompt_gate,
            delay_gate        = delay_gate,
            symmetric_cc      = params.get("symmetric_cc",      True),
            gamma             = params.get("gamma",             1.0),
            crosstalk         = params.get("crosstalk",          0.0),
            direct_excitation = params.get("direct_excitation",  0.0),
            tau_min_s         = params.get("tau_min_s",          1e-6),
            tau_max_s         = params.get("tau_max_s",          1.0),
            n_bins            = params.get("n_bins",             9),
            use_afterpulsing  = params.get("use_afterpulsing",   False),
            ap_params_path    = params.get("ap_params_path",     ""),
            use_flcs_bg       = params.get("use_flcs_bg",        False),
            wohland_window_s  = params.get("wohland_window_s",   None),
            n_bootstrap       = params.get("n_bootstrap",        20),
            # ──  Zeiss .raw-specific parameters ──────────────────────
            channels_pairs    = params.get("channels_pairs",     None),
            n_segments        = params.get("n_segments",         6),
            offset_s          = params.get("offset_s",           0.0),
            # ──  bleaching and burst-specific parameters ──────────────────────
            correct_bleaching = params.get("correct_bleaching",  False),
            use_burst_removal      = params.get("use_burst_removal",      False),   
            burst_threshold_alpha  = params.get("burst_threshold_alpha",  0.02),
            # ──  which auto/cross-correlation pairs to compute ────────
            compute_dd        = params.get("compute_dd",         True),
            compute_aa        = params.get("compute_aa",         True),
            compute_da        = params.get("compute_da",         True),
        )

        if mode == "single":
            filepath = params["filepath"]

            def _progress(pct: float):
                if cancel_event.is_set():
                    raise RuntimeError("cancelled")
                out_queue.put(("progress", float(pct)))

            # run_fcs_export_dispatch routes .ptu -> full pipeline,
            # .raw -> stub (currently NotImplementedError until the
            # Zeiss .raw correlation algorithm is ported)
            res = run_fcs_export_dispatch(
                filepath,
                progress_callback=_progress,
                **export_kwargs,
            )

            if cancel_event.is_set():
                out_queue.put(("cancelled", None))
                return

            out_queue.put(("done", res))

        elif mode == "batch":
            filepaths = params["filepaths"]
            cpu_n     = int(params.get("cpu_n", 1))

            summary = run_fcs_export_batch(
                filepaths,
                progress_queue=out_queue,
                cancel_event=cancel_event,
                cpu_n=cpu_n,
                **export_kwargs,
            )

            if cancel_event.is_set():
                out_queue.put(("cancelled", None))
                return

            out_queue.put(("done", summary))

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    except Exception:
        out_queue.put(("error", traceback.format_exc()))