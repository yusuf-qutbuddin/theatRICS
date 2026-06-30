from __future__ import annotations

import traceback
from theatrics.ics import analysis as ics_analysis


def ics_process_main(params: dict, progress_queue, cancel_event):
    """
    Worker entry point.  Runs in a separate process.
    Sends messages to progress_queue:
        ("progress", float 0-100)
        ("file_done", result_dict)
        ("block_done", info_dict)
        ("error_file", path_str)
        ("done", summary_dict)
        ("error", traceback_str)
        ("cancelled", None)
    """
    try:
        mode   = params.get("mode", "single")
        config = params.get("config", {})

        if cancel_event.is_set():
            progress_queue.put(("cancelled", None))
            return

        if mode == "single":
            path = params["tiff_path"]
            res  = ics_analysis.analyse_tiff(
                path, config,
                progress_queue=progress_queue,
                cancel_event=cancel_event,
            )
            if cancel_event.is_set():
                progress_queue.put(("cancelled", None))
                return
            progress_queue.put(("progress", 100.0))
            progress_queue.put(("done", res))

        elif mode == "batch":
            folder = params["folder"]
            res    = ics_analysis.run_ics_batch(
                folder, config,
                progress_queue=progress_queue,
                cancel_event=cancel_event,
            )
            if cancel_event.is_set():
                progress_queue.put(("cancelled", None))
                return
            progress_queue.put(("progress", 100.0))
            progress_queue.put(("done", res))

        else:
            raise ValueError(f"Unknown ICS mode: {mode!r}")

    except Exception:
        progress_queue.put(("error", traceback.format_exc()))