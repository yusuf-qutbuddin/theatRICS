from __future__ import annotations
import traceback

from theatrics.frap import analysis as frap_analysis


def frap_process_main(params, out_queue, cancel_event):
    try:
        mode = params["mode"]
        config = params.get("config", None)

        out_queue.put(("progress", 0.0))

        if mode == "single":
            res = frap_analysis.analyse_frap(params["czi_path"], config=config)
            if cancel_event.is_set():
                out_queue.put(("cancelled", None))
                return
            out_queue.put(("done", res))
            return

        if mode == "batch":
            res = frap_analysis.run_frap_batch(
                folder=params["folder"],
                pattern=params.get("pattern", "*FRAP*.czi"),
                config=config,
                progress_queue=out_queue,
                cancel_event=cancel_event,
            )
            if cancel_event.is_set():
                out_queue.put(("cancelled", None))
                return
            out_queue.put(("done", res))
            return

        raise ValueError(f"Unknown mode: {mode}")

    except Exception:
        out_queue.put(("error", traceback.format_exc()))