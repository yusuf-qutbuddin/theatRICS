from __future__ import annotations
import traceback

from theatrics.fcsfit import batch as fcs_batch


def fcsfit_process_main(params, out_queue, cancel_event):
    try:
        mode = params["mode"]  # "single" or "batch"
        out_queue.put(("progress", 0.0))

        if mode == "single":
            res = fcs_batch.run_single_csv(cancel_event=cancel_event, **params["kwargs"])
            if cancel_event.is_set():
                out_queue.put(("cancelled", None))
                return
            out_queue.put(("done", res))
            return

        if mode == "batch":
            folder = params["folder"]

            batch_out = fcs_batch.run_batch_folder(
                folder=folder,
                progress_queue=out_queue,
                cancel_event=cancel_event,
                **params["kwargs"],
            )
            if cancel_event.is_set():
                out_queue.put(("cancelled", None))
                return

            # Send a final "done" with summary + last result
            out_queue.put(("done", batch_out))
            return

        raise ValueError(f"Unknown mode: {mode}")

    except Exception:
        out_queue.put(("error", traceback.format_exc()))