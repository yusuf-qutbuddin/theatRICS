"""
theatrics/workers/ptu_image_worker.py

Worker entry point for the PTU Image / FLIM tab.
"""
from __future__ import annotations
import traceback
from theatrics.imaging.ptu_image import run_ptu_image_export, get_channel_info


def ptu_image_worker_main(params: dict, out_queue, cancel_event) -> None:
    """
    params["task"] controls the operation:
        "inspect"  : load the file, return channel info (fast, no images)
        "export"   : full image / FLIM extraction
    """
    try:
        task = params.get("task", "export")
        out_queue.put(("progress", 0.0))

        if cancel_event.is_set():
            out_queue.put(("cancelled", None))
            return

        if task == "inspect":
            info = get_channel_info(params["ptu_path"])
            out_queue.put(("inspected", info))
            return

        # task == "export"
        def _progress(pct: float):
            if cancel_event.is_set():
                raise RuntimeError("cancelled")
            out_queue.put(("progress", float(pct)))

        result = run_ptu_image_export(params, progress_callback=_progress)

        if cancel_event.is_set():
            out_queue.put(("cancelled", None))
            return

        out_queue.put(("done", result))

    except RuntimeError as e:
        if "cancelled" in str(e):
            out_queue.put(("cancelled", None))
        else:
            out_queue.put(("error", traceback.format_exc()))
    except Exception:
        out_queue.put(("error", traceback.format_exc()))