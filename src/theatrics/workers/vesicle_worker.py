from __future__ import annotations
import traceback
from theatrics.vesicle import detection


def vesicle_process_main(params, out_queue, cancel_event):
    """
    Worker for vesicle detection and cropping.

    params keys:
      - phase: "detect" or "crop"
      - czi_path, channel, frame_start, frame_end, frame_step
      - crop_margin, use_cellpose, model_type, diameter, min_area
      - selected_labels (only for crop phase)
    """
    try:
        out_queue.put(("progress", 0.0))

        # reset tracking state for fresh run
        detection._frame0_centroids.clear()

        phase = params["phase"]

        result = detection.process_vesicle_detection(
            czi_path=params["czi_path"],
            channel=params.get("channel", 0),
            frame_start=params.get("frame_start", 0),
            frame_end=params.get("frame_end", None),
            frame_step=params.get("frame_step", 1),
            crop_margin_um=params.get("crop_margin_um", 5.0),
            method=params.get("method", "hough"),
            use_cellpose=params.get("use_cellpose", True),
            model_type=params.get("model_type", "cyto3"),
            diameter=params.get("diameter", None),
            cellpose_gpu=params.get("cellpose_gpu", False),
            cellpose_invert=params.get("cellpose_invert", False),
            min_area_um2=params.get("min_area_um2", 1.0),
            min_radius_um=params.get("min_radius_um", 1.0),
            max_radius_um=params.get("max_radius_um", 20.0),
            radius_step_um=params.get("radius_step_um", 0.5),
            canny_sigma=params.get("canny_sigma", 2.0),
            hough_min_distance_um=params.get("hough_min_distance_um", 5.0),
            hough_threshold_fraction=params.get("hough_threshold_fraction", 0.3),
            fallback_pixel_size_um=params.get("fallback_pixel_size_um", None),
            selected_labels=params.get("selected_labels", None),
            progress_queue=out_queue,
            cancel_event=cancel_event,
        )

        if cancel_event.is_set():
            out_queue.put(("cancelled", None))
            return

        out_queue.put(("done", result))

    except Exception:
        out_queue.put(("error", traceback.format_exc()))