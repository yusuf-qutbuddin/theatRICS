from __future__ import annotations
import traceback
import numpy as np
from pathlib import Path

from theatrics.vesicle import detection


def vesicle_process_main(params, out_queue, cancel_event):
    try:
        out_queue.put(("progress", 0.0))

        detection._frame0_centroids.clear()

        phase = params["phase"]

        if phase == "straighten":
            _run_straighten(params, out_queue, cancel_event)
            return

        # existing detect/crop phases
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
            filter_circularity=params.get("filter_circularity", 0.0),
            filter_eccentricity=params.get("filter_eccentricity", 1.0),
            filter_solidity=params.get("filter_solidity", 0.0),
            preprocess_transmitted=params.get("preprocess_transmitted", False),
            fit_circles=params.get("fit_circles", False),
            min_area_um2=params.get("min_area_um2", 1.0),
            min_radius_um=params.get("min_radius_um", 1.0),
            max_radius_um=params.get("max_radius_um", 20.0),
            radius_step_um=params.get("radius_step_um", 0.5),
            canny_sigma=params.get("canny_sigma", 2.0),
            hough_min_distance_um=params.get("hough_min_distance_um", 5.0),
            hough_threshold_fraction=params.get("hough_threshold_fraction", 0.3),
            fallback_pixel_size_um=params.get("fallback_pixel_size_um", None),
            weight_search_range = params.get("weight_search_range", 2.0),
            threshold_method=params.get("threshold_method", "huang"),
            selected_labels=params.get("selected_labels", None),
            progress_queue=out_queue,
            cancel_event=cancel_event,
            debug=params.get("debug", False),
        )

        if cancel_event.is_set():
            out_queue.put(("cancelled", None))
            return

        out_queue.put(("done", result))

    except Exception:
        out_queue.put(("error", traceback.format_exc()))


def _run_straighten(params, out_queue, cancel_event):
    import tifffile
    import pandas as pd

    vesicles = params["vesicles"]
    pixel_size_um = params.get("pixel_size_um", 1.0)
    thickness_um = params.get("thickness_um", 2.0)
    thickness_px = max(1, int(round(thickness_um / pixel_size_um)))
    straighten_channel = params.get("straighten_channel", 0)

    czi_path = params["czi_path"]
    czi_stem = Path(czi_path).stem
    out_dir = Path(czi_path).parent / f"{czi_stem}_straightened"
    out_dir.mkdir(exist_ok=True)

    total = len(vesicles)
    all_results = []

    for vi, v in enumerate(vesicles):
        if cancel_event.is_set():
            out_queue.put(("cancelled", None))
            return

        res = detection.straighten_vesicle_timeseries(
            czi_path=czi_path,
            channel=straighten_channel,
            center_y=v["centroid_y"],
            center_x=v["centroid_x"],
            radius=v["radius"],
            thickness_px=thickness_px,
            frame_start=params.get("frame_start", 0),
            frame_end=params.get("frame_end", None),
            frame_step=params.get("frame_step", 1),
            cancel_event=cancel_event,
        )

        if res.get("mode") == "cancelled":
            out_queue.put(("cancelled", None))
            return

        lbl = v["label"]
        res["vesicle_label"] = lbl
        res["pixel_size_um"] = pixel_size_um
        res["thickness_um"] = thickness_um

        # save straightened strip TIFF
        tiff_path = str(out_dir / f"vesicle_{lbl}_straightened.tif")
        tifffile.imwrite(tiff_path, res["strips"].astype(np.float32), photometric="minisblack")
        res["tiff_path"] = tiff_path

        # save intensity profile CSV (angle vs frame)
        csv_path = str(out_dir / f"vesicle_{lbl}_intensity_profile.csv")
        angle_cols = [f"{a:.1f}" for a in res["angles_deg"]]
        df = pd.DataFrame(res["intensity_profile"], columns=angle_cols)
        df.insert(0, "frame", range(res["n_frames"]))
        df.to_csv(csv_path, index=False)
        res["profile_csv"] = csv_path

        # save total intensity CSV
        total_csv = str(out_dir / f"vesicle_{lbl}_total_intensity.csv")
        pd.DataFrame({
            "frame": range(res["n_frames"]),
            "total_intensity": res["total_intensity"],
        }).to_csv(total_csv, index=False)
        res["total_csv"] = total_csv

        # convert strips to list for queue serialization
        res["strips"] = res["strips"].tolist()
        res["intensity_profile"] = res["intensity_profile"].tolist()
        res["total_intensity"] = res["total_intensity"].tolist()
        res["angles_deg"] = res["angles_deg"].tolist()

        all_results.append(res)

        out_queue.put(("progress", 100.0 * (vi + 1) / total))

    out_queue.put(("done", {
        "mode": "straighten",
        "output_dir": str(out_dir),
        "results": all_results,
        "pixel_size_um": pixel_size_um,
        "thickness_um": thickness_um,
        "czi_path": czi_path,
    }))