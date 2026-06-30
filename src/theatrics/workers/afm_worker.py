from __future__ import annotations

import traceback
import numpy as np


def afm_worker_main(params: dict, result_queue, cancel_event):
    """
    Worker entry point — runs in a separate process.

    Messages sent to result_queue
    ─────────────────────────────
    ("loaded",    info_dict)         file loaded successfully
    ("profile",   result_dict)       one profile computed
    ("error",     traceback_str)
    ("cancelled", None)
    """
    try:
        from theatrics.afm.analysis import (
            load_jpk_qi, get_file_info, process_profile
        )

        task = params.get("task")

        # ── load file ──────────────────────────────────────────
        if task == "load":
            filepath = params["filepath"]
            channel  = params.get("channel", "height_trace")

            height_nm, pixel_size_nm = load_jpk_qi(filepath, channel)
            info = get_file_info(height_nm, pixel_size_nm)

            result_queue.put((
                "loaded",
                {
                    "height_nm":     height_nm,      # ndarray
                    "pixel_size_nm": pixel_size_nm,
                    "info":          info,
                    "filepath":      filepath,
                }
            ))

        # ── compute profile ────────────────────────────────────
        elif task == "profile":
            if cancel_event.is_set():
                result_queue.put(("cancelled", None))
                return

            # height_data is passed as a numpy array
            res = process_profile(
                height_data   = np.asarray(params["height_nm"]),
                pixel_size_nm = params["pixel_size_nm"],
                start_px      = params["start_px"],
                end_px        = params["end_px"],
                h_start_nm    = params["h_start_nm"],
                h_end_nm      = params["h_end_nm"],
                n_fit_points  = params.get("n_fit_points",  15),
                smooth_window = params.get("smooth_window", 15),
                smooth_poly   = params.get("smooth_poly",    3),
                n_points      = params.get("n_points",      300),
                threshold_fraction = params.get(
                    "threshold_fraction", 0.05
                ),
            )
            result_queue.put(("profile", res))

        # ── refit angles only (slider moved) ──────────────────
        elif task == "refit_angles":
            from theatrics.afm.analysis import (
                find_contact_and_angles, measure_profile
            )
            distances = np.asarray(params["distances_nm"])
            h_adj     = np.asarray(params["h_adj"])
            nfp       = params["n_fit_points"]
            threshold_fraction = params.get("threshold_fraction", 0.05)

            angles = find_contact_and_angles(
                distances, h_adj,
                n_fit_points=nfp,
                threshold_fraction=threshold_fraction,
            )
            measurements = measure_profile(distances, h_adj, angles)
            result_queue.put((
                "refit_done",
                {
                    "angles":       angles,
                    "measurements": measurements,
                    "n_fit_points": nfp,
                    "profile_idx":  params.get("profile_idx", -1),
                }
            ))

        else:
            raise ValueError(f"Unknown AFM task: {task!r}")

    except Exception:
        result_queue.put(("error", traceback.format_exc()))