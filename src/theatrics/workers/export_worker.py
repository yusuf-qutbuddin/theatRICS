import os
import numpy as np
import tifffile
import traceback
from theatrics.modules import export_rics

from pylibCZIrw import czi as pyczi


def export_rics_process_main(params, out_q, cancel_event):
    """
    params keys:
      input_file, channel, crop_factor, window_size, correct_drift
    """
    try:
        out_q.put(("progress", 0.0))
        if cancel_event.is_set():
            out_q.put(("cancelled", None))
            return

        input_file   = params["input_file"]
        channel      = int(params["channel"])
        crop_factor  = float(params["crop_factor"])
        window_size  = int(params["window_size"])
        correct_drift = bool(params["correct_drift"])

        ext = os.path.splitext(input_file)[1].lower()

        out_q.put(("progress", 5.0))

        # ── CZI ──────────────────────────────────────────────────
        if ext == ".czi":
            with pyczi.open_czi(input_file) as czidoc:
                n_frames = czidoc.total_bounding_box["T"][1]
            out_q.put(("progress", 10.0))
            if cancel_event.is_set():
                out_q.put(("cancelled", None))
                return
            RICS_map, sd_map, all_frames, corrected_stack = (
                export_rics.process_all_frames_czi(
                    input_file, n_frames, channel,
                    window_size, crop_factor, correct_drift
                )
            )

        # ── TIFF ─────────────────────────────────────────────────
        elif ext in (".tif", ".tiff"):
            stack  = tifffile.imread(input_file)
            cropped = export_rics.crop_center(stack, crop_factor=crop_factor)
            if correct_drift:
                cropped = export_rics.drift_correct(cropped)
            n_frames = cropped.shape[0]
            out_q.put(("progress", 10.0))
            RICS_map, sd_map, all_frames, corrected_stack = (
                export_rics.process_all_frames_tiff(
                    cropped, n_frames, channel, window_size
                )
            )

        # ── PTU (PicoQuant Luminosa) ──────────────────────────────
        elif ext == ".ptu":
            # check tttrlib is available before doing any work
            if not export_rics.TTTRLIB_AVAILABLE:
                raise ImportError(
                    "tttrlib is not installed. "
                    "Install it with:  pip install tttrlib"
                )
            out_q.put(("progress", 10.0))
            if cancel_event.is_set():
                out_q.put(("cancelled", None))
                return
            RICS_map, sd_map, all_frames, corrected_stack = (
                export_rics.process_all_frames_ptu(
                    input_file,
                    channel=channel,
                    window_size=window_size,
                    crop_factor=crop_factor,
                    correct_drift=correct_drift,
                )
            )

        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if cancel_event.is_set():
            out_q.put(("cancelled", None))
            return

        out_q.put(("progress", 90.0))

        # ── save outputs ──────────────────────────────────────────
        root = os.path.splitext(input_file)[0]
        rics_output           = root + "_RICScorr.tif"
        sd_output             = root + "_RICSunc.tif"
        tiff_output           = root + "_TIFF.tif"
        corrected_tiff_output = root + "_corrected_TIFF.tif"

        tifffile.imwrite(rics_output,           RICS_map,            photometric="minisblack")
        tifffile.imwrite(sd_output,             sd_map,              photometric="minisblack")
        tifffile.imwrite(tiff_output,           all_frames[0],       photometric="minisblack")
        tifffile.imwrite(corrected_tiff_output, corrected_stack[0],  photometric="minisblack")

        if cancel_event.is_set():
            out_q.put(("cancelled", None))
            return

        out_q.put(("progress", 100.0))
        out_q.put(("done", {
            "rics_output":           rics_output,
            "sd_output":             sd_output,
            "corrected_tiff_output": corrected_tiff_output,
            "tiff_output":           tiff_output,
        }))

    except Exception:
        out_q.put(("error", traceback.format_exc()))