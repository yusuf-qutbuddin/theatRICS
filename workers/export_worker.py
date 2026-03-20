import os
import numpy as np
import tifffile
import traceback
from modules import export_rics

from pylibCZIrw import czi as pyczi

def export_rics_process_main(params, out_q):
    """
    params keys:
      input_file, channel, crop_factor, window_size, correct_drift
    """
    try:
        out_q.put(("progress", 0.0))
        input_file = params["input_file"]
        channel = int(params["channel"])
        crop_factor = float(params["crop_factor"])
        window_size = int(params["window_size"])
        correct_drift = bool(params["correct_drift"])

        ext = os.path.splitext(input_file)[1].lower()

        out_q.put(("progress", 5.0))

        if ext == ".czi":
            with pyczi.open_czi(input_file) as czidoc:
                n_frames = czidoc.total_bounding_box["T"][1]
            out_q.put(("progress", 10.0))

            RICS_map, sd_map, all_frames, corrected_stack = export_rics.process_all_frames_czi(
                input_file, n_frames, channel, window_size, crop_factor, correct_drift
            )

        elif ext in [".tif", ".tiff"]:
            stack = tifffile.imread(input_file)
            cropped = export_rics.crop_center(stack, crop_factor=crop_factor)
            if correct_drift:
                cropped = export_rics.drift_correct(cropped)
            n_frames = cropped.shape[0]
            out_q.put(("progress", 10.0))
            RICS_map, sd_map, all_frames, corrected_stack = export_rics.process_all_frames_tiff(
                cropped, n_frames, channel, window_size
            )
        else:
            raise ValueError(f"Unsupported file: {ext}")

        out_q.put(("progress", 90.0))

        rics_output = os.path.splitext(input_file)[0] + "_RICScorr.tif"
        sd_output   = os.path.splitext(input_file)[0] + "_RICSunc.tif"
        tifffile.imwrite(rics_output, RICS_map, photometric="minisblack")
        tifffile.imwrite(sd_output, sd_map, photometric="minisblack")
        tiff_output = os.path.splitext(input_file)[0] + "_TIFF.tif"
        corrected_tiff_output = os.path.splitext(input_file)[0] + "_corrected_TIFF.tif"                                    
        tifffile.imwrite(corrected_tiff_output, corrected_stack[0],photometric="minisblack")
        tifffile.imwrite(tiff_output, all_frames[0],photometric="minisblack")
        out_q.put(("progress", 100.0))
        out_q.put(("done", {
            "rics_output": rics_output,
            "sd_output": sd_output,
            "corrected_tiff_output": corrected_tiff_output,
            "tiff_output": tiff_output
            # optionally also save corrected stack paths if you want
        }))

    except Exception:
        out_q.put(("error", traceback.format_exc()))