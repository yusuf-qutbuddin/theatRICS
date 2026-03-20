# workers/fit_worker.py
import os
import numpy as np
import pandas as pd
import tifffile
import traceback

from modules import rics_fit
from modules import inspect_metadata as im
from pylibCZIrw import czi as pyczi


def _derive_metadata_path(rics_tif_path: str) -> str:
    """
    Your convention: ..._RICScorr.tif -> corresponding .czi with same prefix before last "_...."
    Example:
      sample_01_RICScorr.tif -> sample_01.czi
    """
    root, _ = os.path.splitext(rics_tif_path)
    parts = os.path.basename(root).split("_")
    if len(parts) >= 2:
        base = "_".join(parts[:-1]) + ".czi"
    else:
        base = os.path.basename(root) + ".czi"
    return os.path.join(os.path.dirname(root), base)


def _crop_center_map(R, crop_fast, crop_slow):
    floor_fast = int(np.floor(R.shape[1] * (1 - crop_fast) * 0.5))
    ceil_fast  = int(np.floor(R.shape[1] * 0.5 * (1 + crop_fast)))
    floor_slow = int(np.floor(R.shape[0] * (1 - crop_slow) * 0.5))
    ceil_slow  = int(np.floor(R.shape[0] * 0.5 * (1 + crop_slow)))
    return R[floor_slow:ceil_slow, floor_fast:ceil_fast]


def fit_rics_one_file(params, out_q=None):
    """
    Returns (summary_dict, npz_path).

    params keys expected:
      rics_file, saving_path,
      crop_fast, crop_slow,
      diffusion_model,
      channel_to_use (for metadata),
      fit_pixel_size_nm, fit_pixel_dwell_us, fit_line_time_ms  (fallback)
      psf_size_xy_um, psf_aspect_ratio
      do_fit_1d (bool)
    """
    rics_file = params["rics_file"]
    saving_path = params["saving_path"]

    crop_fast = float(params["crop_fast"])
    crop_slow = float(params["crop_slow"])
    model_type = params["diffusion_model"]

    psf_size_xy_um = float(params["psf_size_xy_um"])
    psf_aspect_ratio = float(params["psf_aspect_ratio"])
    do_fit_1d = bool(params.get("do_fit_1d", False))

    # ---- load map ----
    R = tifffile.imread(rics_file).astype(np.float32)

    # ---- metadata (optional) ----
    metadata_path = _derive_metadata_path(rics_file)
    if os.path.isfile(metadata_path):
        channel_to_use = int(params.get("channel_to_use", 0))
        with pyczi.open_czi(metadata_path) as czidoc:
            Pixel_size_nm, Pixel_dwell_time_us, line_time_ms = im.get_metadata(czidoc, channel_to_use)
        pixel_size_um = float(Pixel_size_nm) * 1e-3
        pixel_time_s  = float(Pixel_dwell_time_us) * 1e-6
        line_time_s   = float(line_time_ms) * 1e-3
        used_metadata = True
    else:
        # fallback to GUI values
        pixel_size_um = float(params["fit_pixel_size_nm"]) * 1e-3
        pixel_time_s  = float(params["fit_pixel_dwell_us"]) * 1e-6
        line_time_s   = float(params["fit_line_time_ms"]) * 1e-3
        used_metadata = False

    # ---- crop + zero center ----
    R = _crop_center_map(R, crop_fast, crop_slow)
    cy, cx = R.shape[0] // 2, R.shape[1] // 2
    R[cy, cx] = 0.0

    # ---- fit ----
    fitter = rics_fit.RICS_fit(
        RICS_map=R,
        pixel_size_um=pixel_size_um,
        pixel_time_s=pixel_time_s,
        line_time_s=line_time_s,
        psf_size_xy_um=psf_size_xy_um,
        psf_aspect_ratio=psf_aspect_ratio
    )

    if model_type == "2Ddiff":
        fit_params, model, residual = fitter.run_2Ddiff_fit()
        D = float(fit_params["diff_coeff"].value)
        amp = float(fit_params["amplitude"].value)
        offset = float(fit_params["offset"].value)
        N = float(0.5 / amp)
        summary = dict(filepath=rics_file, model=model_type, N=N, D=D, offset=offset,
                       residual=float(np.mean(residual)))
    elif model_type == "3Ddiff":
        fit_params, model, residual = fitter.run_3Ddiff_fit()
        D = float(fit_params["diff_coeff"].value)
        amp = float(fit_params["amplitude"].value)
        offset = float(fit_params["offset"].value)
        N = float(0.35 / amp)
        summary = dict(filepath=rics_file, model=model_type, N=N, D=D, offset=offset,
                       residual=float(np.mean(residual)))
    elif model_type == "2comp2Ddiff":
        fit_params, model, residual = fitter.run_2comp2Ddiff_fit()
        D2 = float(fit_params["diff_coeff2"].value)
        D1 = float(fit_params["fact"].value) * D2
        N1 = float(fit_params["N1"].value)
        N2 = float(fit_params["N2"].value)
        offset = float(fit_params["offset"].value)
        summary = dict(filepath=rics_file, model=model_type,
                       N1=N1, D1=D1, N2=N2, D2=D2, offset=offset,
                       residual=float(np.mean(residual)))
    else:
        raise ValueError(f"Unknown diffusion_model: {model_type}")

    # ---- optional 1D fast-axis fit ----
    model_1D = None
    residual_1D = None
    D_1D = None
    if do_fit_1d and model_type != "2comp2Ddiff":
        fit_params_1D, model_1D, residual_1D = fitter.fast_axis_diff_fit()
        D_1D = float(fit_params_1D["diff_coeff"].value)
        summary["D_1D"] = D_1D
        summary["residual_1D"] = float(np.mean(residual_1D))

    # ---- append to CSV ----
    header = not os.path.exists(saving_path)
    pd.DataFrame([summary]).to_csv(saving_path, index=False, mode="a", header=header)

    # ---- save arrays for GUI plotting (avoid queue transfer) ----
    npz_path = os.path.splitext(rics_file)[0] + "_fit_arrays.npz"
    np.savez_compressed(
        npz_path,
        rics_map=R,
        model=model,
        residual=residual,
        model_1D=model_1D if model_1D is not None else np.array([]),
        residual_1D=residual_1D if residual_1D is not None else np.array([]),
    )

    summary["used_metadata"] = used_metadata
    summary["metadata_path"] = metadata_path if used_metadata else ""

    return summary, npz_path


def fit_rics_process_main(params, out_q):
    """
    Worker entry point.
    Supports single file OR batch (list of rics files).

    params keys:
      mode: "single" or "batch"
      rics_file OR rics_files (list)
      ... plus keys required by fit_rics_one_file()
    """
    try:
        out_q.put(("progress", 0.0))

        mode = params.get("mode", "single")

        if mode == "single":
            out_q.put(("progress", 5.0))
            summary, npz_path = fit_rics_one_file(params)
            out_q.put(("progress", 100.0))
            out_q.put(("done", {"summary": summary, "npz_path": npz_path}))
            return

        # batch
        rics_files = list(params["rics_files"])
        n = len(rics_files)
        if n == 0:
            raise ValueError("No files provided for batch fitting")

        for k, f in enumerate(rics_files, start=1):
            # per-file params
            p = dict(params)
            p["rics_file"] = f
            p["mode"] = "single"

            out_q.put(("file_start", {"index": k, "total": n, "file": f}))
            summary, npz_path = fit_rics_one_file(p)
            out_q.put(("file_done", {"summary": summary, "npz_path": npz_path}))

            out_q.put(("progress", 100.0 * k / n))

        out_q.put(("done", {"n_total": n}))

    except Exception:
        out_q.put(("error", traceback.format_exc()))