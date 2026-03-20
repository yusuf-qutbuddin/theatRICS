# workers/diffmap_worker.py
import os
import numpy as np
import multiprocessing as mp
import tifffile
import traceback
import scipy.ndimage

from pylibCZIrw import czi as pyczi
from theatrics.modules import inspect_metadata as im
from theatrics.modules import export_rics
from theatrics.modules import rics_fit


def _process_block_diff_map(args):
    """
    args: (block, y0, x0, pixelsize_um, pixeltime_s, linetime_s,
           psf_xy_um, psf_aspect, model)
    Returns:
      (y0, x0, D, amp, brightness, rics_fast, model_fast)
    """
    (block, y0, x0, pixelsize_um, pixeltime_s, linetime_s,
     psf_xy_um, psf_aspect, model) = args

    # basic validity check
    if np.count_nonzero(~np.isnan(block)) < 0.5 * block.size:
        return (y0, x0, np.nan, np.nan, np.nan, np.array([]), np.array([]))

    try:
        # Compute RICS on block (tiff workflow expects stack)
        RICS_map, sd_map, stack, corrected_stack = export_rics.process_all_frames_tiff(
            block, block.shape[0], 0, window_size=3
        )

        cy = RICS_map.shape[0] // 2
        cx = RICS_map.shape[1] // 2
        RICS_map[cy, cx] = 0.0

        rics_fast = RICS_map[cy, :]

        fitter = rics_fit.RICS_fit(
            RICS_map,
            pixelsize_um, pixeltime_s, linetime_s,
            psf_xy_um, psf_aspect
        )

        if model == "3Ddiff":
            params, modelmap, res = fitter.run_3Ddiff_fit()
        else:
            params, modelmap, res = fitter.run_2Ddiff_fit()

        D = float(params["diff_coeff"].value)
        amp = float(params["amplitude"].value)
        model_fast = modelmap[cy, :]

    except Exception:
        D, amp = np.nan, np.nan
        rics_fast = np.array([])
        model_fast = np.array([])

    brightness = float(np.std(block))
    return (y0, x0, D, amp, brightness, rics_fast, model_fast)


def diffusion_map_process_main(params, out_q, cancel_event=None):
    """
    params keys:
      input_file (czi), channel (int),
      psf_xy_um, psf_aspect_ratio,
      window_size, offset,
      diffusion_model ("2Ddiff" or "3Ddiff"),
      cpu_n (optional)
    """
    try:
        if cancel_event.is_set():
                    
                    out_q.put(("cancelled", None))
                    return

        out_q.put(("progress", 0.0))

        input_file = params["input_file"]
        channel = int(params["channel"])
        psf_xy_um = float(params["psf_xy_um"])
        psf_aspect = float(params["psf_aspect_ratio"])
        window_size = int(params["window_size"])
        offset = int(params["offset"])
        model = params["diffusion_model"]
        cpu_n = int(params.get("cpu_n", mp.cpu_count()))
        cpu_n = max(1, min(cpu_n, mp.cpu_count()))

        # ---- metadata ----
        with pyczi.open_czi(input_file) as czidoc:
            Pixel_size_nm, Pixel_dwell_time_us, line_time_ms = im.get_metadata(czidoc, channel)
            n_frames = czidoc.total_bounding_box["T"][1]

        pixelsize_um = float(Pixel_size_nm) * 1e-3
        pixeltime_s = float(Pixel_dwell_time_us) * 1e-6
        linetime_s = float(line_time_ms) * 1e-3
        if cancel_event.is_set():
                    
                    out_q.put(("cancelled", None))
                    return
        out_q.put(("progress", 5.0))

        # ---- read frames to stack ----
        all_frames = []
        for i_frame in range(n_frames):
            if cancel_event.is_set():
                out_q.put(("cancelled", None))
                return
            frame = export_rics.read_frame(input_file, i_frame, channel)
            all_frames.append(frame)

        stack = np.stack(all_frames, axis=0)  # (T, Y, X)
        h, w = stack.shape[-2], stack.shape[-1]

        out_q.put(("progress", 15.0))

        # ---- build block tasks ----
        nx = (w - window_size) // offset + 1
        ny = (h - window_size) // offset + 1

        tasks = []
        for iy in range(ny):
            for ix in range(nx):
                y0 = iy * offset
                x0 = ix * offset
                block = stack[:, y0:y0 + window_size, x0:x0 + window_size].copy()
                tasks.append((block, y0, x0, pixelsize_um, pixeltime_s, linetime_s, psf_xy_um, psf_aspect, model))

        total = len(tasks)
        if total == 0:
            raise ValueError("No blocks to process (check window_size/offset vs image size)")

        # output maps
        Dmap = np.full((h, w), np.nan, dtype=np.float32)
        Nmap = np.full((h, w), np.nan, dtype=np.float32)
        Bmap = np.full((h, w), np.nan, dtype=np.float32)
        fast_list = []
        model_fast_list = []

        out_q.put(("progress", 20.0))

        # ---- pool over blocks ----
        completed = 0
        with mp.Pool(processes=cpu_n) as pool:
            for result in pool.imap(_process_block_diff_map, tasks, chunksize=1):
                if cancel_event.is_set():
                    pool.terminate()
                    pool.join()
                    out_q.put(("cancelled", None))
                    return

                y0, x0, D, amp, brightness, rics_fast, model_fast = result

                Dmap[y0:y0 + window_size, x0:x0 + window_size] = D
                Nmap[y0:y0 + window_size, x0:x0 + window_size] = amp
                Bmap[y0:y0 + window_size, x0:x0 + window_size] = brightness
                if rics_fast.size:
                    fast_list.append(rics_fast)
                    model_fast_list.append(model_fast)

                completed += 1
                out_q.put(("progress", 20.0 + 75.0 * completed / total))

        # ---- smooth / filter ----
        Dmap = scipy.ndimage.median_filter(Dmap, size=3)
        Nmap = scipy.ndimage.median_filter(Nmap, size=3)
        Bmap = scipy.ndimage.median_filter(Bmap, size=3)

        # ---- save outputs ----
        root, _ = os.path.splitext(input_file)
        diff_map_output = root + "_diff_map.tif"
        aux_output = root + "_diff_aux.npz"

        tifffile.imwrite(diff_map_output, Dmap, photometric="minisblack")
        np.savez_compressed(aux_output, Dmap=Dmap, Nmap=Nmap, Bmap=Bmap,
                            fast_list=np.array(fast_list, dtype=object),
                            model_fast_list=np.array(model_fast_list, dtype=object))

        out_q.put(("progress", 100.0))
        out_q.put(("done", {
            "diff_map_output": diff_map_output,
            "aux_output": aux_output,
            "pixel_size_nm": Pixel_size_nm,
            "pixel_dwell_us": Pixel_dwell_time_us,
            "line_time_ms": line_time_ms
        }))

    except Exception:
        out_q.put(("error", traceback.format_exc()))