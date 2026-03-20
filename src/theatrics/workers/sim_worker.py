# workers/sim_worker.py
import os
import numpy as np
import multiprocessing as mp
import tifffile
import traceback
from theatrics.modules import simRICS


def _dispatch_worker(sim_type: str):
    if sim_type == "isotropic":
        return simRICS.simulate_single_frame
    if sim_type == "anisotropic":
        return simRICS.simulate_single_frame_aniso
    if sim_type == "anisotropic_rotated":
        return simRICS.simulate_single_frame_aniso_rotated
    raise ValueError(f"Unknown sim_type: {sim_type}")


def simulate_rics_process_main(params, out_q, cancel_event=None):
    """
    params keys (suggested):
      sim_type: "isotropic" | "anisotropic" | "anisotropic_rotated"
      img_shape: (H, W)
      n_frames: int
      pixel_dwell_time_us: float
      pixel_size_nm: float
      brightness_khz: float
      n_particles: int
      diffusion_um2_s: float                       (for isotropic)
      diffusion_um2_s_x: float, diffusion_um2_s_y: float  (for anisotropic)
      rotation_deg: float                           (for rotated)
      background: float
      psf_sigma_px: float
      output_path: str
      cpu_n: int
    """
    try:
        

        out_q.put(("progress", 0.0))

        sim_type = params["sim_type"]
        img_shape = tuple(params["img_shape"])
        n_frames = int(params["n_frames"])
        cpu_n = max(1, int(params["cpu_n"]))
        output_path = params["output_path"]

        # Optional: save sim params for reproducibility (like you did)
        txt_path = os.path.splitext(output_path)[0] + ".txt"
        try:
            with open(txt_path, "w") as f:
                for k, v in params.items():
                    f.write(f"{k}: {v}\n")
        except Exception:
            pass

        worker_fn = _dispatch_worker(sim_type)

        # Build frame args (match your simRICS signatures)
        frame_args = []
        for idx in range(n_frames):
            seed = int(np.random.randint(0, 1_000_000))

            if sim_type == "isotropic":
                frame_args.append((
                    idx, img_shape,
                    float(params["pixel_dwell_time_us"]),
                    float(params["pixel_size_nm"]),
                    float(params["brightness_khz"]),
                    int(params["n_particles"]),
                    float(params["diffusion_um2_s"]),
                    float(params["background"]),
                    float(params["psf_sigma_px"]),
                    seed
                ))

            elif sim_type == "anisotropic":
                frame_args.append((
                    idx, img_shape,
                    float(params["pixel_dwell_time_us"]),
                    float(params["pixel_size_nm"]),
                    float(params["brightness_khz"]),
                    int(params["n_particles"]),
                    float(params["diffusion_um2_s_x"]),
                    float(params["diffusion_um2_s_y"]),
                    float(params["background"]),
                    float(params["psf_sigma_px"]),
                    seed
                ))

            else:  # anisotropic_rotated
                frame_args.append((
                    idx, img_shape,
                    float(params["pixel_dwell_time_us"]),
                    float(params["pixel_size_nm"]),
                    float(params["brightness_khz"]),
                    int(params["n_particles"]),
                    float(params["diffusion_um2_s_x"]),
                    float(params["diffusion_um2_s_y"]),
                    float(params["rotation_deg"]),
                    float(params["background"]),
                    float(params["psf_sigma_px"]),
                    seed
                ))

        out_q.put(("progress", 2.0))

        stack = []
        with mp.Pool(processes=cpu_n) as pool:
            for i, frame in enumerate(pool.imap(worker_fn, frame_args, chunksize=1), start=1):

                

                stack.append(frame)
                out_q.put(("progress", 2.0 + 96.0 * (i / n_frames)))
                if cancel_event.is_set():
                    pool.terminate()
                    pool.join()
                    out_q.put(("cancelled", None))
                    return

        stack = np.stack(stack, axis=0)
        tifffile.imwrite(output_path, stack, photometric="minisblack")

        out_q.put(("progress", 100.0))
        out_q.put(("done", {"output_path": output_path, "shape": stack.shape}))

    except Exception:
        out_q.put(("error", traceback.format_exc()))