import numpy as np
import multiprocessing as mp
import traceback
from modules import SFCS_module

def sfcs_process_main_curvefit(input_file, channel, cpu_n, out_q,
                               chunk_lines=500, max_workers=None):
    """
    Whole SFCS pipeline in a separate process.
    Gaussian fitting uses curve_fit in parallel via multiprocessing.Pool + chunking.

    out_q messages: ("progress", pct) / ("done", payload) / ("error", tb)
    """
    try:
        out_q.put(("progress", 0.0))

        frame_data, line_time_s, x, n_lines, n_pixels, root = SFCS_module.read_file(str(input_file), int(channel))
        out_q.put(("progress", 2.0))

        # ---- build chunk tasks ----
        chunk_lines = int(chunk_lines)
        if chunk_lines < 50:
            chunk_lines = 50

        tasks = []
        for i0 in range(0, n_lines, chunk_lines):
            block = frame_data[i0:i0 + chunk_lines]          # view
            tasks.append((i0, block, n_pixels))

        n_tasks = len(tasks)

        peaks = np.full(n_lines, 0.0, dtype=float)
        sigmas = np.full(n_lines, 5.0, dtype=float)

        # workers: don't blindly use 256; curve_fit overhead + OS overhead matters
        cpu_n = int(cpu_n)
        if max_workers is not None:
            cpu_n = min(cpu_n, int(max_workers))
        cpu_n = max(1, cpu_n)

        # IMPORTANT on Windows/Linux: do NOT make this SFCS worker process daemonic
        # (you already fixed that). This pool is created inside this worker process.

        out_q.put(("progress", 5.0))

        # ---- parallel chunk fitting ----
        # chunksize=1 here because "tasks" are already chunky
        with mp.Pool(processes=cpu_n) as pool:
            
            completed_lines = 0
            for chunk_result in pool.imap(SFCS_module.fit_gaussian_chunk, tasks, chunksize=1):
                # chunk_result is list of (i, peak, sigma)
                for (i, peak, sigma) in chunk_result:
                    peaks[i] = peak
                    sigmas[i] = sigma

                completed_lines += len(chunk_result)
                pct = 5.0 + 55.0 * (completed_lines / n_lines)   # 5%->60%
                out_q.put(("progress", pct))

        # ---- Remaining SFCS steps ----
        out_q.put(("progress", 65.0))
        aligned_data = SFCS_module.alignment(frame_data, n_pixels, n_lines, root, peaks)

        out_q.put(("progress", 75.0))
        intensity_traces = SFCS_module.calculate_intensity_trace(aligned_data, n_lines, n_pixels, sigmas, root)

        out_q.put(("progress", 90.0))
        G, G_std = SFCS_module.run_autocorrelation(intensity_traces, line_time_s, root)

        out_q.put(("progress", 100.0))
        out_q.put(("done", dict(
            frame_data=frame_data,
            aligned_data=aligned_data,
            intensity_traces=intensity_traces,
            G=G,
            G_std=G_std
        )))

    except Exception:
        out_q.put(("error", traceback.format_exc()))