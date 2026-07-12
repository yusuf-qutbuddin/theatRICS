from __future__ import annotations

import os
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from skimage import filters
from skimage import img_as_ubyte
from skimage.io import imsave
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# optional picasso import — fall back gracefully if not installed
try:
    import picasso.io as p_io
    PICASSO_AVAILABLE = True
except ImportError:
    PICASSO_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False


DEFAULT_ICS_CONFIG = {
    "block_length": 10,
    "threshold_multiplication": 0.1,
    "frame_skip": 1,
    "bin_frames": 1,
    "save_block_images": True,
    "pattern": "*.tiff",
}


# ────────────────────────────────────────────────────────────────
# Low-level helpers
# ────────────────────────────────────────────────────────────────

def _load_movie(path: str) -> np.ndarray:
    """
    Load a TIFF or PTU movie to a float64 ndarray of shape (T, H, W).

    For TIFF: tries picasso first, falls back to tifffile.
    For PTU:  uses tttrlib via read_ptu_stack() from export_rics.
    """
    ext = os.path.splitext(path)[1].lower()

    # ── PTU (PicoQuant Luminosa raster image) ────────────────────
    if ext == ".ptu":
        from theatrics.modules.export_rics import (
            read_ptu_stack,
            TTTRLIB_AVAILABLE,
        )
        if not TTTRLIB_AVAILABLE:
            raise ImportError(
                "tttrlib is not installed. "
                "Install it with:  pip install tttrlib"
            )
        # read_ptu_stack returns (n_frames, n_lines, n_pixels) float32
        # channel=0 is the default; ICS worker passes channel via config
        channel = 0   # overridden below if passed in config
        arr = read_ptu_stack(path, channel=channel).astype(np.float64)
        return arr

    # ── TIFF ─────────────────────────────────────────────────────
    if PICASSO_AVAILABLE:
        file, _ = p_io.load_movie(path)
        if isinstance(file, p_io.TiffMultiMap):
            n_frames       = len(file)
            shape_y, shape_x = file[0].shape
            arr = np.zeros((n_frames, shape_y, shape_x), dtype=np.float64)
            for i, frame in enumerate(file):
                arr[i] = frame
            return arr
        else:
            arr = np.asarray(file, dtype=np.float64)
    elif TIFFFILE_AVAILABLE:
        arr = tifffile.imread(path).astype(np.float64)
    else:
        raise RuntimeError(
            "Neither picasso nor tifffile is installed. "
            "Install one with:  pip install picasso  or  pip install tifffile"
        )

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 1:
        raise ValueError(f"Unexpected 1D array from {path}")
    return arr


def _prepare_frames(arr: np.ndarray, block_length: int,
                    bin_frames: int) -> np.ndarray:
    """
    Optionally bin frames and truncate to a whole number of blocks.
    Returns the prepared array ready for block-wise analysis.
    """
    # discard leading frame if remainder would be exactly 1
    if arr.shape[0] % block_length == 1:
        arr = arr[1:]

    # frame binning
    if bin_frames > 1:
        usable = (arr.shape[0] // bin_frames) * bin_frames
        arr = arr[:usable]
        binned = arr[::bin_frames].copy()
        for j in range(1, bin_frames):
            binned += arr[j::bin_frames]
        binned /= bin_frames
        arr = binned

    # truncate to whole blocks
    usable = (arr.shape[0] // block_length) * block_length
    return arr[:usable]


def _compute_block(block: np.ndarray,
                   frame_skip: int,
                   threshold_multiplication: float):
    """
    Run ICS on one temporal block.

    Returns
    -------
    mean_px         : (H, W) float64  — time-averaged intensity
    G_map           : (H, W) float64  — normalised correlation map
    mask            : (H, W) bool     — pixels included in statistics
    MIP             : (H, W) float64  — maximum intensity projection
    stats           : dict with keys  mean_G, sd_G, se_G, n_pixels
                      (all NaN if mask is empty)
    """
    mean_px = np.mean(block, axis=0)
    dF = block - mean_px                                      # fluctuations

    # temporal autocorrelation at lag frame_skip
    num = np.mean(dF[:-frame_skip] * dF[frame_skip:], axis=0)

    # normalise by <F>²
    with np.errstate(divide="ignore", invalid="ignore"):
        G_map = num / (mean_px ** 2)
        G_map[~np.isfinite(G_map)] = 0.0

    # threshold mask from MIP
    MIP = np.max(block, axis=0)
    thresh = filters.threshold_yen(MIP)
    mask = MIP > thresh * threshold_multiplication

    if np.any(mask):
        vals = G_map[mask]
        n = len(vals)
        stats = {
            "mean_G": float(np.mean(vals)),
            "sd_G":   float(np.std(vals)),
            "se_G":   float(np.std(vals) / np.sqrt(n)),
            "n_pixels": int(n),
        }
    else:
        stats = {
            "mean_G":   float("nan"),
            "sd_G":     float("nan"),
            "se_G":     float("nan"),
            "n_pixels": 0,
        }

    return mean_px, G_map, mask, MIP, stats


def _save_block_images(base: str, i_block: int,
                       MIP, mean_px, mask, G_map):
    """Save the four diagnostic images for one block."""
    imsave(base + f"_b{i_block}_MIP.tiff",  MIP,
           check_contrast=False)
    imsave(base + f"_b{i_block}_mean.tiff", mean_px,
           check_contrast=False)
    imsave(base + f"_b{i_block}_mask.tiff",
           img_as_ubyte(mask), check_contrast=False)
    imsave(base + f"_b{i_block}_corr.tiff", G_map.astype(np.float32),
           check_contrast=False)


def _make_overview_figure(blocks_df: pd.DataFrame,
                          stem: str,
                          mean_stack: np.ndarray,
                          G_stack: np.ndarray) -> plt.Figure:
    """
    Build a 2×2 overview figure for one TIFF file.

    Top-left  : mean G ± SD vs block number
    Top-right : normalised G vs block number
    Bottom-left  : mean intensity image (last block)
    Bottom-right : G map (last block)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(stem, fontsize=11, fontweight="bold")

    # ── top-left: mean G with error bars ──
    ax = axes[0, 0]
    ax.errorbar(blocks_df.index, blocks_df["mean_G"],
                yerr=blocks_df["sd_G"],
                marker="o", capsize=4, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Block number")
    ax.set_ylabel("Mean G (masked pixels)")
    ax.set_title("ICS correlation vs time")
    ax.grid(True, alpha=0.3)

    # ── top-right: normalised G ──
    ax = axes[0, 1]
    ax.plot(blocks_df.index, blocks_df["Normalized"],
            marker="o", color="seagreen", linewidth=1.5)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Block number")
    ax.set_ylabel("Normalised G (relative to block 0)")
    ax.set_title("Normalised correlation")
    ax.grid(True, alpha=0.3)

    # ── bottom-left: mean intensity of last block ──
    ax = axes[1, 0]
    if mean_stack is not None and len(mean_stack) > 0:
        im = ax.imshow(mean_stack[-1], cmap="gray")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Mean intensity (last block)")
    ax.axis("off")

    # ── bottom-right: G map of last block ──
    ax = axes[1, 1]
    if G_stack is not None and len(G_stack) > 0:
        vmax = float(np.nanpercentile(G_stack[-1], 99))
        im = ax.imshow(G_stack[-1], cmap="hot", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("G map (last block)")
    ax.axis("off")

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────
# Per-file analysis
# ────────────────────────────────────────────────────────────────

def analyse_tiff(path: str,
                 config: dict,
                 progress_queue=None,
                 cancel_event=None) -> dict:
    """
    Full ICS pipeline for a single TIFF or PTU file.
    """
    path = str(path)
    base, _ = os.path.splitext(path)
    stem = os.path.basename(base)
    ext  = os.path.splitext(path)[1].lower()

    block_length           = config.get("block_length", 10)
    threshold_mult         = config.get("threshold_multiplication", 0.1)
    frame_skip             = config.get("frame_skip", 1)
    bin_frames             = config.get("bin_frames", 1)
    save_block_images_flag = config.get("save_block_images", True)
    channel                = config.get("channel", 0)

    # ── load ──────────────────────────────────────────────────────
    if ext == ".ptu":
        from theatrics.modules.export_rics import (
            read_ptu_stack,
            TTTRLIB_AVAILABLE,
        )
        if not TTTRLIB_AVAILABLE:
            raise ImportError(
                "tttrlib is not installed. "
                "Install it with:  pip install tttrlib"
            )
        arr = read_ptu_stack(path, channel=channel).astype(np.float64)
    else:
        arr = _load_movie(path)

    if arr.ndim == 2:
        arr = arr[np.newaxis]

    # ── prepare ───────────────────────────────────────────────────
    arr      = _prepare_frames(arr, block_length, bin_frames)
    n_blocks = arr.shape[0] // block_length

    rows       = []
    mean_stack = []
    G_stack    = []

    for i_block in range(n_blocks):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Cancelled")

        block = arr[i_block * block_length:(i_block + 1) * block_length]

        mean_px, G_map, mask, MIP, stats = _compute_block(
            block, frame_skip, threshold_mult
        )

        if save_block_images_flag:
            _save_block_images(base, i_block, MIP, mean_px, mask, G_map)

        rows.append({"block": i_block, **stats})
        mean_stack.append(mean_px)
        G_stack.append(G_map)

        if progress_queue is not None:
            progress_queue.put((
                "block_done",
                {"block": i_block, "n_blocks": n_blocks,
                 "stats": stats, "stem": stem}
            ))

    # ── build per-file DataFrame ───────────────────────────────────
    df = pd.DataFrame(rows).set_index("block")
    first_valid = df["mean_G"].dropna()
    if len(first_valid) > 0:
        df["Normalized"] = df["mean_G"] / first_valid.iloc[0]
    else:
        df["Normalized"] = float("nan")

    # ── save CSV ───────────────────────────────────────────────────
    csv_path = base + f"_threshold{threshold_mult}_corr.csv"
    df.to_csv(csv_path, index=True)

    # ── build and save overview figure ────────────────────────────
    mean_stack_arr = np.stack(mean_stack) if mean_stack else None
    G_stack_arr    = np.stack(G_stack)    if G_stack    else None

    fig = _make_overview_figure(df, stem, mean_stack_arr, G_stack_arr)
    svg_path = base + "_ICS_overview.svg"
    fig.savefig(svg_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "path":       path,
        "stem":       stem,
        "csv_path":   csv_path,
        "svg_path":   svg_path,
        "blocks_df":  df,
        "mean_stack": mean_stack_arr,
        "G_stack":    G_stack_arr,
        "n_blocks":   n_blocks,
    }


# ────────────────────────────────────────────────────────────────
# Sample-level aggregation
# ────────────────────────────────────────────────────────────────

def aggregate_sample(tiff_results: list[dict],
                     sample_name: str,
                     out_dir: str) -> dict:
    """
    Aggregate normalised correlations across all TIFFs in one sample folder.
    Returns a dict with the aggregated DataFrame and its CSV path.
    """
    norm_cols = []
    for res in tiff_results:
        col = res["blocks_df"]["Normalized"].rename(res["stem"])
        norm_cols.append(col)

    if not norm_cols:
        return {}

    concat = pd.concat(norm_cols, axis=1)
    agg = pd.DataFrame({
        f"{sample_name}_mean": concat.mean(axis=1),
        f"{sample_name}_std":  concat.std(axis=1),
    })

    csv_path = os.path.join(out_dir, f"{sample_name}_aggregated.csv")
    agg.to_csv(csv_path, index=True)

    return {
        "sample": sample_name,
        "agg_df": agg,
        "csv_path": csv_path,
        "n_tiffs": len(tiff_results),
    }


# ────────────────────────────────────────────────────────────────
# Batch — one folder containing sample subfolders
# ────────────────────────────────────────────────────────────────

def run_ics_batch(parent_dir: str,
                  config: dict,
                  progress_queue=None,
                  cancel_event=None) -> dict:
    """
    Iterate over sample subfolders, process all TIFF and PTU files,
    aggregate per sample, and write a global combined CSV and plot.
    """
    parent_dir = str(parent_dir)
    pattern    = config.get("pattern", "*.tiff")
    ext        = os.path.splitext(pattern)[-1].lower()

    # support PTU pattern directly
    is_ptu = ext == ".ptu"

    glob_results = {}
    failed       = []

    subdirs = sorted(
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    )

    def _matching_files(folder):
        return [
            f for f in os.listdir(folder)
            if f.lower().endswith(ext)
        ]

    total_files = sum(
        len(_matching_files(os.path.join(parent_dir, s)))
        for s in subdirs
    )
    done_files = 0

    if progress_queue is not None:
        progress_queue.put(("progress", 0.0))

    for sample in subdirs:
        if cancel_event is not None and cancel_event.is_set():
            break

        sample_path = os.path.join(parent_dir, sample)
        file_names  = _matching_files(sample_path)
        if not file_names:
            continue

        tiff_results = []
        for fname in file_names:
            if cancel_event is not None and cancel_event.is_set():
                break
            fpath = os.path.join(sample_path, fname)
            try:
                res = analyse_tiff(fpath, config,
                                   progress_queue=progress_queue,
                                   cancel_event=cancel_event)
                tiff_results.append(res)
                if progress_queue is not None:
                    progress_queue.put(("file_done", res))
            except Exception:
                failed.append(fpath)
                if progress_queue is not None:
                    progress_queue.put(("error_file", fpath))

            done_files += 1
            if progress_queue is not None:
                progress_queue.put((
                    "progress",
                    100.0 * done_files / max(1, total_files)
                ))

        if tiff_results:
            agg = aggregate_sample(tiff_results, sample, sample_path)
            glob_results[sample] = agg

    # ── global CSV and plot ───────────────────────────────────────
    timestamp     = datetime.now().strftime("%Y-%m-%d_%H-%M")
    combined_path = None
    combined_fig_path = None

    if glob_results:
        frames   = [v["agg_df"] for v in glob_results.values()]
        combined = pd.concat(frames, axis=1)
        combined.dropna(how="all", inplace=True)

        combined_path = os.path.join(
            parent_dir,
            f"{timestamp}_allSamples_ICS.csv"
        )
        combined.to_csv(combined_path)

        fig, ax = plt.subplots(figsize=(8, 5))
        for sample, v in glob_results.items():
            df       = v["agg_df"]
            mean_col = f"{sample}_mean"
            std_col  = f"{sample}_std"
            if mean_col in df.columns:
                ax.errorbar(
                    df.index, df[mean_col],
                    yerr=df[std_col],
                    marker="o", capsize=4, label=sample
                )
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Block number")
        ax.set_ylabel("Normalised G (mean ± SD)")
        ax.set_title("ICS — all samples")
        ax.legend()
        fig.tight_layout()

        combined_fig_path = os.path.join(
            parent_dir,
            f"{timestamp}_allSamples_ICS.svg"
        )
        fig.savefig(combined_fig_path, dpi=300,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return {
        "glob_results":  glob_results,
        "failed":        failed,
        "combined_csv":  combined_path,
        "combined_fig":  combined_fig_path,
        "n_total":       total_files,
        "n_ok":          total_files - len(failed),
        "n_failed":      len(failed),
    }