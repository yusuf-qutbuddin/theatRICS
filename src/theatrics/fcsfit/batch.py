from __future__ import annotations

import os
import glob
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from theatrics.fcsfit import models_and_fit as fit


@dataclass
class BatchSummary:
    n_total: int
    n_ok: int
    n_failed: int
    failed: List[str]


def _strip_csv(path: str) -> str:
    return path[:-4] if path.lower().endswith(".csv") else path


def run_single_csv(
    csv_path: str,
    fitting_model: str,
    tau_domain: Tuple[float, float] = (1e-6, 1.0),
    user_tau_domain: bool = True,
    psf_radius_um: float = 0.25,
    psf_aspect_ratio: float = 5.0,
    given_D: Tuple[float, float] = (435.0, 25.0),
    experiment_T: float = 30.0,
    BG_value: float = 0.0,
    user_initial_params: bool = True,
    initial_params: Optional[Dict[str, Any]] = None,
    goodness_of_fit_criterion: Optional[List[str]] = None,
    figure_display_delay: float = 0.001,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Fit one correlation CSV.

    Writes outputs to a `Results/` folder next to the csv, because
    models_and_fit.main() already implements that behavior.
    """
    if cancel_event is not None and cancel_event.is_set():
        return {"status": "cancelled"}

    if initial_params is None:
        initial_params = {"N": 0.5, "tau diffusion": 1e-4}

    if goodness_of_fit_criterion is None:
        goodness_of_fit_criterion = ["instant_correlation_runsstest"]

    # Compute corrected_D (your batch code does this)
    D_val, D_temp = given_D
    A1 = 20 - D_temp
    A2 = 20 - experiment_T
    B1 = 96 + D_temp
    B2 = 96 + experiment_T
    viscosity_term = 10 ** (
        ((A1 / B1) * (1.2364 - 0.00137 * A1 + 0.0000057 * A1**2))
        - ((A2 / B2) * (1.2364 - 0.00137 * A2 + 0.0000057 * A2**2))
    )
    corrected_D = D_val * ((experiment_T + 273.15) / (D_temp + 273.15)) * viscosity_term

    # Save path: next to file in Results/
    base = _strip_csv(csv_path)
    folder = os.path.dirname(base)
    results_dir = os.path.join(folder, "Results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{fitting_model}_fit_summary.csv")

    # The legacy function expects `path` WITHOUT ".csv" and does `path + ".csv"`
    res = fit.main(
        path=base,
        fitting_model=fitting_model,
        result_name=fitting_model,
        corrected_D=corrected_D,
        save_path=save_path,
        BG=BG_value,
        PSF_radius=psf_radius_um,
        PSF_aspect_ratio=psf_aspect_ratio,
        user_initial_params=user_initial_params,
        initial_params=initial_params,
        figure_display_delay=figure_display_delay,
        user_tau_domain=user_tau_domain,
        tau_domain=tau_domain,
        goodness_of_fit_criterion=goodness_of_fit_criterion,
    )

    res["status"] = "ok"
    res["save_path"] = save_path
    res["results_dir"] = results_dir
    return res


def run_batch_folder(
    folder: str,
    pattern: str = "*.csv",
    progress_queue=None,
    cancel_event=None,
    **kwargs
) -> dict:
    """
    Recursively find matching CSV files in all subfolders of `folder`,
    fit them all, and write one summary CSV in the OUTER folder.

    Per-file outputs are still handled locally by the GUI display/export
    if desired, while the batch summary is written once here.
    """
    csvs = sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))

    n_total = len(csvs)
    n_ok = 0
    failed = []
    last_res = None
    summary_rows = []

    # one global summary in the outer folder
    outer_results_dir = os.path.join(folder, "Results")
    os.makedirs(outer_results_dir, exist_ok=True)

    fitting_model = kwargs["fitting_model"]
    pattern_corrected = pattern[1:]
    summary_csv = os.path.join(outer_results_dir, f"{fitting_model}_{pattern_corrected}_fit_summary.csv")

    if progress_queue is not None:
        progress_queue.put(("progress", 0.0))

    for i, csv_path in enumerate(csvs):
        if cancel_event is not None and cancel_event.is_set():
            break

        try:
            res = run_single_csv(csv_path, cancel_event=cancel_event, **kwargs)
            last_res = res
            n_ok += 1

            # ── per-file: save fit curve CSV + SVG for every file ──
            base = _strip_csv(csv_path)
            fitting_model = kwargs["fitting_model"]

            tau = np.asarray(res["tau"], dtype=float)
            G = np.asarray(res["G"], dtype=float)
            sigma = np.asarray(res["sigma_G"], dtype=float)
            pred = np.asarray(res["ccPrediction"], dtype=float)

            per_file_results_dir = os.path.join(os.path.dirname(base), "Results")
            os.makedirs(per_file_results_dir, exist_ok=True)
            edit_base = os.path.join(per_file_results_dir,
                                     os.path.basename(base) + "_" + fitting_model)

            # fit curve CSV
            pd.DataFrame({
                "tau": tau, "G": G,
                "sigma G": sigma, "cc Fit": pred,
            }).to_csv(edit_base + ".csv", index=False)

            # fit curve SVG
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.semilogx(tau, G, "r", label="G observed")
            ax.semilogx(tau, pred, "g", label="G fit")
            ax.fill_between(tau, G - sigma, G + sigma,
                            color="b", alpha=0.2, label="±σ")
            ax.set_xlabel("τ (s)")
            ax.set_ylabel("G(τ)")
            ax.set_title(os.path.basename(csv_path))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(edit_base + ".svg",
                        dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            # MEMFCS distribution is already saved by models_and_fit.main()
            # but log the paths if present
            if fitting_model == "g3diffMEMFCS":
                dist_csv = edit_base.replace(
                    "_" + fitting_model, ""
                ) + "_MEMFCS_distribution.csv"
                # (already written by _save_memfcs_distribution inside main())

            # summary row
            estimate = res.get("estimate_data", {})
            row = {}
            for k, v in estimate.items():
                if v == [None]:
                    continue
                row[k] = v[0] if (isinstance(v, list) and len(v) == 1) else v
            row["Filename"] = res.get("base_path", csv_path)
            summary_rows.append(row)

            if progress_queue is not None:
                progress_queue.put(("file_done", res))
                progress_queue.put(("progress", 100.0 * (i + 1) / max(1, n_total)))

        except Exception:
            failed.append(csv_path)
            if progress_queue is not None:
                progress_queue.put(("progress", 100.0 * (i + 1) / max(1, n_total)))

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(summary_csv, index=False)
    else:
        summary_csv = None

    return {
        "summary": {
            "n_total": n_total,
            "n_ok": n_ok,
            "n_failed": len(failed),
            "failed": failed,
            "summary_csv": summary_csv,
        },
        "last_res": last_res,
    }