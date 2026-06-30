"""
debug_memfcs_jaynes.py

Tests MEMFCS with Shannon-Jaynes relative entropy using a single-
component 3D Gaussian diffusion fit as the invariant measure (prior).

Shannon-Jaynes entropy:
    S_SJ = -Σ p_i · ln(p_i / m_i)

where m_i is the prior distribution obtained from a standard
single-component 3D diffusion fit.

Comparison:
    Method A: standard Shannon entropy (flat prior m_i = 1/n)  — v9
    Method B: Shannon-Jaynes entropy (3D fit prior)            — new

The gradient of S_SJ:
    ∂S_SJ/∂α_i = -(ln(p_i/m_i) + 1)/Z + (1/Z) Σ_j p_j(ln(p_j/m_j) + 1)
               = -(ln(p_i/m_i) - S_SJ) / Z     [after simplification]

This reduces to standard Shannon gradient when m_i = 1/n (flat prior).

Run:
    python debug_memfcs_jaynes.py path/to/file.csv
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


# ════════════════════════════════════════════════════════════════
# Kernel
# ════════════════════════════════════════════════════════════════

def build_kernel(tau, tau_D, psf_aspect_ratio):
    S2    = psf_aspect_ratio ** 2
    ratio = tau[:, None] / tau_D[None, :]
    return 1.0 / ((1.0 + ratio) * np.sqrt(1.0 + ratio / S2))


# ════════════════════════════════════════════════════════════════
# Standard 3D Gaussian diffusion fit — produces the prior
# ════════════════════════════════════════════════════════════════

def fit_single_component(tau, G_data, sigma_G, psf_aspect_ratio=5.0):
    """
    Fit G(τ) = G0 / ((1 + τ/τ_D) · sqrt(1 + τ/(S²·τ_D)))

    using scipy curve_fit with the data amplitude G_data[0]
    as a fixed constraint (matching what MEMFCS uses for normalisation).

    Returns
    -------
    tau_D_fit : fitted diffusion time (s)
    G0_fit    : fitted amplitude
    success   : bool
    G_pred    : (n_tau,) predicted curve
    chi2      : reduced chi2 of single-component fit
    """
    S2 = psf_aspect_ratio ** 2
    G0_fixed = float(np.mean(G_data[:10]))

    def model(tau, tau_D):
        ratio = tau / tau_D
        return G0_fixed / ((1.0 + ratio) * np.sqrt(1.0 + ratio / S2))

    try:
        # initial guess: τ_D from midpoint of G curve
        half_max = G0_fixed / 2.0
        idx_half = np.argmin(np.abs(G_data - half_max))
        tau_D_0  = max(float(tau[idx_half]), 1e-7)

        popt, _ = curve_fit(
            model, tau, G_data,
            p0=[tau_D_0],
            bounds=(1e-8, 1.0),
            sigma=sigma_G,
            absolute_sigma=True,
            maxfev=10000,
        )
        tau_D_fit = float(popt[0])
        G_pred    = model(tau, tau_D_fit)
        residual  = G_pred - G_data
        chi2      = float(np.sum((residual / sigma_G)**2) / len(tau))

        return tau_D_fit, G0_fixed, True, G_pred, chi2

    except Exception as e:
        print(f"  Single-component fit failed: {e}")
        return 1e-4, G0_fixed, False, np.zeros_like(tau), np.inf


# ════════════════════════════════════════════════════════════════
# Prior distribution from single-component fit
# ════════════════════════════════════════════════════════════════

def make_prior_from_fit(tau_D_grid, tau_D_fit, width_decades=0.5):
    """
    Construct the invariant measure m_i from the single-component fit.

    The prior is a log-normal distribution centred at tau_D_fit
    with width width_decades in log10 space.

    m_i ∝ exp(-0.5 · (log10(tau_D_i) - log10(tau_D_fit))² / width²)

    Normalised to sum to 1.

    Parameters
    ----------
    tau_D_grid   : (n_comp,) grid of diffusion times
    tau_D_fit    : peak of the prior (from single-component fit)
    width_decades: width of the log-normal prior in decades
                   (0.5 = one half-decade, relatively tight)
                   (1.0 = one decade, broader)
                   (2.0 = two decades, very broad, approaching flat)

    Returns
    -------
    m : (n_comp,) prior distribution, sums to 1
    """
    log_grid = np.log10(tau_D_grid)
    log_peak = np.log10(max(tau_D_fit, 1e-10))
    m        = np.exp(-0.5 * ((log_grid - log_peak) / width_decades)**2)
    m        = m / np.sum(m)
    m        = np.maximum(m, 1e-10)   # floor to prevent ln(0)
    return m


# ════════════════════════════════════════════════════════════════
# Forward model with G(0) normalisation
# ════════════════════════════════════════════════════════════════

def forward_normalised(alpha, K, G_data):
    G_raw  = K @ alpha
    target = float(np.mean(G_data[:10]))
    scale  = target / max(float(G_raw[0]), 1e-300)
    return G_raw * scale, scale


def chi2_only(alpha, K, G_data, sigma_G):
    G_fit, _ = forward_normalised(alpha, K, G_data)
    r        = G_fit - G_data
    return float(np.sum((r / sigma_G)**2) / len(G_data))


# ════════════════════════════════════════════════════════════════
# Entropy functions
# ════════════════════════════════════════════════════════════════

def entropy_shannon(alpha):
    """
    Standard Shannon entropy: S = -Σ p_i ln(p_i)
    Flat prior m_i = 1/n implicitly.
    Returns S, grad_S
    """
    a     = np.maximum(alpha, 1e-300)
    Z     = float(np.sum(a))
    p     = a / Z
    p     = np.maximum(p, 1e-300)
    log_p = np.log(p)
    S     = float(-np.sum(p * log_p))
    grad  = (S - log_p) / Z
    return S, grad


def entropy_jaynes(alpha, m):
    """
    Shannon-Jaynes relative entropy: S_SJ = -Σ p_i · ln(p_i / m_i)

    m_i is the prior (invariant measure), normalised to sum to 1.
    Maximum is 0, achieved when p_i = m_i for all i.
    S_SJ ≥ standard Shannon entropy when m is peaked
    (peaked prior gives more room to explore concentrated solutions).

    Gradient:
        ∂S_SJ/∂α_i = -(ln(p_i/m_i) - S_SJ) / Z

    This equals the Shannon gradient when m_i = 1/n (flat prior)
    because then ln(p_i/m_i) = ln(p_i) + ln(n) and the constant
    ln(n) cancels in the gradient.

    Parameters
    ----------
    alpha : (n_comp,) current amplitudes
    m     : (n_comp,) prior distribution, must sum to 1 and be > 0

    Returns
    -------
    S_SJ  : scalar Shannon-Jaynes entropy  (≥ 0, maximum when p = m)
    grad  : (n_comp,) gradient ∂S_SJ/∂α_i
    """
    a   = np.maximum(alpha, 1e-300)
    Z   = float(np.sum(a))
    p   = a / Z
    p   = np.maximum(p, 1e-300)
    m_s = np.maximum(m, 1e-300)

    log_ratio = np.log(p / m_s)        # ln(p_i / m_i)
    S_SJ      = float(-np.sum(p * log_ratio))   # S_SJ = -Σ p·ln(p/m)

    # gradient: ∂S_SJ/∂α_i = -(ln(p_i/m_i) - S_SJ) / Z
    grad = -(log_ratio - S_SJ) / Z

    return S_SJ, grad


# ════════════════════════════════════════════════════════════════
# Step directions
# ════════════════════════════════════════════════════════════════

def compute_gradients_full(alpha, K, G_data, sigma_G):
    """Compute chi2, scale, residuals, and chi2 gradient."""
    M            = len(G_data)
    w            = 1.0 / sigma_G ** 2
    G_fit, scale = forward_normalised(alpha, K, G_data)
    r            = G_fit - G_data
    chi2         = float(np.sum((r / sigma_G)**2) / M)
    D_chi2       = (2.0 * scale / M) * (K.T @ (w * r))
    return chi2, D_chi2, G_fit, scale


def step_shannon(alpha, D_chi2):
    """
    v9 step using standard Shannon entropy (flat prior).
    """
    a_norm = np.maximum(alpha / (np.sum(alpha) + 1e-300), 1e-300)
    log_a  = np.log(a_norm)
    S      = float(-np.sum(a_norm * log_a))
    D_S    = -1.0 - log_a

    alpha_f = np.abs(D_chi2) / (20.0 * np.abs(D_S) + 1e-300)
    e_G     = a_norm * (alpha_f * D_S - D_chi2 / 2.0)
    return e_G, S


def step_jaynes(alpha, D_chi2, m):
    """
    Step using Shannon-Jaynes relative entropy with prior m.

    e_G = a_norm * (alpha_f * D_S_SJ - D_chi2 / 2)

    where D_S_SJ is the gradient of the relative entropy.
    alpha_f is computed the same way as v9 but using D_S_SJ.
    """
    a_norm = np.maximum(alpha / (np.sum(alpha) + 1e-300), 1e-300)
    S_SJ, D_S_SJ = entropy_jaynes(a_norm, m)

    alpha_f = np.abs(D_chi2) / (20.0 * np.abs(D_S_SJ) + 1e-300)
    e_G     = a_norm * (alpha_f * D_S_SJ - D_chi2 / 2.0)
    return e_G, S_SJ


# ════════════════════════════════════════════════════════════════
# Alpha-chop and p-chop
# ════════════════════════════════════════════════════════════════

def alpha_chop(alpha, delta, K, G_data, sigma_G,
               chi2_current, chi2_target, max_chops=60):
    a1     = np.maximum(alpha + delta, 0.0)
    chi2_1 = chi2_only(a1, K, G_data, sigma_G)
    if chi2_current > chi2_target:
        if chi2_1 <= chi2_target:
            x_lo, x_hi = 0.0, 1.0
            for _ in range(max_chops):
                x_m    = 0.5 * (x_lo + x_hi)
                a_m    = np.maximum(alpha + x_m * delta, 0.0)
                chi2_m = chi2_only(a_m, K, G_data, sigma_G)
                if chi2_m > chi2_target:
                    x_lo = x_m
                else:
                    x_hi = x_m
                if abs(x_hi - x_lo) < 1e-14:
                    break
            return 0.5 * (x_lo + x_hi)
        else:
            return 1.0
    else:
        x = 1.0
        for _ in range(max_chops):
            a_t = np.maximum(alpha + x * delta, 0.0)
            c_t = chi2_only(a_t, K, G_data, sigma_G)
            if c_t >= chi2_target:
                break
            x *= 0.5
        return x


def p_chop(alpha, delta, x, max_chops=60):
    for _ in range(max_chops):
        if np.all(alpha + x * delta >= 0.0):
            break
        x *= 0.5
    return x


# ════════════════════════════════════════════════════════════════
# Run both methods
# ════════════════════════════════════════════════════════════════

def run_both(tau, G_data, sigma_G,
             m,                          # prior for Jaynes
             tau_D_fit,                  # fitted single-component τ_D
             psf_aspect_ratio = 5.0,
             tau_D_log_range  = (-7.0, -1.0),
             n_components     = 200,
             n_iterations     = 20000,
             chi2_target      = 1.0,
             stop_criterion   = 5e-6,
             stop_window      = 100,
             check_every      = 200,
             verbose          = True):

    tau     = np.asarray(tau,     dtype=float)
    G_data  = np.asarray(G_data,  dtype=float)
    sigma_G = np.maximum(np.asarray(sigma_G, dtype=float), 1e-20)

    n_comp = n_components
    tau_D  = np.logspace(tau_D_log_range[0], tau_D_log_range[1], n_comp)
    K      = build_kernel(tau, tau_D, psf_aspect_ratio)

    # identical flat initialisation for both
    alpha_sh = np.full(n_comp, 1.0 / n_comp)
    alpha_jy = np.full(n_comp, 1.0 / n_comp)

    hist = {
        "shannon": {"chi2": [], "S": [], "x": []},
        "jaynes":  {"chi2": [], "S": [], "x": []},
    }
    conv_sh = False
    conv_jy = False

    print(f"\n{'─'*70}")
    print(f"{'Iter':>6}  "
          f"{'chi2_Sh':>12}  {'S_Sh':>8}  "
          f"{'chi2_Jy':>12}  {'S_Jy':>8}")
    print(f"{'─'*70}")

    for iteration in range(n_iterations):

        # ── Shannon step ───────────────────────────────────────
        if not conv_sh:
            chi2_sh, D_chi2_sh, _, _ = compute_gradients_full(
                alpha_sh, K, G_data, sigma_G
            )
            e_sh, S_sh = step_shannon(alpha_sh, D_chi2_sh)
            x_sh = alpha_chop(alpha_sh, e_sh, K, G_data, sigma_G,
                              chi2_sh, chi2_target)
            x_sh = p_chop(alpha_sh, e_sh, x_sh)
            a_new = alpha_sh + x_sh * e_sh
            a_new = np.where(a_new <= 0, 1e-4/n_comp, a_new)
            alpha_sh = a_new
            hist["shannon"]["chi2"].append(float(chi2_sh))
            hist["shannon"]["S"].append(float(S_sh))
            hist["shannon"]["x"].append(float(x_sh))

        # ── Jaynes step ────────────────────────────────────────
        if not conv_jy:
            chi2_jy, D_chi2_jy, _, _ = compute_gradients_full(
                alpha_jy, K, G_data, sigma_G
            )
            e_jy, S_jy = step_jaynes(alpha_jy, D_chi2_jy, m)
            x_jy = alpha_chop(alpha_jy, e_jy, K, G_data, sigma_G,
                              chi2_jy, chi2_target)
            x_jy = p_chop(alpha_jy, e_jy, x_jy)
            a_new = alpha_jy + x_jy * e_jy
            a_new = np.where(a_new <= 0, 1e-4/n_comp, a_new)
            alpha_jy = a_new
            hist["jaynes"]["chi2"].append(float(chi2_jy))
            hist["jaynes"]["S"].append(float(S_jy))
            hist["jaynes"]["x"].append(float(x_jy))

        if verbose and iteration % check_every == 0:
            c_sh = hist["shannon"]["chi2"][-1] if hist["shannon"]["chi2"] else 0
            s_sh = hist["shannon"]["S"][-1]    if hist["shannon"]["S"]    else 0
            c_jy = hist["jaynes"]["chi2"][-1]  if hist["jaynes"]["chi2"]  else 0
            s_jy = hist["jaynes"]["S"][-1]     if hist["jaynes"]["S"]     else 0
            print(f"  {iteration:6d}  "
                  f"{c_sh:12.5f}  {s_sh:8.5f}  "
                  f"{c_jy:12.5f}  {s_jy:8.5f}")

        for key, conv_flag in [("shannon", conv_sh), ("jaynes", conv_jy)]:
            if (not conv_flag
                    and iteration >= 2*stop_window
                    and iteration % check_every == 0):
                h   = hist[key]["chi2"]
                w1  = h[iteration - 2*stop_window:iteration - stop_window]
                w2  = h[iteration - stop_window:iteration]
                rel = abs(sum(w1) - sum(w2)) / (abs(sum(w1)) + 1e-300)
                if rel < stop_criterion:
                    if key == "shannon":
                        conv_sh = True
                    else:
                        conv_jy = True
                    print(f"\n  ✓ {key} converged at iter {iteration}"
                          f"  (Δchi2/chi2={rel:.2e})")

        if conv_sh and conv_jy:
            break

    def finalise(alpha):
        G_fit, _ = forward_normalised(alpha, K, G_data)
        wr       = (G_fit - G_data) / sigma_G
        a_n      = np.maximum(alpha / np.sum(alpha), 1e-300)
        chi2_f   = chi2_only(alpha, K, G_data, sigma_G)
        pk       = int(np.argmax(alpha))
        return {
            "G_fit": G_fit, "weighted_r": wr,
            "chi2": chi2_f,
            "peak_tau_D": float(tau_D[pk]),
            "mean_tau_D": float(np.sum(tau_D*alpha)/np.sum(alpha)),
            "alpha": alpha,
        }

    return finalise(alpha_sh), finalise(alpha_jy), hist, tau_D


# ════════════════════════════════════════════════════════════════
# Verify both step directions reduce chi2
# ════════════════════════════════════════════════════════════════

def verify_steps(tau, G_data, sigma_G, m,
                 psf_aspect_ratio=5.0):
    print(f"\n{'═'*65}")
    print(f"STEP DIRECTION VERIFICATION — Shannon vs Jaynes")
    print(f"{'═'*65}")

    tau_D = np.logspace(-7, -1, 200)
    K     = build_kernel(tau, tau_D, psf_aspect_ratio)
    alpha = np.full(200, 1.0 / 200)

    chi2, D_chi2, G_fit, scale = compute_gradients_full(
        alpha, K, G_data, sigma_G
    )
    e_sh, S_sh = step_shannon(alpha, D_chi2)
    e_jy, S_jy = step_jaynes(alpha, D_chi2, m)

    print(f"\n  chi2 (flat init) : {chi2:.4f}")
    print(f"  S_Shannon        : {S_sh:.6f}  (max=ln(200)={np.log(200):.6f})")
    print(f"  S_Jaynes         : {S_jy:.6f}  "
          f"(max=0 when p=m, min=-ln(200)={-np.log(200):.6f})")
    print(f"\n  |e_Shannon|      : {np.linalg.norm(e_sh):.4e}")
    print(f"  |e_Jaynes|       : {np.linalg.norm(e_jy):.4e}")

    print(f"\n  {'x':>8}  {'chi2_Shannon':>14}  {'chi2_Jaynes':>14}")
    all_ok_sh = True
    all_ok_jy = True
    for x_t in [1.0, 0.5, 0.1, 0.01, 1e-3, 1e-4, 2e-4]:
        a_sh = np.maximum(alpha + x_t * e_sh, 0.0)
        a_jy = np.maximum(alpha + x_t * e_jy, 0.0)
        c_sh = chi2_only(a_sh, K, G_data, sigma_G)
        c_jy = chi2_only(a_jy, K, G_data, sigma_G)
        ok_sh = "✓" if c_sh < chi2 else "✗"
        ok_jy = "✓" if c_jy < chi2 else "✗"
        if c_sh >= chi2: all_ok_sh = False
        if c_jy >= chi2: all_ok_jy = False
        print(f"  {x_t:8.1e}  {c_sh:12.4f}{ok_sh}  {c_jy:12.4f}{ok_jy}")

    print(f"\n  Shannon step valid: {'✓ YES' if all_ok_sh else '✗ NO'}")
    print(f"  Jaynes step valid:  {'✓ YES' if all_ok_jy else '✗ NO'}")
    return all_ok_sh, all_ok_jy


# ════════════════════════════════════════════════════════════════
# Analyse how prior width affects the result
# ════════════════════════════════════════════════════════════════

def scan_prior_widths(tau, G_data, sigma_G,
                      tau_D_fit, tau_D_grid,
                      psf_aspect_ratio=5.0,
                      widths=(0.25, 0.5, 1.0, 2.0),
                      n_iterations=5000):
    """
    Run Jaynes MEMFCS for several prior widths and compare results.
    This shows how much the prior influences the recovered distribution.
    """
    print(f"\n{'═'*65}")
    print(f"PRIOR WIDTH SENSITIVITY ANALYSIS")
    print(f"{'═'*65}")
    print(f"  Single-component τ_D_fit = {tau_D_fit:.4e} s")
    print(f"  Testing widths (in log10 decades): {widths}")

    tau     = np.asarray(tau,     dtype=float)
    G_data  = np.asarray(G_data,  dtype=float)
    sigma_G = np.maximum(np.asarray(sigma_G, dtype=float), 1e-20)

    n_comp  = len(tau_D_grid)
    K       = build_kernel(tau, tau_D_grid, psf_aspect_ratio)

    results = {}
    for w in widths:
        m     = make_prior_from_fit(tau_D_grid, tau_D_fit, width_decades=w)
        alpha = np.full(n_comp, 1.0 / n_comp)

        for iteration in range(n_iterations):
            chi2, D_chi2, _, _ = compute_gradients_full(
                alpha, K, G_data, sigma_G
            )
            e_G, _ = step_jaynes(alpha, D_chi2, m)
            x      = alpha_chop(alpha, e_G, K, G_data, sigma_G,
                                chi2, 1.0)
            x      = p_chop(alpha, e_G, x)
            a_new  = alpha + x * e_G
            a_new  = np.where(a_new <= 0, 1e-4/n_comp, a_new)
            alpha  = a_new

        G_fit, _ = forward_normalised(alpha, K, G_data)
        chi2_f   = chi2_only(alpha, K, G_data, sigma_G)
        pk       = int(np.argmax(alpha))
        results[w] = {
            "alpha": alpha.copy(),
            "chi2":  chi2_f,
            "peak_tau_D": float(tau_D_grid[pk]),
            "prior": m,
        }

        PSF_radius = 0.25
        pk_D = PSF_radius**2 / (4 * tau_D_grid[pk])
        print(f"  width={w:.2f} decades:  "
              f"chi2={chi2_f:.4f}  "
              f"peak τ_D={tau_D_grid[pk]:.3e} s  "
              f"peak D={pk_D:.2f} µm²/s")

    return results


# ════════════════════════════════════════════════════════════════
# Load data
# ════════════════════════════════════════════════════════════════

csv_path = sys.argv[1] if len(sys.argv) > 1 else "your_file.csv"
data     = pd.read_csv(csv_path, header=None)
tau_raw  = data.iloc[:, 0].to_numpy(dtype=float)
G_raw    = data.iloc[:, 1].to_numpy(dtype=float)

try:
    sigma_raw = data.iloc[:, 3].to_numpy(dtype=float)
    bad = ~np.isfinite(sigma_raw) | (sigma_raw <= 0)
    if np.any(bad):
        sigma_raw[bad] = max(float(np.nanstd(G_raw)), 1e-6)
except Exception:
    sigma_raw = np.full_like(G_raw, max(float(np.nanstd(G_raw)), 1e-6))

mask    = (tau_raw >= 1e-6) & (tau_raw <= 1.0)
tau     = tau_raw[mask]
G_data  = G_raw[mask]
sigma_G = sigma_raw[mask]

psf_aspect_ratio = 5.0
PSF_radius       = 0.25  # µm

print(f"\n{'═'*65}")
print(f"DATA:  n={len(tau)}  G[0]={G_data[0]:.6f}")
print(f"sigma: [{sigma_G.min():.4e}, {sigma_G.max():.4e}]")
print(f"{'═'*65}")


# ════════════════════════════════════════════════════════════════
# Step 1 — fit single component to get the prior
# ════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"STEP 1 — SINGLE COMPONENT 3D DIFFUSION FIT (prior)")
print(f"{'═'*65}")

tau_D_fit, G0_fit, fit_ok, G_pred_fit, chi2_single = (
    fit_single_component(tau, G_data, sigma_G, psf_aspect_ratio)
)

D_fit = PSF_radius**2 / (4.0 * tau_D_fit)
print(f"\n  Fit result:")
print(f"    tau_D_fit = {tau_D_fit:.4e} s")
print(f"    D_fit     = {D_fit:.4f} µm²/s")
print(f"    G0_fit    = {G0_fit:.6f}")
print(f"    chi2      = {chi2_single:.4f}  "
      f"({'good' if chi2_single < 5 else 'poor — heterogeneous sample'})")

if not fit_ok:
    print("  WARNING: fit failed — using tau_D = 1e-4 s as fallback")


# ════════════════════════════════════════════════════════════════
# Step 2 — build the prior distribution
# ════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"STEP 2 — PRIOR DISTRIBUTION FROM FIT")
print(f"{'═'*65}")

tau_D_grid = np.logspace(-7, -1, 200)
K_full     = build_kernel(tau, tau_D_grid, psf_aspect_ratio)

width_decades = 0.5    # half a decade — fairly tight prior
m = make_prior_from_fit(tau_D_grid, tau_D_fit, width_decades=width_decades)

print(f"\n  Prior: log-normal centred at tau_D_fit={tau_D_fit:.3e} s")
print(f"  Width: {width_decades} decades in log10 space")
print(f"  Prior peak index: {int(np.argmax(m))}")
print(f"  Prior sum: {m.sum():.6f}  (should be 1.0)")
print(f"  Prior max: {m.max():.4e}")
print(f"  Prior min: {m.min():.4e}")
print(f"  Prior at tau_D_fit: {m[int(np.argmin(np.abs(tau_D_grid - tau_D_fit)))]:.4e}")

# compare Shannon and Jaynes at flat init
alpha_flat = np.full(200, 1.0/200)
a_n        = alpha_flat / np.sum(alpha_flat)
S_sh_flat  = float(-np.sum(a_n * np.log(np.maximum(a_n, 1e-300))))
S_jy_flat, _ = entropy_jaynes(alpha_flat, m)

print(f"\n  At flat initialisation:")
print(f"    S_Shannon = {S_sh_flat:.6f}  (max = ln(200) = {np.log(200):.6f})")
print(f"    S_Jaynes  = {S_jy_flat:.6f}  "
      f"(max = 0, achieved when p = prior)")
print(f"    Flat init is {'close to' if abs(S_jy_flat) < 0.5 else 'far from'}"
      f" the prior (S_Jaynes = {S_jy_flat:.4f})")


# ════════════════════════════════════════════════════════════════
# Step 3 — verify step directions
# ════════════════════════════════════════════════════════════════

ok_sh, ok_jy = verify_steps(tau, G_data, sigma_G, m, psf_aspect_ratio)


# ════════════════════════════════════════════════════════════════
# Step 4 — run both methods
# ════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"STEP 4 — RUNNING Shannon vs Jaynes MEMFCS")
print(f"{'═'*65}")

res_sh, res_jy, hist, tau_D = run_both(
    tau, G_data, sigma_G, m, tau_D_fit,
    psf_aspect_ratio = psf_aspect_ratio,
    tau_D_log_range  = (-7.0, -1.0),
    n_components     = 200,
    n_iterations     = 20000,
    chi2_target      = 1.0,
    stop_criterion   = 5e-6,
    stop_window      = 100,
    check_every      = 200,
    verbose          = True,
)

D_dist = PSF_radius**2 / (4.0 * tau_D)

def report(name, res):
    pk_D = PSF_radius**2 / (4*res["peak_tau_D"])
    mn_D = PSF_radius**2 / (4*res["mean_tau_D"])
    wr   = res["weighted_r"]
    print(f"\n  [{name}]")
    print(f"    Final chi2  : {res['chi2']:.6f}")
    print(f"    G_fit[0]    : {res['G_fit'][0]:.6f}"
          f"  vs  G_data[0]={G_data[0]:.6f}")
    print(f"    peak tau_D  : {res['peak_tau_D']:.4e} s")
    print(f"    mean tau_D  : {res['mean_tau_D']:.4e} s")
    print(f"    peak D      : {pk_D:.4f} µm²/s")
    print(f"    mean D      : {mn_D:.4f} µm²/s")
    print(f"    max |wr|    : {np.max(np.abs(wr[np.isfinite(wr)])):.2f}")

print(f"\n{'═'*65}")
print(f"FINAL COMPARISON")
print(f"{'═'*65}")
print(f"  Single-component fit: D = {D_fit:.4f} µm²/s  "
      f"chi2 = {chi2_single:.4f}")
report("Shannon (flat prior)", res_sh)
report(f"Jaynes (3D fit prior, width={width_decades})", res_jy)


# ════════════════════════════════════════════════════════════════
# Step 5 — prior width sensitivity
# ════════════════════════════════════════════════════════════════

width_results = scan_prior_widths(
    tau, G_data, sigma_G,
    tau_D_fit, tau_D_grid,
    psf_aspect_ratio = psf_aspect_ratio,
    widths           = (0.25, 0.5, 1.0, 2.0),
    n_iterations     = 5000,
)


# ════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 4, figsize=(20, 13))
fig.suptitle(
    f"MEMFCS — Shannon (flat prior) vs Shannon-Jaynes (3D fit prior)\n"
    f"{csv_path.split('/')[-1]}  "
    f"Single-comp fit: D={D_fit:.1f} µm²/s  chi2={chi2_single:.2f}",
    fontsize=11
)

# ── single component fit ──────────────────────────────────────
ax = axes[0, 0]
ax.semilogx(tau, G_data,    'r',  lw=1.5, label='G observed')
ax.semilogx(tau, G_pred_fit,'k--',lw=2,   label=f'1-comp fit D={D_fit:.1f}')
ax.fill_between(tau, G_data-sigma_G, G_data+sigma_G,
                alpha=0.15, color='red')
ax.set_xlabel('τ (s)'); ax.set_ylabel('G(τ)')
ax.set_title(f'Single-component fit\nchi2={chi2_single:.3f}')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── correlation curves ────────────────────────────────────────
ax = axes[0, 1]
ax.semilogx(tau, G_data,        'r',  lw=1.5, label='G observed')
ax.semilogx(tau, res_sh["G_fit"],'g--',lw=2,   label='Shannon fit')
ax.semilogx(tau, res_jy["G_fit"],'b--',lw=2,   label='Jaynes fit')
ax.fill_between(tau, G_data-sigma_G, G_data+sigma_G,
                alpha=0.15, color='red')
ax.set_xlabel('τ (s)'); ax.set_ylabel('G(τ)')
ax.set_title('Correlation curves')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── residuals ─────────────────────────────────────────────────
ax = axes[0, 2]
ax.semilogx(tau, res_sh["weighted_r"], 'g', lw=1, label='Shannon')
ax.semilogx(tau, res_jy["weighted_r"], 'b', lw=1, label='Jaynes', alpha=0.7)
ax.axhline( 0, color='k', lw=0.8)
ax.axhline( 3, color='r', lw=0.8, ls='--', label='±3σ')
ax.axhline(-3, color='r', lw=0.8, ls='--')
ax.set_xlabel('τ (s)')
ax.set_title('Weighted residuals')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── prior distribution ────────────────────────────────────────
ax = axes[0, 3]
ax.semilogx(tau_D_grid, m / m.max(), 'orange', lw=2,
            label=f'Prior (width={width_decades}d)')
ax.axvline(tau_D_fit, color='orange', ls='--', lw=1.5,
           label=f'τ_D_fit={tau_D_fit:.3e} s')
ax.set_xlabel('τ_D (s)'); ax.set_ylabel('Normalised prior m')
ax.set_title('Prior distribution (invariant measure)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── D distributions ───────────────────────────────────────────
ax = axes[1, 0]
ax.semilogx(D_dist, res_sh["alpha"], 'g', lw=2, label='Shannon')
ax.axvline(PSF_radius**2/(4*res_sh["peak_tau_D"]),
           color='g', ls='--',
           label=f'peak={PSF_radius**2/(4*res_sh["peak_tau_D"]):.1f} µm²/s')
ax.axvline(D_fit, color='k', ls=':', lw=1.5,
           label=f'1-comp={D_fit:.1f} µm²/s')
ax.set_xlabel('D (µm²/s)'); ax.set_ylabel('α')
ax.set_title('Shannon — D distribution')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.semilogx(D_dist, res_jy["alpha"], 'b', lw=2, label='Jaynes')
ax.axvline(PSF_radius**2/(4*res_jy["peak_tau_D"]),
           color='b', ls='--',
           label=f'peak={PSF_radius**2/(4*res_jy["peak_tau_D"]):.1f} µm²/s')
ax.axvline(D_fit, color='k', ls=':', lw=1.5,
           label=f'1-comp={D_fit:.1f} µm²/s')
ax.set_xlabel('D (µm²/s)'); ax.set_ylabel('α')
ax.set_title('Jaynes — D distribution')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── chi2 convergence ──────────────────────────────────────────
ax = axes[1, 2]
ax.semilogy(hist["shannon"]["chi2"], 'g', lw=1, label='Shannon', alpha=0.8)
ax.semilogy(hist["jaynes"]["chi2"],  'b', lw=1, label='Jaynes',  alpha=0.8)
ax.axhline(1.0, color='r', ls='--', label='target=1')
ax.set_xlabel('Iteration'); ax.set_ylabel('chi2')
ax.set_title('chi2 convergence')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── entropy history ───────────────────────────────────────────
ax = axes[1, 3]
ax.plot(hist["shannon"]["S"], 'g', lw=1, label='S_Shannon', alpha=0.8)
ax.plot(hist["jaynes"]["S"],  'b', lw=1, label='S_Jaynes',  alpha=0.8)
ax.axhline(np.log(200), color='g', ls='--', lw=0.8,
           label=f'Shannon max={np.log(200):.3f}')
ax.axhline(0, color='b', ls='--', lw=0.8, label='Jaynes max=0 (p=m)')
ax.set_xlabel('Iteration'); ax.set_ylabel('S')
ax.set_title('Entropy histories\n(different scales — different definitions)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── prior width sensitivity ───────────────────────────────────
ax = axes[2, 0]
colors_w = ['navy', 'steelblue', 'seagreen', 'orange']
for (w, res_w), col in zip(width_results.items(), colors_w):
    ax.semilogx(D_dist, res_w["alpha"],
                color=col, lw=2, label=f'width={w}d')
ax.axvline(D_fit, color='k', ls=':', lw=1.5,
           label=f'1-comp={D_fit:.1f} µm²/s')
ax.set_xlabel('D (µm²/s)'); ax.set_ylabel('α')
ax.set_title('Prior width sensitivity\n(Jaynes, 5000 iters)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2, 1]
for (w, res_w), col in zip(width_results.items(), colors_w):
    D_dist_w = PSF_radius**2 / (4.0 * tau_D_grid)
    ax.semilogx(D_dist_w, res_w["prior"] / res_w["prior"].max(),
                color=col, lw=1.5, ls='--', label=f'prior w={w}d',
                alpha=0.7)
ax.axvline(D_fit, color='k', ls=':', lw=1.5)
ax.set_xlabel('D (µm²/s)'); ax.set_ylabel('Normalised prior')
ax.set_title('Prior shapes for each width')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── residual histograms ───────────────────────────────────────
ax = axes[2, 2]
wr_sh = res_sh["weighted_r"][np.isfinite(res_sh["weighted_r"])]
ax.hist(wr_sh, bins=40, density=True, color='green', alpha=0.6,
        label='Shannon')
xs = np.linspace(-5, 5, 200)
ax.plot(xs, np.exp(-0.5*xs**2)/np.sqrt(2*np.pi),
        'r--', lw=1.5, label='N(0,1)')
ax.set_xlim(-6, 6)
ax.set_xlabel('weighted residual'); ax.set_ylabel('density')
ax.set_title('Shannon residual distribution')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2, 3]
wr_jy = res_jy["weighted_r"][np.isfinite(res_jy["weighted_r"])]
ax.hist(wr_jy, bins=40, density=True, color='blue', alpha=0.6,
        label='Jaynes')
ax.plot(xs, np.exp(-0.5*xs**2)/np.sqrt(2*np.pi),
        'r--', lw=1.5, label='N(0,1)')
ax.set_xlim(-6, 6)
ax.set_xlabel('weighted residual'); ax.set_ylabel('density')
ax.set_title('Jaynes residual distribution')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.tight_layout()
out = csv_path.replace('.csv', '_memfcs_jaynes.svg')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nPlot saved: {out}")

print(f"\n{'═'*65}")
print(f"INTERPRETATION GUIDE")
print(f"{'═'*65}")
print(f"""
Single-component fit chi2 = {chi2_single:.3f}

If chi2_single < 2:
  The data is consistent with a single species. Both Shannon and
  Jaynes should give a single peak at the same D. Jaynes will give
  a sharper peak (prior reinforces the single-species interpretation).
  The two methods agreeing is strong evidence for homogeneity.

If chi2_single >> 2:
  The data requires more than one species. Shannon MEMFCS should
  reveal the heterogeneity. Jaynes MEMFCS will only deviate from
  the single-species prior as much as the data demands — secondary
  peaks will be suppressed unless they are strongly required by data.
  This makes Jaynes more conservative but less prone to spurious peaks.

Prior width sensitivity:
  - Narrow prior (0.25d): very conservative, strong single-species bias
  - Width 0.5d: moderate, reveals heterogeneity if chi2 > 2
  - Width 1.0d: closer to flat Shannon prior
  - Width 2.0d: nearly equivalent to flat Shannon prior

If all prior widths give the same peak: result is robust
If results vary strongly with width: data cannot resolve the distribution
""")