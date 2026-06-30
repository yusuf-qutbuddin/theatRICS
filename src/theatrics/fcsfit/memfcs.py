"""
theatrics/fcsfit/memfcs.py

MEMFCS — Maximum Entropy Method for FCS data analysis.

Based on Sengupta et al. (2003) Biophys J 84:1977-1984.

Algorithm
─────────
1. Flat initialisation: all alpha_i = 1/n_components
2. G(0) normalisation at every iteration — pins the fit amplitude
   to mean(G_data[:10]), preventing divergence
3. Element-wise step direction balancing entropy and chi2 gradients
4. Alpha-chop bisection to land on chi2_target (Skilling & Bryan 1984)
5. P-chop positivity enforcement
6. Stopping when chi2 stops changing between consecutive windows
"""
from __future__ import annotations
import numpy as np


# ────────────────────────────────────────────────────────────────
# Kernel  (Eq. 7 of Sengupta et al. 2003)
# ────────────────────────────────────────────────────────────────

def build_kernel(tau: np.ndarray,
                 tau_D: np.ndarray,
                 psf_aspect_ratio: float) -> np.ndarray:
    """
    K[i,j] = 1 / ((1 + τ_i/τ_D_j) · sqrt(1 + τ_i/(S²·τ_D_j)))

    Shape: (n_tau, n_tau_D)
    S = psf_aspect_ratio
    """
    S2    = psf_aspect_ratio ** 2
    ratio = tau[:, None] / tau_D[None, :]
    return 1.0 / ((1.0 + ratio) * np.sqrt(1.0 + ratio / S2))


# ────────────────────────────────────────────────────────────────
# Forward model with G(0) normalisation
# ────────────────────────────────────────────────────────────────

def _forward(alpha: np.ndarray,
             K: np.ndarray,
             G_data: np.ndarray) -> tuple[np.ndarray, float]:
    """
    G_raw   = K @ alpha
    scale   = mean(G_data[:10]) / G_raw[0]
    G_fit   = G_raw * scale

    Pinning G_fit[0] to the data amplitude prevents the
    amplitude from diverging during iteration.
    """
    G_raw  = K @ alpha
    target = float(np.mean(G_data[:10]))
    scale  = target / max(float(G_raw[0]), 1e-300)
    return G_raw * scale, scale


def _chi2_only(alpha: np.ndarray,
               K: np.ndarray,
               G_data: np.ndarray,
               sigma_G: np.ndarray) -> float:
    """Cheap chi2 evaluation for line search — no gradients."""
    G_fit, _ = _forward(alpha, K, G_data)
    r        = G_fit - G_data
    return float(np.sum((r / sigma_G) ** 2) / len(G_data))


# ────────────────────────────────────────────────────────────────
# Step direction
# ────────────────────────────────────────────────────────────────

def _step_direction(
        alpha: np.ndarray,
        K: np.ndarray,
        G_data: np.ndarray,
        sigma_G: np.ndarray,
) -> tuple[np.ndarray, float, float, np.ndarray, float]:
    """
    Compute the search direction e_G and current state.

    Step formula
    ────────────
    D_chi2   = (2·scale/M) · K^T · (w · r)      gradient of chi2
    D_S      = -1 - ln(p)                        gradient of entropy
    alpha_f  = |D_chi2| / (20 · |D_S|)           element-wise balance
    e_G      = a_norm · (alpha_f · D_S - D_chi2/2)

    The element-wise multiplication by a_norm ensures:
    - small components take small steps (prevents blow-up)
    - the direction always reduces chi2 (verified empirically)

    Returns
    ───────
    e_G   : (n_comp,) search direction
    chi2  : current reduced chi2
    S     : current Shannon entropy
    G_fit : current normalised fit
    scale : current amplitude scale factor
    """
    M            = len(G_data)
    w            = 1.0 / sigma_G ** 2
    G_fit, scale = _forward(alpha, K, G_data)
    r            = G_fit - G_data
    chi2         = float(np.sum((r / sigma_G) ** 2) / M)

    # normalised amplitudes for entropy
    a_norm = np.maximum(alpha / (np.sum(alpha) + 1e-300), 1e-300)
    log_a  = np.log(a_norm)
    S      = float(-np.sum(a_norm * log_a))

    # gradients
    D_chi2  = (2.0 * scale / M) * (K.T @ (w * r))     # (n_comp,)
    D_S     = -1.0 - log_a                             # (n_comp,)

    # element-wise balance factor
    alpha_f = np.abs(D_chi2) / (20.0 * np.abs(D_S) + 1e-300)

    # combined search direction
    e_G = a_norm * (alpha_f * D_S - D_chi2 / 2.0)

    return e_G, chi2, S, G_fit, scale


# ────────────────────────────────────────────────────────────────
# Alpha-chop line search  (Skilling & Bryan 1984)
# ────────────────────────────────────────────────────────────────

def _alpha_chop(
        alpha: np.ndarray,
        delta: np.ndarray,
        K: np.ndarray,
        G_data: np.ndarray,
        sigma_G: np.ndarray,
        chi2_current: float,
        chi2_target: float,
        max_chops: int = 60,
) -> float:
    """
    Find x in (0, 1] such that chi2(alpha + x·delta) ≈ chi2_target.

    When chi2_current > chi2_target:
        - full step (x=1) goes below target → bisect to land on target
        - full step stays above target → take it and keep iterating

    When chi2_current ≤ chi2_target:
        - already at/below target → take smallest x that keeps chi2 ≥ target
    """
    a1     = np.maximum(alpha + delta, 0.0)
    chi2_1 = _chi2_only(a1, K, G_data, sigma_G)

    if chi2_current > chi2_target:
        if chi2_1 <= chi2_target:
            # bisect to land exactly on target
            x_lo, x_hi = 0.0, 1.0
            for _ in range(max_chops):
                x_m    = 0.5 * (x_lo + x_hi)
                a_m    = np.maximum(alpha + x_m * delta, 0.0)
                chi2_m = _chi2_only(a_m, K, G_data, sigma_G)
                if chi2_m > chi2_target:
                    x_lo = x_m
                else:
                    x_hi = x_m
                if abs(x_hi - x_lo) < 1e-14:
                    break
            return 0.5 * (x_lo + x_hi)
        else:
            return 1.0   # still above target — take full step
    else:
        # below target — find smallest x that keeps chi2 ≥ target
        x = 1.0
        for _ in range(max_chops):
            a_t = np.maximum(alpha + x * delta, 0.0)
            c_t = _chi2_only(a_t, K, G_data, sigma_G)
            if c_t >= chi2_target:
                break
            x *= 0.5
        return x


def _p_chop(
        alpha: np.ndarray,
        delta: np.ndarray,
        x: float,
        max_chops: int = 60,
) -> float:
    """
    Halve x until alpha + x·delta has no negative components.
    Paper: 'Care is taken to avoid negative value for αi
    by using only a fraction of x.'
    """
    for _ in range(max_chops):
        if np.all(alpha + x * delta >= 0.0):
            break
        x *= 0.5
    return x

# ────────────────────────────────────────────────────────────────
# Single-component 3D diffusion fit  (produces the Jaynes prior)
# ────────────────────────────────────────────────────────────────

def fit_single_component(
        tau:              np.ndarray,
        G_data:           np.ndarray,
        sigma_G:          np.ndarray,
        psf_aspect_ratio: float = 5.0,
) -> tuple[float, float, bool, np.ndarray, float]:
    """
    Fit G(τ) = G0 / ((1 + τ/τ_D) · sqrt(1 + τ/(S²·τ_D)))
    with G0 fixed to mean(G_data[:10]) to match the MEMFCS
    normalisation convention.

    Returns
    -------
    tau_D_fit : best-fit diffusion time (s)
    G0_fit    : amplitude (= mean(G_data[:10]))
    success   : bool
    G_pred    : (n_tau,) predicted curve
    chi2      : reduced chi2 of the single-component fit
    """
    from scipy.optimize import curve_fit as _curve_fit

    S2      = psf_aspect_ratio ** 2
    G0      = float(np.mean(G_data[:10]))

    def _model(t, tau_D):
        r = t / tau_D
        return G0 / ((1.0 + r) * np.sqrt(1.0 + r / S2))

    try:
        half     = G0 / 2.0
        idx_half = int(np.argmin(np.abs(G_data - half)))
        tau_D_0  = max(float(tau[idx_half]), 1e-7)

        popt, _ = _curve_fit(
            _model, tau, G_data,
            p0=[tau_D_0],
            bounds=(1e-8, 1.0),
            sigma=sigma_G,
            absolute_sigma=True,
            maxfev=10000,
        )
        tau_D_fit = float(popt[0])
        G_pred    = _model(tau, tau_D_fit)
        chi2_sc   = float(
            np.sum(((G_pred - G_data) / sigma_G) ** 2) / len(tau)
        )
        return tau_D_fit, G0, True, G_pred, chi2_sc

    except Exception:
        return 1e-4, G0, False, np.zeros_like(tau), np.inf


# ────────────────────────────────────────────────────────────────
# Prior distribution from single-component fit
# ────────────────────────────────────────────────────────────────

def make_jaynes_prior(
        tau_D_grid:     np.ndarray,
        tau_D_fit:      float,
        width_decades:  float = 0.5,
) -> np.ndarray:
    """
    Log-normal prior centred at tau_D_fit with width in log10 decades.

    m_i ∝ exp(-0.5 · ((log10(τ_D_i) - log10(τ_D_fit)) / width)²)

    Normalised to sum to 1 with a small floor to prevent ln(0).

    width_decades = 0.5  → prior spans ~factor of 3 around peak
    width_decades = 1.0  → prior spans ~one decade
    width_decades = 2.0  → nearly flat, close to Shannon prior
    """
    log_grid = np.log10(tau_D_grid)
    log_peak = np.log10(max(tau_D_fit, 1e-10))
    m        = np.exp(
        -0.5 * ((log_grid - log_peak) / width_decades) ** 2
    )
    m = m / np.sum(m)
    m = np.maximum(m, 1e-10)
    return m


# ────────────────────────────────────────────────────────────────
# Shannon-Jaynes relative entropy
# ────────────────────────────────────────────────────────────────

def _entropy_jaynes(
        alpha: np.ndarray,
        m:     np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    S_SJ = -Σ p_i · ln(p_i / m_i)     p_i = α_i / Σα_i

    Maximum is 0, achieved when p = m.
    Reduces to standard Shannon when m is flat (m_i = 1/n).

    Gradient:
        ∂S_SJ/∂α_i = -(ln(p_i/m_i) - S_SJ) / Z
    """
    a         = np.maximum(alpha, 1e-300)
    Z         = float(np.sum(a))
    p         = a / Z
    p         = np.maximum(p, 1e-300)
    m_s       = np.maximum(m, 1e-300)
    log_ratio = np.log(p / m_s)
    S_SJ      = float(-np.sum(p * log_ratio))
    grad      = -(log_ratio - S_SJ) / Z
    return S_SJ, grad


# ────────────────────────────────────────────────────────────────
# Jaynes MEMFCS run — identical structure to run_memfcs
# ────────────────────────────────────────────────────────────────

def run_memfcs_jaynes(
        tau:              np.ndarray,
        G_data:           np.ndarray,
        sigma_G:          np.ndarray,
        m:                np.ndarray,
        psf_aspect_ratio: float = 5.0,
        tau_D_log_range:  tuple[float, float] = (-7.0, -1.0),
        n_components:     int   = 200,
        n_iterations:     int   = 20000,
        chi2_target:      float = 1.0,
        stop_criterion:   float = 5e-6,
        stop_window:      int   = 100,
        check_every:      int   = 200,
        verbose:          bool  = False,
) -> dict:
    """
    MEMFCS with Shannon-Jaynes relative entropy.

    Identical algorithm to run_memfcs but uses the relative entropy
    gradient instead of the standard Shannon gradient.

    Parameters
    ──────────
    m   : (n_components,) prior distribution from make_jaynes_prior()
    all other parameters identical to run_memfcs()
    """
    tau     = np.asarray(tau,     dtype=float)
    G_data  = np.asarray(G_data,  dtype=float)
    sigma_G = np.maximum(np.asarray(sigma_G, dtype=float), 1e-20)

    n_comp = n_components
    tau_D  = np.logspace(tau_D_log_range[0], tau_D_log_range[1], n_comp)
    K      = build_kernel(tau, tau_D, psf_aspect_ratio)

    # ensure prior matches the grid length
    if len(m) != n_comp:
        raise ValueError(
            f"Prior length {len(m)} does not match "
            f"n_components={n_comp}."
        )

    alpha = np.full(n_comp, 1.0 / n_comp)

    chi2_history = []
    S_history    = []
    converged    = False
    iteration    = 0

    for iteration in range(n_iterations):

        # forward model + chi2 gradient (identical to run_memfcs)
        M            = len(G_data)
        w            = 1.0 / sigma_G ** 2
        G_fit, scale = _forward(alpha, K, G_data)
        r            = G_fit - G_data
        chi2         = float(np.sum((r / sigma_G) ** 2) / M)
        D_chi2       = (2.0 * scale / M) * (K.T @ (w * r))

        # Shannon-Jaynes entropy and gradient
        a_norm   = np.maximum(alpha / (np.sum(alpha) + 1e-300), 1e-300)
        S_SJ, D_S_SJ = _entropy_jaynes(a_norm, m)

        # step direction (same formula as v9, using Jaynes gradient)
        alpha_f  = np.abs(D_chi2) / (20.0 * np.abs(D_S_SJ) + 1e-300)
        e_G      = a_norm * (alpha_f * D_S_SJ - D_chi2 / 2.0)

        # alpha-chop and p-chop (identical to run_memfcs)
        x = _alpha_chop(alpha, e_G, K, G_data, sigma_G,
                        chi2, chi2_target)
        x = _p_chop(alpha, e_G, x)

        alpha_new = alpha + x * e_G
        alpha_new = np.where(alpha_new <= 0.0, 1e-4 / n_comp, alpha_new)
        alpha     = alpha_new

        chi2_history.append(float(chi2))
        S_history.append(float(S_SJ))

        if verbose and iteration % check_every == 0:
            print(f"  [MEMFCS-Jaynes] iter={iteration:6d}  "
                  f"chi2={chi2:.5f}  S_SJ={S_SJ:.5f}  x={x:.3e}")

        if (iteration % check_every == 0
                and iteration >= 2 * stop_window):
            w1  = chi2_history[
                iteration - 2*stop_window:iteration - stop_window
            ]
            w2  = chi2_history[
                iteration - stop_window:iteration
            ]
            rel = abs(sum(w1) - sum(w2)) / (abs(sum(w1)) + 1e-300)
            if rel < stop_criterion:
                converged = True
                if verbose:
                    print(f"  [MEMFCS-Jaynes] Converged at iter {iteration}"
                          f"  (Δchi2/chi2={rel:.2e})")
                break

    G_fit_final, scale_final = _forward(alpha, K, G_data)
    weighted_r   = (G_fit_final - G_data) / sigma_G
    a_norm_final = np.maximum(alpha / np.sum(alpha), 1e-300)
    S_final, _   = _entropy_jaynes(a_norm_final, m)
    chi2_final   = _chi2_only(alpha, K, G_data, sigma_G)
    peak_idx     = int(np.argmax(alpha))
    peak_tau_D   = float(tau_D[peak_idx])
    mean_tau_D   = float(np.sum(tau_D * alpha) / np.sum(alpha))

    return {
        "tau_D":            tau_D,
        "alpha":            alpha,
        "G_fit":            G_fit_final,
        "scale":            scale_final,
        "chi2":             chi2_final,
        "S":                S_final,
        "weighted_r":       weighted_r,
        "chi2_history":     chi2_history,
        "S_history":        S_history,
        "peak_tau_D":       peak_tau_D,
        "mean_tau_D":       mean_tau_D,
        "n_iterations_run": iteration + 1,
        "converged":        converged,
        "prior":            m,
    }
# ────────────────────────────────────────────────────────────────
# Main solver
# ────────────────────────────────────────────────────────────────

def run_memfcs(
        tau:              np.ndarray,
        G_data:           np.ndarray,
        sigma_G:          np.ndarray,
        psf_aspect_ratio: float = 5.0,
        tau_D_log_range:  tuple[float, float] = (-7.0, -1.0),
        n_components:     int   = 200,
        n_iterations:     int   = 20000,
        chi2_target:      float = 1.0,
        stop_criterion:   float = 5e-6,
        stop_window:      int   = 100,
        check_every:      int   = 200,
        verbose:          bool  = False,
) -> dict:
    """
    Run MEMFCS and return a result dictionary.

    Parameters
    ──────────
    tau              : (n_tau,) lag times in seconds
    G_data           : (n_tau,) measured correlation
    sigma_G          : (n_tau,) measurement uncertainties
    psf_aspect_ratio : S = z₀/w₀
    tau_D_log_range  : (log10 min, log10 max) τ_D search range
    n_components     : number of τ_D grid points
    n_iterations     : maximum iterations
    chi2_target      : aimed reduced chi2 (1.0 = good fit)
    stop_criterion   : convergence threshold on relative Δchi2
    stop_window      : iterations per window for stopping check
    check_every      : how often to check convergence
    verbose          : print progress

    Returns
    ───────
    dict with keys:
        tau_D            : (n_components,) τ_D grid (s)
        alpha            : (n_components,) recovered amplitudes
        G_fit            : (n_tau,) fitted correlation
        scale            : final amplitude scale factor
        chi2             : final reduced chi2
        S                : final Shannon entropy
        weighted_r       : (n_tau,) weighted residuals
        chi2_history     : list of chi2 per iteration
        S_history        : list of S per iteration
        peak_tau_D       : τ_D at maximum amplitude (s)
        mean_tau_D       : amplitude-weighted mean τ_D (s)
        n_iterations_run : iterations actually performed
        converged        : True if stopping criterion met
    """
    tau     = np.asarray(tau,     dtype=float)
    G_data  = np.asarray(G_data,  dtype=float)
    sigma_G = np.maximum(np.asarray(sigma_G, dtype=float), 1e-20)

    n_comp = n_components
    tau_D  = np.logspace(tau_D_log_range[0], tau_D_log_range[1], n_comp)
    K      = build_kernel(tau, tau_D, psf_aspect_ratio)

    # flat initialisation — paper: "equal values for all αi"
    alpha = np.full(n_comp, 1.0 / n_comp)

    chi2_history = []
    S_history    = []
    converged    = False
    iteration    = 0

    for iteration in range(n_iterations):

        # step direction and current state
        e_G, chi2, S, G_fit, scale = _step_direction(
            alpha, K, G_data, sigma_G
        )

        # alpha-chop: bisect to land on chi2_target
        x = _alpha_chop(
            alpha, e_G, K, G_data, sigma_G,
            chi2, chi2_target
        )

        # p-chop: enforce positivity
        x = _p_chop(alpha, e_G, x)

        # update
        alpha_new = alpha + x * e_G
        alpha_new = np.where(alpha_new <= 0.0, 1e-4 / n_comp, alpha_new)
        alpha     = alpha_new

        chi2_history.append(float(chi2))
        S_history.append(float(S))

        if verbose and iteration % check_every == 0:
            print(f"  [MEMFCS] iter={iteration:6d}  "
                  f"chi2={chi2:.5f}  S={S:.5f}  x={x:.3e}")

        # stopping criterion
        if (iteration % check_every == 0
                and iteration >= 2 * stop_window):
            w1  = chi2_history[
                iteration - 2*stop_window:iteration - stop_window
            ]
            w2  = chi2_history[
                iteration - stop_window:iteration
            ]
            rel = abs(sum(w1) - sum(w2)) / (abs(sum(w1)) + 1e-300)
            if rel < stop_criterion:
                converged = True
                if verbose:
                    print(f"  [MEMFCS] Converged at iter {iteration}"
                          f"  (Δchi2/chi2={rel:.2e})")
                break

    # final quantities
    G_fit_final, scale_final = _forward(alpha, K, G_data)
    weighted_r = (G_fit_final - G_data) / sigma_G

    a_norm     = np.maximum(alpha / np.sum(alpha), 1e-300)
    S_final    = float(-np.sum(a_norm * np.log(a_norm)))
    chi2_final = _chi2_only(alpha, K, G_data, sigma_G)

    peak_idx   = int(np.argmax(alpha))
    peak_tau_D = float(tau_D[peak_idx])
    mean_tau_D = float(np.sum(tau_D * alpha) / np.sum(alpha))

    return {
        "tau_D":            tau_D,
        "alpha":            alpha,
        "G_fit":            G_fit_final,
        "scale":            scale_final,
        "chi2":             chi2_final,
        "S":                S_final,
        "weighted_r":       weighted_r,
        "chi2_history":     chi2_history,
        "S_history":        S_history,
        "peak_tau_D":       peak_tau_D,
        "mean_tau_D":       mean_tau_D,
        "n_iterations_run": iteration + 1,
        "converged":        converged,
    }