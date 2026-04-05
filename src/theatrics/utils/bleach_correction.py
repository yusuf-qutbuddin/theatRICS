import numpy as np
from scipy.optimize import curve_fit

def _exp1(t, a, tb, c):
    return a * np.exp(-t / tb) + c

def _exp2(t, a1, tb1, a2, tb2, c):
    return a1 * np.exp(-t / tb1) + a2 * np.exp(-t / tb2) + c

def fit_bleach_trend(t, F, model="exp1"):
    """
    Fit f(t) to slow decay of fluorescence.
    Returns f_t (array), f0 (scalar), popt (params), model_used.
    """
    t = np.asarray(t, dtype=float)
    F = np.asarray(F, dtype=float)

    # robust initial guesses
    c0 = float(np.percentile(F, 10))
    a0 = float(np.max(F) - c0)
    tb0 = float((t[-1] - t[0]) / 3.0) if t[-1] > t[0] else 1.0

    if model == "exp1":
        p0 = [a0, tb0, c0]
        bounds = ([0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(_exp1, t, F, p0=p0, bounds=bounds, maxfev=20000)
        f_t = _exp1(t, *popt)

    elif model == "exp2":
        # second component: slower by default
        p0 = [0.7*a0, tb0/3.0, 0.3*a0, tb0, c0]
        bounds = ([0.0, 1e-12, 0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf, np.inf, np.inf])
        popt, _ = curve_fit(_exp2, t, F, p0=p0, bounds=bounds, maxfev=50000)
        f_t = _exp2(t, *popt)

    else:
        raise ValueError("model must be 'exp1' or 'exp2'")

    f0 = float(f_t[0])
    return f_t, f0, popt, model


def depletion_correct(F, f_t, f0=None, eps=1e-12):
    """
    Apply Eq.: https://doi.org/10.1016/j.bpj.2008.12.3888
      Fc = F / sqrt(f(t)/f(0)) + f(0) * (1 - sqrt(f(t)/f(0)))
    """
    F = np.asarray(F, dtype=float)
    f_t = np.asarray(f_t, dtype=float)
    if f0 is None:
        f0 = float(f_t[0])

    ratio = np.maximum(f_t / max(f0, eps), eps)
    s = np.sqrt(ratio)

    Fc = (F / s) + f0 * (1.0 - s)
    return Fc