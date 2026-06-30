"""
debug_memfcs.py

Run this from the command line to diagnose the MEMFCS fitting:
    python debug_memfcs.py path/to/your_file.csv

It will print detailed diagnostics at every step and save plots.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── load your CSV ──────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else r"/fs/pool/pool-schwille-user/Qutbuddin_Yusuf/_Protocols/RICS_fit/test_data/FCS/Measurement_R1_P0_K1_Corr_ch1ch1.csv"

data    = pd.read_csv(csv_path, header=None)
tau     = data.iloc[:, 0].to_numpy(dtype=float)
G_data  = data.iloc[:, 1].to_numpy(dtype=float)
try:
    sigma_G = data.iloc[:, 3].to_numpy(dtype=float)
    bad = ~np.isfinite(sigma_G) | (sigma_G <= 0)
    if np.any(bad):
        sigma_G[bad] = max(float(np.nanstd(G_data)), 1e-6)
except Exception:
    sigma_G = np.full_like(G_data, max(float(np.nanstd(G_data)), 1e-6))

# apply tau domain
mask   = (tau >= 1e-6) & (tau <= 1.0)
tau    = tau[mask]
G_data = G_data[mask]
sigma_G = sigma_G[mask]

print("=" * 60)
print("DATA DIAGNOSTICS")
print("=" * 60)
print(f"  n_tau points      : {len(tau)}")
print(f"  tau range         : {tau[0]:.3e} — {tau[-1]:.3e} s")
print(f"  G range           : {G_data.min():.4f} — {G_data.max():.4f}")
print(f"  G[0] (amplitude)  : {G_data[0]:.4f}")
print(f"  sigma_G range     : {sigma_G.min():.4e} — {sigma_G.max():.4e}")
print(f"  mean sigma_G      : {sigma_G.mean():.4e}")
print(f"  G[0] / sigma_G[0] : {G_data[0]/sigma_G[0]:.1f}  (SNR at first point)")
print()

# ── parameters ─────────────────────────────────────────────────
PSF_aspect_ratio = 5.0
PSF_radius_um    = 0.25
n_comp           = 200
tau_D_log_range  = (-7.0, -1.0)
chi2_target      = 1.0

# ── build kernel ───────────────────────────────────────────────
tau_D = np.logspace(tau_D_log_range[0], tau_D_log_range[1], n_comp)
S2    = PSF_aspect_ratio ** 2
ratio = tau[:, None] / tau_D[None, :]
K     = 1.0 / ((1.0 + ratio) * np.sqrt(1.0 + ratio / S2))

print("=" * 60)
print("KERNEL DIAGNOSTICS")
print("=" * 60)
print(f"  K shape           : {K.shape}")
print(f"  K min / max       : {K.min():.4f} / {K.max():.4f}")
print(f"  K[:, 0] (fastest) : {K[:3, 0]}  ...")
print(f"  K[:, -1] (slowest): {K[:3, -1]}  ...")
print()

# ── flat initialisation ────────────────────────────────────────
alpha = np.ones(n_comp) / n_comp
G_model_flat = K @ alpha

print("=" * 60)
print("FLAT INITIALISATION DIAGNOSTICS")
print("=" * 60)
print(f"  alpha sum         : {alpha.sum():.6f}  (should be 1.0)")
print(f"  K @ alpha range   : {G_model_flat.min():.6f} — {G_model_flat.max():.6f}")
print(f"  K @ alpha [0]     : {G_model_flat[0]:.6f}")
print(f"  G_data [0]        : {G_data[0]:.6f}")
print()

# ── C initialisation ───────────────────────────────────────────
C_init_v1 = float(np.mean(G_data[:5])) / max(float(np.mean(G_model_flat[:5])), 1e-300)
C_init_v2 = float(G_data[0]) / max(float(G_model_flat[0]), 1e-300)
C_init_v3 = float(np.sum(G_data * G_model_flat) / np.sum(G_model_flat ** 2))

print("=" * 60)
print("C INITIALISATION OPTIONS")
print("=" * 60)
print(f"  v1 (mean first 5) : C = {C_init_v1:.6f}")
print(f"  v2 (first point)  : C = {C_init_v2:.6f}")
print(f"  v3 (least squares): C = {C_init_v3:.6f}")
print()
print(f"  C * K@alpha[0]    (v1): {C_init_v1 * G_model_flat[0]:.6f}  vs G_data[0]={G_data[0]:.6f}")
print(f"  C * K@alpha[0]    (v2): {C_init_v2 * G_model_flat[0]:.6f}  vs G_data[0]={G_data[0]:.6f}")
print(f"  C * K@alpha[0]    (v3): {C_init_v3 * G_model_flat[0]:.6f}  vs G_data[0]={G_data[0]:.6f}")
print()

# use the least-squares C — most robust
C = max(C_init_v3, 1e-10)

# ── initial chi-squared ────────────────────────────────────────
G_fit_init  = C * G_model_flat
residual    = G_fit_init - G_data
weighted_r  = residual / sigma_G
chi2_init   = float(np.sum(weighted_r ** 2) / len(tau))

print("=" * 60)
print("INITIAL CHI-SQUARED")
print("=" * 60)
print(f"  chi2 (initial)    : {chi2_init:.4f}  (target = {chi2_target})")
print(f"  ratio to target   : {chi2_init / chi2_target:.2f}x")
print()

if chi2_init < chi2_target:
    print("  WARNING: initial chi2 is already BELOW target.")
    print("  This means sigma_G may be overestimated or the flat")
    print("  distribution already fits the data too well.")
    print("  The alpha-chop will immediately return x=1 without")
    print("  reducing chi2, and the algorithm may not converge.")
    print()

# ── entropy at flat init ───────────────────────────────────────
Z     = float(np.sum(alpha))
p     = alpha / Z
log_p = np.log(np.maximum(p, 1e-300))
S_init = float(-np.sum(p * log_p))

print("=" * 60)
print("ENTROPY AT FLAT INITIALISATION")
print("=" * 60)
print(f"  S (flat)          : {S_init:.6f}")
print(f"  ln(n_comp)        : {np.log(n_comp):.6f}  (should equal S at flat)")
print()

# ── gradient diagnostics ───────────────────────────────────────
n_tau = len(tau)
w     = 1.0 / sigma_G ** 2
wr    = w * residual

grad_chi2     = (2.0 * C / n_tau) * (K.T @ wr)
hess_diag_chi2 = (2.0 * C ** 2 / n_tau) * (K.T ** 2 @ w)
grad_S        = (S_init - log_p) / Z

print("=" * 60)
print("GRADIENT DIAGNOSTICS AT INITIALISATION")
print("=" * 60)
print(f"  |grad_chi2|       : {np.linalg.norm(grad_chi2):.4e}")
print(f"  |grad_S|          : {np.linalg.norm(grad_S):.4e}")
print(f"  grad_chi2 range   : {grad_chi2.min():.4e} — {grad_chi2.max():.4e}")
print(f"  grad_S range      : {grad_S.min():.4e} — {grad_S.max():.4e}")
print(f"  hess_diag_chi2    : {hess_diag_chi2.min():.4e} — {hess_diag_chi2.max():.4e}")
print()

# ── test one alpha-chop step ───────────────────────────────────
def _normalise(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-300 else v

e1 = _normalise(grad_S)
e2 = _normalise(grad_chi2)
hess_pos = np.maximum(hess_diag_chi2, 1e-300)
e3 = _normalise(hess_pos * e1)
E  = np.column_stack([e1, e2, e3])

gS_3   = E.T @ grad_S
gChi_3 = E.T @ grad_chi2
H3     = E.T @ (hess_pos[:, None] * E)

print("=" * 60)
print("3D SUBSPACE DIAGNOSTICS")
print("=" * 60)
print(f"  gS_3   : {gS_3}")
print(f"  gChi_3 : {gChi_3}")
print(f"  H3     :\n{H3}")
print(f"  cond(H3): {np.linalg.cond(H3):.4e}")
print()

try:
    reg = 1e-10 * max(np.max(np.abs(H3)), 1.0) * np.eye(3)
    c   = np.linalg.solve(H3 + reg, gS_3)
    delta = E @ c
    print(f"  c (subspace step): {c}")
    print(f"  |delta|          : {np.linalg.norm(delta):.4e}")
    print(f"  delta range      : {delta.min():.4e} — {delta.max():.4e}")
    print()
except Exception as ex:
    print(f"  SOLVE FAILED: {ex}")
    delta = grad_S / (np.linalg.norm(grad_S) + 1e-300)

# ── alpha-chop trace ───────────────────────────────────────────
print("=" * 60)
print("ALPHA-CHOP TRACE")
print("=" * 60)
x = 1.0
for chop_i in range(20):
    alpha_trial = np.maximum(alpha + x * delta, 0.0)
    G_trial     = C * (K @ alpha_trial)
    res_trial   = G_trial - G_data
    chi2_trial  = float(np.sum((res_trial / sigma_G) ** 2) / n_tau)
    print(f"  chop {chop_i:2d}  x={x:.4e}  chi2={chi2_trial:.4f}  "
          f"alpha_trial range=[{alpha_trial.min():.3e}, {alpha_trial.max():.3e}]")
    if chi2_trial <= chi2_target:
        print(f"  → accepted at x={x:.4e}")
        break
    x *= 0.5
else:
    print(f"  → alpha-chop exhausted 20 halvings, x={x:.4e}")
print()

# ── run a short iteration trace ────────────────────────────────
print("=" * 60)
print("ITERATION TRACE (first 20 iterations)")
print("=" * 60)

alpha = np.ones(n_comp) / n_comp
C_ls  = float(np.sum(G_data * (K @ alpha)) / np.sum((K @ alpha) ** 2))
C     = max(C_ls, 1e-10)

for it in range(20):
    Z      = float(np.sum(alpha))
    p      = np.maximum(alpha / Z, 1e-300)
    log_p  = np.log(p)
    S      = float(-np.sum(p * log_p))
    grad_S = (S - log_p) / Z

    G_fit_it  = C * (K @ alpha)
    residual  = G_fit_it - G_data
    wr_it     = residual / sigma_G
    chi2      = float(np.sum(wr_it ** 2) / n_tau)
    grad_chi2 = (2.0 * C / n_tau) * (K.T @ (w * residual))
    hess_diag_chi2 = (2.0 * C ** 2 / n_tau) * (K.T ** 2 @ w)

    # C newton step
    K_a    = K @ alpha
    g_C    = float((2.0 / n_tau) * np.sum(w * residual * K_a))
    h_C    = float((2.0 / n_tau) * np.sum(w * K_a ** 2))
    C      = max(C - g_C / max(h_C, 1e-300), 1e-10)

    e1 = _normalise(grad_S)
    e2 = _normalise(grad_chi2)
    hess_pos = np.maximum(hess_diag_chi2, 1e-300)
    e3 = _normalise(hess_pos * e1)
    E  = np.column_stack([e1, e2, e3])
    gS_3 = E.T @ grad_S
    H3   = E.T @ (hess_pos[:, None] * E)
    try:
        reg   = 1e-10 * max(np.max(np.abs(H3)), 1.0) * np.eye(3)
        c     = np.linalg.solve(H3 + reg, gS_3)
        delta = E @ c
    except Exception:
        delta = grad_S / (np.linalg.norm(grad_S) + 1e-300)

    x = 1.0
    for _ in range(50):
        a_trial   = np.maximum(alpha + x * delta, 0.0)
        chi2_t    = float(np.sum(((C * (K @ a_trial) - G_data) / sigma_G) ** 2) / n_tau)
        if chi2_t <= chi2_target:
            break
        x *= 0.5

    for _ in range(50):
        if np.all(alpha + x * delta >= 0.0):
            break
        x *= 0.5

    alpha_new = alpha + x * delta
    alpha_new = np.where(alpha_new < 0, 0.0, alpha_new)
    alpha_new = np.where(alpha_new == 0, 1e-300, alpha_new)
    alpha     = alpha_new

    print(f"  it={it:3d}  chi2={chi2:.5f}  S={S:.5f}  C={C:.4e}  "
          f"x={x:.3e}  |delta|={np.linalg.norm(delta):.3e}  "
          f"alpha=[{alpha.min():.3e}, {alpha.max():.3e}]")

print()

# ── save diagnostic plots ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("MEMFCS Debug", fontsize=12)

ax = axes[0, 0]
ax.semilogx(tau, G_data, 'r', label='G observed')
ax.semilogx(tau, C * (K @ alpha), 'g', label='G fit (20 iters)')
ax.semilogx(tau, C_init_v3 * (K @ np.ones(n_comp)/n_comp), 'b--',
            label='G fit (flat init)')
ax.set_xlabel("τ (s)")
ax.set_ylabel("G(τ)")
ax.set_title("Correlation curve")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.semilogx(tau_D, alpha, 'g')
ax.set_xlabel("τ_D (s)")
ax.set_ylabel("Amplitude")
ax.set_title("α distribution after 20 iterations")
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.semilogx(tau, G_data, 'r', label='G data')
ax.semilogx(tau, C_init_v3 * G_model_flat, 'b--',
            label=f'Flat model × C={C_init_v3:.4f}')
ax.set_xlabel("τ (s)")
ax.set_ylabel("G(τ)")
ax.set_title("Flat model vs data (should overlap if C is right)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
K_col_sums = K.sum(axis=0)
ax.semilogx(tau_D, K_col_sums, 'steelblue')
ax.set_xlabel("τ_D (s)")
ax.set_ylabel("Σ_i K[i,j]")
ax.set_title("Kernel column sums (sensitivity at each τ_D)")
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = csv_path.replace(".csv", "_memfcs_debug.svg")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Debug plot saved to: {out_path}")