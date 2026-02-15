import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import multipletau

np.random.seed(42)
n_lines, n_pixels = 5000, 32
D = 1  # Diffusion coeff (μm²/s → px²/line after scaling)
dt = 1e-3  # Line time (s)
sigma_psf_x = 1.5  # Lateral PSF width (px)

sigma_psf_y = 1.0  # PSF extent perpendicular (px)
lam_bg = 10  # Background photons/px
N_mol = 20  # Molecules

# y-positions: diffusion PERPENDICULAR to scan lines (membrane plane)
dy = np.sqrt(2 * D * dt) * np.random.randn(N_mol, n_lines)
y_pos = np.cumsum(dy, axis=1)  # y[t] random walk

data = np.full((n_lines, n_pixels), lam_bg)
xx = np.arange(n_pixels)
X, T = np.meshgrid(xx, np.arange(n_lines))
x_center = n_pixels // 2

for mol in range(N_mol):
    psf_x = np.exp(-(X - x_center)**2 / (2 * sigma_psf_x**2))  # Lateral Gaussian
    psf_y = np.exp(-y_pos[mol, :, np.newaxis]**2 / (2 * sigma_psf_y**2))  # Perp
    psf_2d = psf_x * psf_y  # 2D PSF
    data += np.random.poisson(100 * psf_2d / N_mol)

data = np.random.poisson(data).astype(np.uint16)
tiff.imwrite('sfcs_realistic_psf.tif', data)

# Gaussian fit function
def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

x = np.arange(n_pixels)
peaks = np.zeros(n_lines)
sigmas = np.full(n_lines, 5.0)
for i in range(n_lines):
    y_smooth = gaussian_filter1d(data[i], sigma=1)
    try:
        popt, _ = curve_fit(gaussian, x, y_smooth, p0=[np.max(y_smooth), n_pixels//2, 5])
        peaks[i], sigmas[i] = popt[1], np.abs(popt[2])  # Ensure positive sigma
    except:
        peaks[i] = np.argmax(y_smooth)
intensity_traces = np.zeros(n_lines)
center_target = n_pixels // 2

for i in range(n_lines):
    half_width = 2.5 * sigmas[i]
    start = max(0, int(center_target - half_width))
    end = min(n_pixels, int(center_target + half_width))
    intensity_traces[i] = np.sum(data[i, start:end])

G = multipletau.autocorrelate(intensity_traces,
                                  m=12, deltat=1e-3, normalize=True)
print(G)
plt.semilogx(G[:, 0], G[:, 1])
plt.ylim(0,0.01)
plt.show()
# data = np.random.poisson(10, (n_lines, n_pixels))
# for i in range(n_lines):
#     center = np.random.normal(n_pixels//2, 2)
#     data[i, max(0, int(center-3)):min(n_pixels, int(center+3))] += np.random.poisson(500, 6)

# # Fixed scan: x from 0→31 each line, PSF ONLY at x_center=16
# x_center = n_pixels // 2
# xx = np.arange(n_pixels)
#
# # Intensity: Poisson(λ = bg + sum_mol * PSF(y(t)))
# data = np.full((n_lines, n_pixels), lam_bg)
# for mol in range(N_mol):
#     # Detection ONLY when y≈0 (membrane) AND x=16 (scan position)
#     psf_y = np.exp(-y_pos[mol, :, np.newaxis]**2 / (2 * sigma_psf_y**2))
#     data[:, x_center] += np.random.poisson(100 * psf_y.flatten() / N_mol).astype(float)
#
# data = np.random.poisson(data).astype(np.uint16)
# tiff.imwrite('sfcs_perp_diffusion.tif', data)
#
# print(f'Peak at x={x_center}. Molecules diffuse in y → bursts at x=16 when crossing plane.')