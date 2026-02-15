import numpy as np
import tifffile as tiff  # pip install tifffile
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import multipletau
import matplotlib.pyplot as plt
from pylibCZIrw import czi as pyczi

np.random.seed(42)
n_lines, n_pixels = 5000, 32
data = np.random.poisson(10, (n_lines, n_pixels))
for i in range(n_lines):
    center = np.random.normal(n_pixels//2, 2)
    data[i, max(0, int(center-3)):min(n_pixels, int(center+3))] += np.random.poisson(500, 6)

# Convert to a suitable integer type for TIFF
data = data.astype(np.uint16)

# Save as TIFF; shape is (n_lines, n_pixels)
tiff.imwrite('sfcs_lines.tif', data)

with pyczi.open_czi(input_file) as czidoc:
    total_bounding_box = czidoc.total_bounding_box
    n_frames = total_bounding_box['T'][1]

all_frames = []
for i_frame in range(n_frames):
    frame_data = read_frame(filepath, i_frame, channel_to_use)
    all_frames.append(frame_data)
all_frames = np.stack(all_frames, axis = 0)

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

#alignment
center_target = n_pixels // 2
aligned_data = np.zeros_like(data)
for i in range(n_lines):
    shift_amt = int(center_target - peaks[i])
    aligned_data[i] = np.roll(data[i], shift_amt)

# Save as TIFF; shape is (n_lines, n_pixels)
tiff.imwrite('sfcs_lines_aligned.tif', aligned_data)

# Make the intensity trace here
# Sum photons in ±2.5σ window per line for intensity trace
intensity_traces = np.zeros(n_lines)
for i in range(n_lines):
    half_width = 2.5 * sigmas[i]
    start = max(0, int(center_target - half_width))
    end = min(n_pixels, int(center_target + half_width))
    intensity_traces[i] = np.sum(aligned_data[i, start:end])

G = multipletau.autocorrelate(intensity_traces,
                                  m=12, deltat=1e-3, normalize=True)



