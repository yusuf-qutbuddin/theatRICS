import numpy as np
import tifffile as tiff  # pip install tifffile
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import multipletau
import matplotlib.pyplot as plt
from pylibCZIrw import czi as pyczi
import multiprocessing
import os
import pandas as pd

# np.random.seed(42)
# n_lines, n_pixels = 5000, 32
# data = np.random.poisson(10, (n_lines, n_pixels))
# for i in range(n_lines):
#     center = np.random.normal(n_pixels//2, 2)
#     data[i, max(0, int(center-3)):min(n_pixels, int(center+3))] += np.random.poisson(500, 6)

def read_frame(filepath,
               channel):
    with pyczi.open_czi(filepath) as czidoc:
        metadata = czidoc.metadata['ImageDocument']['Metadata']
        try:
            Frame_time_s = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel']['LaserScanInfo']['LineTime'])
        except:
            Frame_time_s = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel']['LaserScanInfo']['FrameTime'])

        total_bounding_rectangle = czidoc.total_bounding_rectangle
        data_frame = czidoc.read(roi=total_bounding_rectangle,
                                 plane={
                                     'C': channel})
    return data_frame, Frame_time_s

def gaussian(x, amp, cen, sigma):
        return amp * np.exp(-(x - cen) ** 2 / (2 * sigma ** 2))

def fit_gaussian_line(args):

    """Fitting function for multiprocessing - unpacks (i, line_data, x, n_pixels)"""
    (i, line_data, x, n_pixels) = args
    y = line_data.astype(float)

    # Skip empty / almost flat lines
    if np.allclose(y, 0) or np.nanmax(y) <= 0:
        result = (i,np.argmin(y),5.0)
        return result

    y_smooth = gaussian_filter1d(y, sigma=1)

    # Safer initial guess
    amp0 = float(np.max(y_smooth))
    cen0 = float(np.argmax(y_smooth))
    sig0 = 5.0

    # Reasonable bounds
    bounds = (
        [0, 0, 0.5],  # lower
        [10 * amp0, n_pixels - 1, n_pixels]  # upper
    )

    try:
        popt, _ = curve_fit(
            gaussian, x, y_smooth,
            p0=[amp0, cen0, sig0],
            bounds=bounds,
            maxfev=2000
        )
        result = (i, popt[1], abs(popt[2]))
        return result
    except Exception:
        # Fallback: simple maximum
        result = (i, np.argmax(y_smooth), sig0)
        return result


def wohland_bootstrap(intensity_traces, line_time_s,G_lags, n_synthetic=500, n_pairs_per_lag=1000):
    """
    Wohland method: Synthetic ACFs via random time pair sampling
    n_synthetic: Number of synthetic correlation curves
    n_pairs_per_lag: Photon pairs sampled per lag time
    """
    print("Running Bootstrap standard deviation...")

    trace = intensity_traces.astype(float) - np.mean(intensity_traces)  # DEMEAN FIRST
    n_samples = len(trace)
    lags = (G_lags / line_time_s).astype(int)

    # Precompute global normalization from original G
    mean_intensity = np.mean(intensity_traces)
    global_norm = mean_intensity ** 2  # <I>² for proper FCS normalization

    synthetic_Gs = []

    for _ in range(n_synthetic):
        G_synth = np.zeros(len(lags))

        for lag_idx, tau in enumerate(lags):
            if tau >= n_samples:
                G_synth[lag_idx] = 0
                continue

            t_starts = np.random.randint(0, n_samples - tau, n_pairs_per_lag)
            corr_pairs = trace[t_starts] * trace[t_starts + tau]
            G_synth[lag_idx] = np.mean(corr_pairs)

        # FIXED: Use global normalization, not G_synth[0]
        G_synth = G_synth / global_norm
        synthetic_Gs.append(G_synth)

    return np.std(synthetic_Gs, axis=0)

def run_autocorrelation(intensity_traces, line_time_s, root):
    print("Running multipletau for autocorrelation...")
    G = multipletau.autocorrelate(intensity_traces, m=12, deltat=line_time_s, normalize=True)
    G_std = wohland_bootstrap(intensity_traces, line_time_s,G[:,0], n_synthetic=1000, n_pairs_per_lag=10000)
    countrate = np.full_like(G_std, np.mean(intensity_traces))
    correlate_df = pd.DataFrame({
        'lag_time': G[:, 0],
        'correlation': G[:, 1],
        'countrate': countrate,  # Same for row 1 & 2
        'std_dev': G_std
    })
    correlate_df.to_csv(root+'_correlation.csv',
                        index=False, header = False)
    # Plot
    return G, G_std


def read_file(filepath, channel):
    root, ext = os.path.splitext(filepath)
    channel_to_use = channel
    frame_data, line_time_s = read_frame(filepath, channel_to_use)

    tiff.imwrite(root + ".tif", frame_data)
    n_lines = frame_data.shape[0]
    n_pixels = frame_data.shape[1]
    frame_data = frame_data.reshape(n_lines, n_pixels)
    x = np.arange(n_pixels)

    return frame_data, line_time_s, x, n_lines, n_pixels, root
def alignment(frame_data, n_pixels, n_lines, root, peaks):
    # #alignment
    center_target = n_pixels // 2
    aligned_data = np.zeros_like(frame_data)
    for i in range(n_lines):
        shift_amt = int(center_target - peaks[i])
        aligned_data[i] = np.roll(frame_data[i], shift_amt)

    # Save as TIFF; shape is (n_lines, n_pixels)
    tiff.imwrite(root + "_aligned.tif", aligned_data)
    return aligned_data

def calculate_intensity_trace(aligned_data, n_lines, n_pixels, sigmas, root):
    # Make the intensity trace here
    # Sum photons in ±2.5σ window per line for intensity trace
    intensity_traces = np.zeros(n_lines)
    center_target = n_pixels // 2

    for i in range(n_lines):
        half_width = 2.5 * sigmas[i]
        start = max(0, int(center_target - half_width))
        end = min(n_pixels, int(center_target + half_width))
        intensity_traces[i] = np.sum(aligned_data[i, start:end])

    intensity_df = pd.DataFrame({
        'line_number': range(len(intensity_traces)),
        'intensity': intensity_traces
    })
    intensity_df.to_csv(root + "_intensity_trace.csv", index=False)
    return intensity_traces
# Usage
if __name__ == "__main__":
    pass
    #
    # filepath = r'X:\AlNahas_Kareem\Raquel_mastersproject\20260220_sfcs\New-09_xy.czi'
    #
    # # Prepare arguments for multiprocessing
    # args_list = [(i, frame_data[i], x, n_pixels) for i in range(n_lines)]
    #
    # # Parallel Gaussian fitting - uses all available CPU cores
    # print("Fitting Gaussians with multiprocessing...")
    # pool = multiprocessing.Pool(processes=20)
    # results = pool.starmap(fit_gaussian_line, args_list)
    #
    # # Unpack results (maintain original order)
    # peaks = np.full(n_lines, 0.0)
    # sigmas = np.full(n_lines, 5.0)
    # for i, peak, sigma in results:
    #     peaks[i] = peak
    #     sigmas[i] = sigma
    #
    # print(f"Fitting complete. Found {np.sum(sigmas > 0)} valid peaks.")
    # # for i in range(n_lines):
    # #     y_smooth = gaussian_filter1d(frame_data[i], sigma=1)
    # #     try:
    # #         popt, _ = curve_fit(gaussian, x, y_smooth, p0=[np.max(y_smooth), n_pixels//2, 5])
    # #         peaks[i], sigmas[i] = popt[1], np.abs(popt[2])  # Ensure positive sigma
    # #     except:
    # #         peaks[i] = np.argmax(y_smooth)
    #
    #
    #
    # # Make the intensity trace here
    # # Sum photons in ±2.5σ window per line for intensity trace
    # intensity_traces = np.zeros(n_lines)
    # for i in range(n_lines):
    #     half_width = 2.5 * sigmas[i]
    #     start = max(0, int(center_target - half_width))
    #     end = min(n_pixels, int(center_target + half_width))
    #     intensity_traces[i] = np.sum(aligned_data[i, start:end])
    #
    # intensity_df = pd.DataFrame({
    #     'line_number': range(len(intensity_traces)),
    #     'intensity': intensity_traces
    # })
    # intensity_df.to_csv(root+"_intensity_trace.csv", index=False)
    #
    #
    # plt.plot(intensity_traces)
    # plt.xlabel('Line #')
    # plt.ylabel('Intensity')
    # plt.savefig(root+"_intensity_trace.svg")
    # plt.close()
    #
    #
    # run_autocorrelation(intensity_traces, line_time_s, root)
    #
    # #
    #