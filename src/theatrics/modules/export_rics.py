# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 22:14:02 2025

@author: yusufqq (improved from a previous version of Krohn)
"""

import os
from pylibCZIrw import czi as pyczi
import numpy as np
import multiprocessing
import tifffile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.fftpack as fftpack
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

# # File

# folder = r''
# file_extension = '.tif' # or '.czi'
# ending_word = ''

# # Parameters
# channel_to_use = 0
# crop_factor = 0.5 # creates a square roi in the centre with this crop factor prints the size of the ROI
# window_size = 3 # WIndow size for moving average correction (has to be an odd number)
# parallel_processing = False
# num_workers = 256
# correct_drift = False
# #%% Functions


def get_files_from_folder(folder_path, extension, suffix):
    """
    Return a list of full file paths in 'folder_path' with the given 'extension'.
    extension should include the dot, e.g., '.tif' or '.czi'
    """
    all_files = os.listdir(folder_path)
    filtered_files = []
    for f in all_files:
        fname, ext = os.path.splitext(f)
        if ext.lower() == extension.lower() and fname.lower().endswith(suffix.lower()):
            filtered_files.append(os.path.join(folder_path, f))
    return filtered_files
def estimate_drift(stack, roi=None, gauss_sigma=2):
    """
    Estimate x/y drift between frames using cross-correlation in an ROI.

    stack: (n_frames, height, width) array
    roi: (y_from, y_to, x_from, x_to) or None for central square
    gauss_sigma: Gaussian filter sigma applied to correlation map
    Returns:
        drift_x, drift_y -- arrays of length (n_frames-1)
    """
    if roi is None:
        # Center square ROI at 50% of image size
        h, w = stack.shape[1:]
        size = min(h, w) // 2
        y_from = (h - size)//2
        x_from = (w - size)//2
        roi = (y_from, y_from+size, x_from, x_from+size)
    y_from, y_to, x_from, x_to = roi
    ref = stack[0, y_from:y_to, x_from:x_to]
    y0, x0 = (ref.shape[0]//2, ref.shape[1]//2)
    drift_x = []
    drift_y = []
    for i in range(1, stack.shape[0]):
        frame = stack[i, y_from:y_to, x_from:x_to]
        # Cross-correlation via FFT
        corr = fftconvolve(ref, frame[::-1, ::-1], mode='same')
        corr = gaussian_filter(corr, gauss_sigma)
        y_peak, x_peak = np.unravel_index(np.argmax(corr), corr.shape)
        drift_y.append(int(y_peak - y0))
        drift_x.append(int(x_peak - x0))
    # Prepend 0 drift for frame 0
    drift_x = np.array([0]+drift_x)
    drift_y = np.array([0]+drift_y)
    return drift_x, drift_y

def apply_drift(stack, drift_x, drift_y):
    """
    Apply integer circular shifts to stack (like circshift in MATLAB)
    """
    aligned = np.zeros_like(stack)
    for i, (dx, dy) in enumerate(zip(drift_x, drift_y)):
        aligned[i] = np.roll(stack[i], shift=(dy, dx), axis=(0, 1))
    return aligned

def plot_drift(drift_x, drift_y):
    plt.figure()
    plt.plot(drift_x, label='x drift')
    plt.plot(drift_y, label='y drift')
    plt.xlabel('Frame')
    plt.ylabel('Drift (pixels)')
    plt.legend()
    plt.tight_layout()
    plt.show()
def drift_correct(image_stack):
    drift_x, drift_y = estimate_drift(image_stack, roi=None)
    plot_drift(drift_x, drift_y)
    aligned_stack = apply_drift(image_stack, drift_x, drift_y)
    return aligned_stack



def autocorrelation_map(image):
    """
    Calculate a two-dimensional autocorrelation map from an image.

    Parameters:
    image (numpy array): A 2D numpy array representing the image.

    Returns:
    autocorrelation_map (numpy array): A 2D numpy array representing the autocorrelation map.
    """
    # Apply 2D Hanning window to reduce edge artifacts
    # window = np.outer(windows.hann(image.shape[0]), windows.hann(image.shape[1]))
    # image_windowed = image * window
    
    # Calculate the number of pixels per frame
    total_pixels = image.size
    
    
    # Calculate the Fourier transform of the image
    F = fftpack.fft2(image) 

    # # Calculate the power spectrum in the Fourier domain
    RICS_frame_F = np.abs(F)**2
    
    # Calculate the inverse fourier transform to get the RICS map
    RICS_frame = np.real(fftpack.ifft2(RICS_frame_F))
    
    # Calculate mean intensity and then normalize with total pixels * mean_intensity_squared 
    mean_intensity = np.mean(image)
    RICS_frame = RICS_frame/(total_pixels*mean_intensity**2) - 1 # Subtraction of 1 results in uncorrelated pixels to have 0 value (the normalization procedure is accroding to PAM from the FAB Lab)
    
    
    # # Shift the autocorrelation map such that the zero lag component is in the centre
    RICS_frame = fftpack.fftshift(RICS_frame)  # center zero lag
    return RICS_frame





def moving_average_correction(image_stack, window_size=3):
    # Perform moving average correction such that the average pixel means are subtracted and total mean of the image is added back 
    assert window_size % 2 == 1, "window_size must be odd"
    num_frames = image_stack.shape[0]
    half_w = window_size // 2
    corrected_frames = []
    for t in range(half_w, num_frames - half_w):
        local_stack = image_stack[t - half_w : t + half_w + 1]
        moving_avg = np.mean(local_stack, axis=0)
        corrected = image_stack[t] - moving_avg + np.mean(image_stack[t])
        corrected_frames.append(corrected)
    return np.stack(corrected_frames, axis=0)
    # return image_stack

def read_frame(filepath, 
               frame, 
               channel, crop_factor):
    with pyczi.open_czi(filepath) as czidoc:
        total_bounding_rectangle = czidoc.total_bounding_rectangle
        data_frame = czidoc.read(roi = total_bounding_rectangle,
                                 plane = {'T':frame,
                                          'C':channel})
        data_frame = data_frame.reshape([data_frame.shape[0], data_frame.shape[1]])
        data_frame = crop_center(data_frame, crop_factor)
    return data_frame.reshape([data_frame.shape[0], data_frame.shape[1]])


def process_frame(filepath,
                  frame_to_use,
                  channel_to_use):
    # Read frame
    image_frame = read_frame(filepath, 
                              frame_to_use, 
                              channel_to_use, crop_factor)
    
    # Get correlation map
    RICS_frame = autocorrelation_map(image_frame)
    
    return RICS_frame


def bootstrap_sd(RICS_frames_array,
                 n_bs_reps = 10):
    
    bs_rep_maps = np.zeros(shape = [RICS_frames_array.shape[0],
                                    RICS_frames_array.shape[1],
                                    n_bs_reps])
    
    correction = RICS_frames_array.shape[2] / (RICS_frames_array.shape[2] - 1)
    for i_rep in range(n_bs_reps):
        indxs = np.random.choice(RICS_frames_array.shape[2],
                                 size = RICS_frames_array.shape[2],
                                 replace = True)
        
        bs_resample = RICS_frames_array[:,:,indxs]
        
        bs_rep_maps[:,:,i_rep] = np.mean(bs_resample,
                                         axis = 2) * correction
        
    sd_map = np.std(bs_rep_maps,
                    axis = 2)
    
    return sd_map
    
    
def process_all_frames_czi(filepath, 
                       n_frames,
                       channel_to_use,
                       window_size, crop_factor, correct_drift):
   
    all_frames = []
    for i_frame in range(n_frames):
        frame_data = read_frame(filepath, i_frame, channel_to_use, crop_factor)
    
        all_frames.append(frame_data)
    all_frames = np.stack(all_frames, axis = 0)
    if correct_drift:
        all_frames = drift_correct(all_frames)
    else:
        pass
    # Apply moving average correction
    corrected_stack = moving_average_correction(all_frames, window_size=window_size)
    reduced_frame_count = corrected_stack.shape[0]

    # Process autocorrelation map for each corrected frame
    RICS_frames_list = [autocorrelation_map(corrected_stack[i]) for i in range(reduced_frame_count)]
    RICS_frames_array = np.stack(RICS_frames_list, axis=2)  # Now [height, width, reduced_frame_count]                                                                
    sd_map = bootstrap_sd(RICS_frames_array)
    RICS_map = np.mean(RICS_frames_array,
                       axis = 2) * reduced_frame_count / (reduced_frame_count - 1)
    return RICS_map, sd_map, all_frames, corrected_stack

def process_all_frames_tiff(stack, 
                       n_frames,
                       channel_to_use,
                       window_size = 3):
    
    corrected_stack = moving_average_correction(stack, window_size=window_size)
    reduced_frame_count = corrected_stack.shape[0]
    # Process all frames of the stack and compute autocorrelation maps
    
    RICS_maps = [autocorrelation_map(corrected_stack[i]) for i in range(reduced_frame_count)]
    
    RICS_frames_array = np.stack(RICS_maps, axis=2)  # shape (height, width, n_frames)
    
    # Bootstrap standard deviation
    sd_map = bootstrap_sd(RICS_frames_array)
    
    # Mean and correction
    RICS_map = np.mean(RICS_frames_array, axis=2) * n_frames / (n_frames - 1)
    
    return RICS_map, sd_map, stack, corrected_stack

def crop_center(image, crop_factor):
    """
    Crop a centered region of interest (ROI) from an image based on crop_factor.

    Parameters:
    image (np.array): 2D or 3D numpy array (if 3D, crops spatial dimensions)
    crop_factor (float): fraction of the image to keep in each dimension (0 < crop_factor <= 1)

    Returns:
    cropped_image (np.array): cropped ROI image
    """
    # Get image shape
    if image.ndim == 2:
        height, width = image.shape
    elif image.ndim == 3:
        # Assume shape (frames, height, width)
        height, width = image.shape[1], image.shape[2]
    else:
        raise ValueError("Unsupported image shape")

    # Calculate crop size
    crop_height = int(height * crop_factor)
    crop_width = int(width * crop_factor)

    # Calculate centered start and end indices
    start_y = (height - crop_height) // 2
    end_y = start_y + crop_height
    start_x = (width - crop_width) // 2
    end_x = start_x + crop_width

    # Crop image
    if image.ndim == 2:
        cropped_image = image[start_y:end_y, start_x:end_x]
    else:
        cropped_image = image[:, start_y:end_y, start_x:end_x]

    return cropped_image



# Processing  
def plot_rics_workflow(all_frames, corrected_stack, RICS_map, sd_map, filepath):
    # Crop for visualization
    crop_factor_fast_ax = 0.5
    crop_factor_slow_ax = 0.5
    floor_fast_ax = int(np.floor(RICS_map.shape[1] * (1 - crop_factor_fast_ax) * 0.5))
    ceil_fast_ax = int(np.floor(RICS_map.shape[1] * 0.5 * (1 + crop_factor_fast_ax)))
    floor_slow_ax = int(np.floor(RICS_map.shape[0] * (1 - crop_factor_slow_ax) * 0.5))
    ceil_slow_ax = int(np.floor(RICS_map.shape[0] * 0.5 * (1 + crop_factor_slow_ax)))
    RICS_map = RICS_map[floor_slow_ax:ceil_slow_ax,
                        floor_fast_ax:ceil_fast_ax] 
    sd_map = sd_map[floor_slow_ax:ceil_slow_ax,
                        floor_fast_ax:ceil_fast_ax]
    center_y = RICS_map.shape[0] // 2
    center_x = RICS_map.shape[1] // 2
    RICS_map[center_y, center_x] = 0.0
    sd_map[center_y, center_x] = 0.0
    fig = plt.figure(figsize=(12, 6), facecolor='none')  # transparent figure background
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 2])  # The third column is twice as wide
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(all_frames[0], cmap='gray')
    ax1.set_title("Raw Image (Frame 0)")
    ax1.axis('off')
    ax1.patch.set_alpha(0)  # transparent background
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(corrected_stack[0], cmap='gray')
    ax2.set_title("After Moving Avg Correction (Frame 0)")
    ax2.axis('off')
    ax2.patch.set_alpha(0)
    
    ax3 = fig.add_subplot(gs[1, 0])
    img3 = ax3.imshow(RICS_map, cmap='jet')
    ax3.set_title("RICS map (Autocorrelation)")
    plt.colorbar(img3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis('off')
    ax3.patch.set_alpha(0)
    
    ax4 = fig.add_subplot(gs[1, 1])
    img4 = ax4.imshow(sd_map, cmap='jet')
    ax4.set_title("sd map (Uncertainty)")
    plt.colorbar(img4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.axis('off')
    ax4.patch.set_alpha(0)
    
    ax5 = fig.add_subplot(gs[:, 2], projection='3d')
    X = np.arange(RICS_map.shape[1])
    Y = np.arange(RICS_map.shape[0])
    X, Y = np.meshgrid(X, Y)
    ax5.view_init(elev=20, azim=90)
    ax5.plot_surface(X, Y, RICS_map, cmap='jet', linewidth=0, antialiased=False)
    ax5.set_title('RICS Map')
    ax5.set_xlabel('X (fast axis)')
    ax5.set_ylabel('Y (slow axis)')
    ax5.patch.set_alpha(0)
    
    plt.tight_layout()
    # Save as transparent svg
    fig.savefig(os.path.splitext(filepath)[0] + '.svg', dpi=300, transparent=True)
    plt.close()


def process_single_file(filepath):
    print("Processing: "+ str(filepath))
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.czi':
        # Import pylibCZIrw only if needed 
        # Get metadata
        with pyczi.open_czi(filepath) as czidoc:
            
            # Here's a list of attributes and methods to 
            # try out when one wants to change stuff
            # CZIreader_methods = [method_name for method_name in dir(czidoc)]
            
            # Read out some metadata
            # metadata = czidoc.metadata['ImageDocument']['Metadata'] # These two layers are pro-forma, there is nothing else in here
            # px_size_nm = float(metadata['Scaling']['Items']['Distance'][0]['Value']) * 1E9
            print(f'Data type: {czidoc.get_channel_pixel_type(0)}')
            
            # get the image dimensions as a dictionary, where the key identifies the dimension
            total_bounding_box = czidoc.total_bounding_box
            # total_bounding_rectangle = czidoc.total_bounding_rectangle
            n_frames = total_bounding_box['T'][1]
            RICS_map, sd_map, all_frames, corrected_stack = process_all_frames_czi(filepath, 
                                                  n_frames,
                                                  channel_to_use,
                                                  window_size = 3)
    
    elif ext in ['.tif', '.tiff']:
        stack = tifffile.imread(filepath)
        
        cropped_img = crop_center(stack, crop_factor=crop_factor)  # keeps central part of the image with the crop factor
        print(cropped_img.shape)
        # print("ROI Size = "+ str(cropped_img.shape[1])+"x"+str(cropped_img.shape[2]))
        print("Window size = "+str(window_size))
        n_frames = cropped_img.shape[0]
        if correct_drift:
            cropped_img = drift_correct(cropped_img)
        else:
            pass
        
        RICS_map, sd_map, all_frames, corrected_stack = process_all_frames_tiff(cropped_img, 
                                              n_frames,
                                              channel_to_use,
                                              window_size = window_size)
    # Actual processing
   
    plot_rics_workflow(all_frames, corrected_stack, RICS_map, sd_map,filepath)
    print('ROI: '+str(RICS_map.shape[0])+"x"+str(RICS_map.shape[1]))
    _ = tifffile.imwrite(os.path.splitext(filepath)[0] + '_RICScorr.tif', 
                         RICS_map, 
                         photometric='minisblack')
    _ = tifffile.imwrite(os.path.splitext(filepath)[0] + '_RICSunc.tif', 
                         sd_map, 
                         photometric='minisblack')
  
    
# ────────────────────────────────────────────────────────────────
# PTU (PicoQuant Luminosa) support
# ────────────────────────────────────────────────────────────────

try:
    import tttrlib
    TTTRLIB_AVAILABLE = True
except ImportError:
    TTTRLIB_AVAILABLE = False


def _ptu_get_tag(header_data: dict, key: str, fallback=None):
    """
    Safely extract a scalar value from header.data.
    header.data values are always lists; take the first element.
    Returns fallback if key is missing or value is None.
    """
    val = header_data.get(key, None)
    if val is None:
        return fallback
    if isinstance(val, list):
        if len(val) == 0:
            return fallback
        val = val[0]
    if val is None:
        return fallback
    return val


def read_ptu_metadata(filepath: str) -> dict:
    """
    Read pixel size, pixel dwell time, line time, and image dimensions
    from a PicoQuant Luminosa PTU CLSM image file.

    Uses header.data (a plain dict, values are lists) and
    CLSMImage direct attributes where available.

    Confirmed working tag names from Luminosa PTU header:
        ImgHdr_PixResol       — pixel size in µm/pixel
        ImgHdr_TimePerPixel   — pixel dwell time in ms
        ImgHdr_PixX           — number of pixels (fast axis)
        ImgHdr_PixY           — number of lines (slow axis)
        ImgHdr_MaxFrames      — maximum frames set in acquisition

    CLSMImage direct attributes (more reliable than header tags):
        clsm_image.pixel_duration  — pixel dwell time in seconds
        clsm_image.line_duration   — line time in seconds
        clsm_image.n_frames        — actual number of frames recorded
        clsm_image.n_lines         — number of lines
        clsm_image.n_pixel         — number of pixels per line

    Parameters
    ----------
    filepath : str
        Path to the .ptu file.

    Returns
    -------
    dict with keys:
        pixel_size_nm       : float  — nm/pixel
        pixel_dwell_time_us : float  — µs per pixel
        line_time_ms        : float  — ms per line
        n_frames            : int    — actual frames recorded
        n_lines             : int    — lines per frame
        n_pixels            : int    — pixels per line
        channels            : list   — available routing channel indices
    """
    if not TTTRLIB_AVAILABLE:
        raise ImportError(
            "tttrlib is not installed. "
            "Install it with:  pip install tttrlib"
        )

    tttr_data   = tttrlib.TTTR(str(filepath))
    header_data = tttr_data.header.data   # plain dict, values are lists

    # ── pixel size ───────────────────────────────────────────────
    # ImgHdr_PixResol is in µm/pixel
    pix_resol_um = _ptu_get_tag(header_data, "ImgHdr_PixResol", None)
    pixel_size_nm = float(pix_resol_um) * 1000.0 if pix_resol_um is not None else None

    # ── image dimensions from header ─────────────────────────────
    n_pixels_hdr = _ptu_get_tag(header_data, "ImgHdr_PixX", None)
    n_lines_hdr  = _ptu_get_tag(header_data, "ImgHdr_PixY", None)

    # ── CLSMImage for reliable timing and actual frame count ─────
    # channels=[0] is safe here — we only need timing, not all photons
    clsm_image = tttrlib.CLSMImage(tttr_data, fill=False, channels=[0])

    # pixel_duration and line_duration are in milliseconds
    pixel_dwell_time_us = float(clsm_image.pixel_duration) * 1e3
    line_time_ms        = float(clsm_image.line_duration)  

    # actual frame count from CLSM (more reliable than ImgHdr_MaxFrames)
    n_frames = int(clsm_image.n_frames)
    n_lines  = int(clsm_image.n_lines)
    n_pixels = int(clsm_image.n_pixel)

    # ── available routing channels ───────────────────────────────
    channels = sorted(int(c) for c in np.unique(tttr_data.routing_channels))

    return {
        "pixel_size_nm":       pixel_size_nm,
        "pixel_dwell_time_us": pixel_dwell_time_us,
        "line_time_ms":        line_time_ms,
        "n_frames":            n_frames,
        "n_lines":             n_lines,
        "n_pixels":            n_pixels,
        "channels":            channels,
    }


def read_ptu_stack(filepath: str,
                   channel: int = 0) -> np.ndarray:
    """
    Load a PicoQuant Luminosa PTU CLSM image file and return the
    photon count image stack as a numpy array.

    Parameters
    ----------
    filepath : str
        Path to the .ptu file.
    channel  : int
        Routing channel index to extract (e.g. 0, 1, 2, 4 for Luminosa).
        Use read_ptu_metadata() to see available channels first.

    Returns
    -------
    stack : np.ndarray, shape (n_frames, n_lines, n_pixels), dtype float32
        Photon count image stack.

    Raises
    ------
    ImportError : if tttrlib is not installed
    ValueError  : if the requested channel is not present in the file
    """
    if not TTTRLIB_AVAILABLE:
        raise ImportError(
            "tttrlib is not installed. "
            "Install it with:  pip install tttrlib"
        )

    tttr_data = tttrlib.TTTR(str(filepath))

    # validate channel
    available = sorted(int(c) for c in np.unique(tttr_data.routing_channels))
    if channel not in available:
        raise ValueError(
            f"Channel {channel} not found in PTU file. "
            f"Available channels: {available}"
        )

    # fill=True fills the image with photon counts
    # channels=[channel] selects the routing channel
    clsm_image = tttrlib.CLSMImage(tttr_data, fill=True, channels=[channel])

    # intensity shape from debug: (n_frames, n_lines, n_pixels)
    # this is confirmed for Luminosa single-channel selection
    intensity = clsm_image.intensity

    # defensive: handle unlikely extra leading dim
    if intensity.ndim == 4:
        # (n_channels_selected, n_frames, n_lines, n_pixels) — take ch 0
        intensity = intensity[0]

    return intensity.astype(np.float32)


def process_all_frames_ptu(filepath: str,
                            channel: int = 0,
                            window_size: int = 3,
                            crop_factor: float = 0.5,
                            correct_drift: bool = False) -> tuple:
    """
    Full RICS export pipeline for a PicoQuant Luminosa PTU file.

    Reads the photon count image stack, optionally crops and
    drift-corrects it, then computes the RICS correlation map and
    uncertainty map using the same pipeline as CZI and TIFF.

    Parameters
    ----------
    filepath     : str   — path to .ptu file
    channel      : int   — routing channel index (0-based, see available
                           channels from read_ptu_metadata())
    window_size  : int   — moving-average window for background subtraction
    crop_factor  : float — fraction of image to keep (centred crop)
    correct_drift: bool  — apply drift correction before correlation

    Returns
    -------
    RICS_map        : np.ndarray (H, W)
    sd_map          : np.ndarray (H, W)
    all_frames      : np.ndarray (n_frames, H, W) — raw cropped stack
    corrected_stack : np.ndarray (n_frames, H, W) — background-subtracted
    """
    if not TTTRLIB_AVAILABLE:
        raise ImportError(
            "tttrlib is not installed. "
            "Install it with:  pip install tttrlib"
        )

    # load full stack — shape (n_frames, n_lines, n_pixels)
    stack    = read_ptu_stack(filepath, channel=channel)
    n_frames = stack.shape[0]

    # crop centre
    cropped = crop_center(stack, crop_factor=crop_factor)

    # optional drift correction
    if correct_drift:
        cropped = drift_correct(cropped)

    # reuse existing TIFF pipeline (works on any float32 stack)
    RICS_map, sd_map, all_frames, corrected_stack = process_all_frames_tiff(
        cropped, n_frames, 0, window_size
    )

    return RICS_map, sd_map, all_frames, corrected_stack#%% Main

if __name__ == '__main__':
    pass