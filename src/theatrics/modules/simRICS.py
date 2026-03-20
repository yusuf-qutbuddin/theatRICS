# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 20:26:25 2025

@author: yusufqq
"""
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import multiprocessing
# from scipy.ndimage import gaussian_filter

def simulate_single_frame(args):
    (frame_idx, img_shape, pixel_dwell_time_us, pixel_size_nm,
     brightness_khz, n_particles, diffusion_um2_s, background, psf_sigma_px, seed) = args

    np.random.seed(seed)  # Ensures random, but reproducible (optional)

    pixel_size_um = pixel_size_nm / 1000.0
    pixel_dwell_time_s = pixel_dwell_time_us * 1e-6
    D_px2_dw = diffusion_um2_s * pixel_dwell_time_s / pixel_size_um ** 2 # diffusion step (in pixel units) during a pixel dwell time
    brightness_per_dwell = brightness_khz * 1000 * pixel_dwell_time_s

    # Random initial positions for this frame
    positions = np.random.rand(n_particles, 2) * np.array(img_shape)

    img = np.ones(img_shape) * background
    # Raster scan per pixel
    for iy in range(img_shape[0]):
        for ix in range(img_shape[1]):
            for ip in range(n_particles):
                y, x = positions[ip]
                dist2 = (y - iy) ** 2 + (x - ix) ** 2
                if dist2 <= 400:
                    spot_intensity = brightness_per_dwell * np.exp(-dist2 / (2 * psf_sigma_px ** 2))
                    img[iy, ix] += spot_intensity
                else:
                    pass
            # Brownian update per pixel dwell
            displ = np.random.normal(scale=np.sqrt(2 * D_px2_dw), size=(n_particles, 2))
            positions = (positions + displ) % np.array(img_shape)
    # img = np.random.poisson(img)
    return img.astype(np.uint16)

def simulate_single_frame_aniso(args):
    (frame_idx, img_shape, pixel_dwell_time_us, pixel_size_nm,
     brightness_khz, n_particles, diffusion_um2_s_x, diffusion_um2_s_y, background, psf_sigma_px, seed) = args

    np.random.seed(seed)  # Ensures random, but reproducible (optional)

    pixel_size_um = pixel_size_nm / 1000.0
    pixel_dwell_time_s = pixel_dwell_time_us * 1e-6
    D_px2_dw_x = diffusion_um2_s_x * pixel_dwell_time_s / pixel_size_um ** 2
    D_px2_dw_y = diffusion_um2_s_y * pixel_dwell_time_s / pixel_size_um ** 2
    
    brightness_per_dwell = brightness_khz * 1000 * pixel_dwell_time_s

    # Random initial positions for this frame
    positions = np.random.rand(n_particles, 2) * np.array(img_shape)

    img = np.ones(img_shape) * background
    # Raster scan per pixel
    for iy in range(img_shape[0]):
        for ix in range(img_shape[1]):
            for ip in range(n_particles):
                y, x = positions[ip]
                dist2 = (y - iy) ** 2 + (x - ix) ** 2
                spot_intensity = brightness_per_dwell * np.exp(-dist2 / (2 * psf_sigma_px ** 2))
                img[iy, ix] += spot_intensity
            # Brownian update per pixel dwell
            displ_x = np.random.normal(scale=np.sqrt(2 * D_px2_dw_x), size=n_particles)
            displ_y = np.random.normal(scale=np.sqrt(2 * D_px2_dw_y), size=n_particles)
            displ = np.stack([displ_y, displ_x], axis=1)
            positions = (positions + displ) % np.array(img_shape)
    # img = np.random.poisson(img)
    return img.astype(np.uint16)

def simulate_single_frame_aniso_rotated(args):
    (frame_idx, img_shape, pixel_dwell_time_us, pixel_size_nm,
     brightness_khz, n_particles, diffusion_um2_s_x, diffusion_um2_s_y,
     rotation_deg, background, psf_sigma_px, seed) = args
     
    np.random.seed(seed)
    pixel_size_um = pixel_size_nm / 1000.0
    pixel_dwell_time_s = pixel_dwell_time_us * 1e-6

    # Rotation matrix
    theta = np.deg2rad(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])

    # Diffusion tensor
    D = np.array([[diffusion_um2_s_x, 0],
                  [0, diffusion_um2_s_y]])
    
    # Rotated diffusion tensor
    D_rot = R @ D @ R.T
   
    # Covariance matrix for displacements
    Sigma = 2 * D_rot * pixel_dwell_time_s / (pixel_size_um**2)

    # Cholesky decomposition for sampling correlated displacements
    L = np.linalg.cholesky(Sigma)

    brightness_per_dwell = brightness_khz * 1000 * pixel_dwell_time_s

    positions = np.random.rand(n_particles, 2) * np.array(img_shape)
    img = np.ones(img_shape) * background

    for iy in range(img_shape[0]):
        for ix in range(img_shape[1]):
            for ip in range(n_particles):
                y, x = positions[ip]
                dist2 = (y - iy) ** 2 + (x - ix) ** 2
                spot_intensity = brightness_per_dwell * np.exp(-dist2 / (2 * psf_sigma_px ** 2))
                img[iy, ix] += spot_intensity
            
            # Sample correlated displacements for all particles
            normal_samples = np.random.normal(size=(n_particles, 2))
            displ = normal_samples @ L.T  # correlated displacements
            
            positions = (positions + displ) % np.array(img_shape)

    return img.astype(np.uint16)

def plot_tiff_frame0(tif_path, cmap='gray'):
    """
    Load TIFF stack and plot frame 0.
    """
    stack = tifffile.imread(tif_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(stack[0], cmap=cmap)
    plt.title(f'Frame 0 of {tif_path}')
    plt.axis('off')
    plt.show()
    return stack[0]  # returns the first frame for further analysis if needed



if __name__ == '__main__':
    pass
    
    # parallel_rics_raster_aniso(             # specialized use when you want anisotropy
    #     img_shape=(256, 256),
    #     n_frames=100,
    #     pixel_dwell_time_us=50,
    #     pixel_size_nm=20,
    #     brightness_khz=2000,
    #     avg_n_particles=250,
    #     diffusion_um2_s_x=10,
    #     diffusion_um2_s_y=5,
    #     background=0,
    #     psf_sigma_px=5, 
    #     output_path=output_path
    # )
    # parallel_rics_raster_aniso_rotated(             # specialized use when you want anisotropy and ability to rotate scanning frame
    #     img_shape=(256, 256),
    #     n_frames=100,
    #     pixel_dwell_time_us=50,
    #     pixel_size_nm=20,
    #     brightness_khz=2000,
    #     avg_n_particles=250,
    #     diffusion_um2_s_x=100,
    #     diffusion_um2_s_y=10,
    #     rotation_deg=90,
    #     background=0,
    #     psf_sigma_px=5, 
    #     output_path=output_path
    # )

    


