# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:19:18 2024

@author: yusufqq (improved from a previous version of Krohn)
"""
import os
import skimage.io as skio
from pylibCZIrw import czi as pyczi

import numpy as np
import lmfit
import matplotlib.pyplot as plt 
# from mpl_toolkits.mplot3d import Axes3D  # register 3D projection
# import tifffile
import matplotlib.gridspec as gridspec
import pandas as pd
import inspect_metadata as im


#
# folder = r'/fs/pool/pool-schwille-user/Qutbuddin_Yusuf/_Protocols/RICS_fit/Sample_Data/Simulations/Sim_2_90'
# file_extension = '.tif' # or '.czi'
# ending_word = 'RICScorr'
# save_path = os.path.join(folder, 'Results')
#
# PSF_size_xy = 0.2
# PSF_aspect_ratio = 4.985423166
# crop_factor_fast_ax = 0.5
# crop_factor_slow_ax = 0.5
# file_for_metadata = ''
#
# if file_for_metadata:
#     metadata_path = os.path.join(folder, file_for_metadata)
#     with pyczi.open_czi(metadata_path) as czidoc:
#         Pixel_size_nm, Pixel_dwell_time_us, line_time_ms = im.get_metadata(czidoc)
#
# else:
#
#     # Parameters
#     Pixel_size_nm = 20 #nm
#     Pixel_dwell_time_us = 50 #us
#     line_time_ms = 12.8#ms
#
#
# # Models
# # diffusion_model = '3Ddiff'
# diffusion_model = '2Ddiff'
#
#
# # paramnter rescaling
# pixel_size_um = Pixel_size_nm * 1e-3
# psf_size_xy_um = PSF_size_xy
# psf_aspect_ratio = PSF_aspect_ratio
# pixel_time_s = Pixel_dwell_time_us * 1e-6
# line_time_s = line_time_ms * 1e-3
#


# Classes

class RICS_fit():
    
    def __init__(self,
                 RICS_map,
                 pixel_size_um,
                 pixel_time_s,
                 line_time_s,
                 psf_size_xy_um,
                 psf_aspect_ratio):
        
        # Data
        
       

        self.data_RICS = RICS_map
        # Geometry
        self.center_fast_ax = RICS_map.shape[1] // 2
        self.center_slow_ax = RICS_map.shape[0] // 2
        self.data_RICS_1D = self.data_RICS[self.center_slow_ax,:]
        # self.data_RICS_1D = self.data_RICS[:,self.center_fast_ax]

        self.n_points = len(self.data_RICS_1D)
        self.center = self.n_points//2
        self.x_lag = np.arange(self.n_points) - self.center
        # Calibration
        self.px_time_s = pixel_time_s
        self.line_time_s = line_time_s
        self.px_size_um = pixel_size_um
        self.psf_size_xy_um = psf_size_xy_um
        self.psf_aspect_ratio = psf_aspect_ratio
        

        
        meshgrid_slow_ax = np.repeat(np.reshape(np.arange(RICS_map.shape[0]),
                                                [RICS_map.shape[0], 1]),
                                     RICS_map.shape[1],
                                     axis = 1) - self.center_fast_ax
        meshgrid_fast_ax = np.repeat(np.reshape(np.arange(RICS_map.shape[1]),
                                                [1, RICS_map.shape[1]]),
                                     RICS_map.shape[0],
                                     axis = 0) - self.center_slow_ax
        self.spatial_lag_map = pixel_size_um**2 * (meshgrid_fast_ax**2 + meshgrid_slow_ax**2)
        self.temporal_lag_map = np.abs(pixel_time_s * meshgrid_fast_ax + line_time_s * meshgrid_slow_ax)
        self.spatial_lag_map_1D = self.spatial_lag_map[self.center_slow_ax,:]
        self.temporal_lag_map_1D = self.temporal_lag_map[self.center_slow_ax,:]
        # self.spatial_lag_map_1D = self.spatial_lag_map[:,self.center_fast_ax]
        # self.temporal_lag_map_1D = self.temporal_lag_map[:,self.center_fast_ax]
        

        return None
    
    
    def rics_3Ddiff_func(self,
                       amplitude,
                       diff_coeff,
                       offset):
        
        # Initialize map
        rics_model = np.ones_like(self.data_RICS,
                                  dtype = np.float64)
        # Apply amplitude
        rics_model*= amplitude
        
        # Apply xy diffusion term
        rics_model /= 1 + 4 * diff_coeff * self.temporal_lag_map / self.psf_size_xy_um**2
        
        # # Apply z diffusion term
        rics_model /= np.sqrt(1 + 4 * diff_coeff * self.temporal_lag_map / self.psf_size_xy_um**2 / self.psf_aspect_ratio**2)
        
        # Apply scanning term
        rics_model *= np.exp(- self.spatial_lag_map / (4 * diff_coeff * self.temporal_lag_map + self.psf_size_xy_um**2))
        
        # Add offset
        rics_model += offset
        return rics_model
    def rics_3Ddiff_residual(self,
                           params):
        # Evaluate model
        rics_model = self.rics_3Ddiff_func(params['amplitude'].value,
                                         params['diff_coeff'].value,
                                         params['offset'].value)
        # Get residual
        residual = rics_model - self.data_RICS
        
        return residual
    def rics_3Ddiff_cost(self,
                       params):
        # Get residual
        residual = self.rics_3Ddiff_residual(params)
        
        # Zero the center pixel residual - that one is not considered in fitting!
        residual[self.center_fast_ax, self.center_slow_ax] = 0.
        
        # Average to get cost
        cost = np.sum(residual**2)
        
        return cost  
    def run_3Ddiff_fit(self):
        # Initial parameters
        init_params = lmfit.Parameters()
        init_params.add(name = 'amplitude',
                        value = 1.,
                        vary = True,
                        min = 0.,
                        max = np.max(self.data_RICS) * 1.5)
        init_params.add(name = 'diff_coeff',
                        value = 1.,
                        vary = True,
                        min = 1e-3,
                        max = 1e4)
        init_params.add(name = 'offset',
                        value = 0.,
                        vary = True,
                        min = -np.max(self.data_RICS) / 10.,
                        max = np.max(self.data_RICS) / 10.)
        
        # Run fit
        result = lmfit.minimize(self.rics_3Ddiff_cost,
                                params = init_params, 
                                method = 'basinhopping')
        
        fit_params = result.params
        model = self.rics_3Ddiff_func(fit_params['amplitude'].value,
                                    fit_params['diff_coeff'].value,
                                    fit_params['offset'].value)
        residual = self.rics_3Ddiff_residual(fit_params)
        center_y = residual.shape[0] // 2
        center_x = residual.shape[1] // 2
        residual[center_y, center_x] = 0.0
        
        return fit_params, model, residual

    def rics_2Ddiff_func(self,
                    amplitude,
                    diff_coeff,
                    offset):
     
        # Initialize map
        rics_model = np.ones_like(self.data_RICS,
                                  dtype = np.float64)
        # Apply amplitude
        rics_model*= amplitude
        
        # Apply xy diffusion term
        rics_model /= 1 + 4 * diff_coeff * self.temporal_lag_map / self.psf_size_xy_um**2
        
        # # Apply z diffusion term
        # rics_model /= np.sqrt(1 + 4 * diff_coeff * self.temporal_lag_map / self.psf_size_xy_um**2 / self.psf_aspect_ratio**2)
        
        # Apply scanning term
        rics_model *= np.exp(- self.spatial_lag_map / (4 * diff_coeff * self.temporal_lag_map + self.psf_size_xy_um**2))
        
        # Add offset
        rics_model += offset
        return rics_model
    def rics_2Ddiff_residual(self,
                        params):
        # Evaluate model
        rics_model = self.rics_2Ddiff_func(params['amplitude'].value,
                                         params['diff_coeff'].value,
                                         params['offset'].value)
        # Get residual
        residual = rics_model - self.data_RICS
        
        return residual
    def rics_2Ddiff_cost(self,
                   params):
        # Get residual
        residual = self.rics_2Ddiff_residual(params)
        
        # Zero the center pixel residual - that one is not considered in fitting!
        residual[self.center_slow_ax, self.center_fast_ax] = 0.
        
        # Average to get cost
        cost = np.sum(residual**2)
        
        return cost  
    def run_2Ddiff_fit(self):
       # Initial parameters
       init_params = lmfit.Parameters()
       init_params.add(name = 'amplitude',
                       value = 1.,
                       vary = True,
                       min = 0.,
                       max = np.max(self.data_RICS) * 1.5)
       init_params.add(name = 'diff_coeff',
                       value = 1.,
                       vary = True,
                       min = 1e-3,
                       max = 1e4)
       init_params.add(name = 'offset',
                       value = 0.,
                       vary = True,
                       min = -np.max(self.data_RICS) / 10.,
                       max = np.max(self.data_RICS) / 10.)
       
       # Run fit
       result = lmfit.minimize(self.rics_2Ddiff_cost,
                               params = init_params, 
                               method = 'basinhopping')
       
       fit_params = result.params
       model = self.rics_2Ddiff_func(fit_params['amplitude'].value,
                                   fit_params['diff_coeff'].value,
                                   fit_params['offset'].value)
       residual = self.rics_2Ddiff_residual(fit_params)
       center_y = residual.shape[0] // 2
       center_x = residual.shape[1] // 2
       residual[center_y, center_x] = 0.0
       
       return fit_params, model, residual

    def rics_2comp2Ddiff_func(self,
                    N1,
                    N2,
                    fact,
                    diff_coeff_2,
                    offset):
     
        # Initialize map
        rics_model = np.ones_like(self.data_RICS,
                                  dtype = np.float64)
        term_1 = np.ones_like(self.data_RICS,
                                  dtype = np.float64)
        term_2 = np.ones_like(self.data_RICS,
                                  dtype = np.float64)

        # Apply amplitude 
        rics_model*= 0.5/(N1+N2)**2


        # the two terms for 2 components
        term_1 *= N1
        term_1 /= 1 + 4 * fact*diff_coeff_2 * self.temporal_lag_map / self.psf_size_xy_um**2
        term_1 *= np.exp(- self.spatial_lag_map / (4 * fact*diff_coeff_2 * self.temporal_lag_map + self.psf_size_xy_um**2))

        term_2 *= N2
        term_2 /= 1 + 4 * diff_coeff_2 * self.temporal_lag_map / self.psf_size_xy_um**2
        term_2 *= np.exp(- self.spatial_lag_map / (4 * diff_coeff_2 * self.temporal_lag_map + self.psf_size_xy_um**2))


        # Apply xy diffusion term
        rics_model *= (term_1 + term_2)
        
       
        # Add offset
        rics_model += offset
        return rics_model
    def rics_2comp2Ddiff_residual(self,
                        params):
        # Evaluate model
        rics_model = self.rics_2comp2Ddiff_func(params['N1'].value,
                                         params['N2'].value,
                                         params['fact'].value,
                                         params['diff_coeff2'].value,
                                         params['offset'].value)
        # Get residual
        residual = rics_model - self.data_RICS
        
        return residual
    def rics_2comp2Ddiff_cost(self,
                   params):
        # Get residual
        residual = self.rics_2comp2Ddiff_residual(params)
        
        # Zero the center pixel residual - that one is not considered in fitting!
        residual[self.center_slow_ax, self.center_fast_ax] = 0.
        
        # Average to get cost
        cost = np.sum(residual**2)
        
        return cost  
    def run_2comp2Ddiff_fit(self):
       # Initial parameters
       init_params = lmfit.Parameters()
       init_params.add(name = 'N1',
                       value = 1.,
                       vary = True,
                       min = 0.,
                       max = np.max(self.data_RICS) * 1.5)
       init_params.add(name = 'N2',
                       value = 1.,
                       vary = True,
                       min = 0.,
                       max = np.max(self.data_RICS) * 1.5)
       init_params.add(name = 'fact',
                       value = 1.,
                       vary = True,
                       min = 1,
                       max = 1e4)
       init_params.add(name = 'diff_coeff2',
                       value = 1.,
                       vary = True,
                       min = 1e-3,
                       max = 1e4)
       init_params.add(name = 'offset',
                       value = 0.,
                       vary = True,
                       min = -np.max(self.data_RICS) / 10.,
                       max = np.max(self.data_RICS) / 10.)
       
       # Run fit
       result = lmfit.minimize(self.rics_2comp2Ddiff_cost,
                               params = init_params, 
                               method = 'basinhopping')
       
       fit_params = result.params
       model = self.rics_2comp2Ddiff_func(fit_params['N1'].value,
                                   fit_params['N2'].value,
                                   fit_params['fact'],
                                   fit_params['diff_coeff2'].value,
                                   fit_params['offset'].value)
       residual = self.rics_2comp2Ddiff_residual(fit_params)
       center_y = residual.shape[0] // 2
       center_x = residual.shape[1] // 2
       residual[center_y, center_x] = 0.0
       
       return fit_params, model, residual

    def fast_axis_diff_func(self, amplitude, diff_coeff, offset):
        # Initialize map
        rics_model = np.ones_like(self.data_RICS_1D,
                                  dtype = np.float64)
        # Apply amplitude
        rics_model*= amplitude
        
        # Apply xy diffusion term
        rics_model /= (1 + 4 * diff_coeff * self.temporal_lag_map_1D / self.psf_size_xy_um**2)
        
        # # Apply z diffusion term
        # rics_model /= np.sqrt(1 + 4 * diff_coeff * self.temporal_lag_map / self.psf_size_xy_um**2 / self.psf_aspect_ratio**2)
        
        # Apply scanning term
        rics_model *= np.exp(- self.spatial_lag_map_1D / (4 * diff_coeff * self.temporal_lag_map_1D + self.psf_size_xy_um**2))
        
        # Add offset
        rics_model += offset
        return rics_model
    def fast_axis_diff_residual(self,
                        params):
        # Evaluate model
        rics_model = self.fast_axis_diff_func(params['amplitude'].value,
                                         params['diff_coeff'].value,
                                         params['offset'].value)
        # Get residual
        residual = rics_model - self.data_RICS_1D

        return residual
    def fast_axis_diff_cost(self,
                   params):
        # Get residual
        residual = self.fast_axis_diff_residual(params)
        
        # Zero the center pixel residual - that one is not considered in fitting!
        residual[self.center_fast_ax] = 0.
        
        # Average to get cost
        cost = np.sum(residual**2)
        
        return cost 
    def fast_axis_diff_fit(self):
       # Initial parameters
       init_params = lmfit.Parameters()
       init_params.add(name = 'amplitude',
                       value = 1.,
                       vary = True,
                       min = 0.,
                       max = np.max(self.data_RICS_1D) * 1.5)
       init_params.add(name = 'diff_coeff',
                       value = 1.,
                       vary = True,
                       min = 1E-1,
                       max = 1e4)
       init_params.add(name = 'offset',
                       value = 0.,
                       vary = True,
                       min = -np.max(self.data_RICS_1D) / 10.,
                       max = np.max(self.data_RICS_1D) / 10.)
       
       # Run fit
       result = lmfit.minimize(self.fast_axis_diff_cost,
                               params = init_params, 
                               method = 'basinhopping')
       
       fit_params = result.params
       model = self.fast_axis_diff_func(fit_params['amplitude'].value,
                                   fit_params['diff_coeff'].value,
                                   fit_params['offset'].value)
       residual = self.fast_axis_diff_residual(fit_params)
       center_y_1D = residual.shape[0]//2
       residual[center_y_1D] = 0.0
       return fit_params, model, residual
# Functions   
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

def plot_fitting_workflow(image, model_diff, residual_diff, filepath):
    fig = plt.figure(figsize=(10, 5), facecolor='none')  # Transparent figure background
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])
    X = np.arange(model_diff.shape[1])
    Y = np.arange(model_diff.shape[0])
    X, Y = np.meshgrid(X, Y)
    
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.view_init(elev=20, azim=90)
    img1 = ax1.plot_surface(X, Y, image, cmap='jet', linewidth=0, antialiased=False)
    ax1.set_title("RICS Map")
    ax1.set_xlabel('X (fast axis)')
    ax1.set_ylabel('Y (slow axis)')
    ax1.patch.set_alpha(0)  # Make axis background transparent
    
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.view_init(elev=20, azim=90)
    img2 = ax2.plot_surface(X, Y, model_diff, cmap='jet', linewidth=0, antialiased=False)
    ax2.set_title("RICS Fit")
    ax2.set_xlabel('X (fast axis)')
    ax2.set_ylabel('Y (slow axis)')
    ax2.patch.set_alpha(0)
    
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax3.view_init(elev=20, azim=90)
    img3 = ax3.plot_surface(X, Y, residual_diff, cmap='jet', linewidth=0, antialiased=False)
    ax3.set_title("RICS Residuals")
    ax3.set_xlabel('X (fast axis)')
    ax3.set_ylabel('Y (slow axis)')
    ax3.patch.set_alpha(0)
    
    plt.tight_layout()
    fig.savefig(os.path.splitext(filepath)[0] + '.svg', dpi=300)
    plt.close()


def plot_rics_1D_fit(fast_axis_data, model, residual, px_size_um, filepath=None):
    n_points = len(fast_axis_data)
    center = n_points // 2
    x_lag = (np.arange(n_points) - center) * px_size_um
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1]},
                            facecolor='none')
    
    axs[0].plot(x_lag, fast_axis_data, 'ko-', label='Data (Fast axis)')
    axs[0].plot(x_lag, model, 'r-', label='Fit')
    axs[0].set_ylabel('Autocorrelation G(Δx)')
    axs[0].set_title('RICS 1D Autocorrelation and Fit')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].patch.set_alpha(0)
    
    axs[1].plot(x_lag, residual, 'b.-')
    axs[1].axhline(0, color='k', linestyle='--')
    axs[1].set_xlabel('Pixel lag Δx (μm)')
    axs[1].set_ylabel('Residuals')
    axs[1].set_title('Fit Residuals (Model - Data)')
    axs[1].grid(True)
    axs[1].patch.set_alpha(0)
    
    plt.tight_layout()
    if filepath:
        fig.savefig(os.path.splitext(filepath)[0] + '_1D.svg', dpi=300)
    plt.show()

if __name__ == '__main__':  
    pass