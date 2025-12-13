
#!/usr/bin/env python3
"""
Created on Friday 3rd October 2025

@author: yusufqq

Modular RICS Analysis GUI Application
Imports existing RICS modules and provides a unified interface
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sv_ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('agg')

import tifffile
from pylibCZIrw import czi as pyczi
import multiprocessing
import threading
from threading import Thread
import queue
import tifffile
from tqdm import tqdm
import scipy.ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import simRICS
import export_rics
import rics_fit
import random
import pandas as pd
# Import your existing modules

class ModularRICSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("theatRICS")
        
        self.root.geometry("1400x900")

        # Initialize variables
        self.current_image_stack = None
        self.current_file = None
        self.current_corrected_stack = None
        self.current_rics_map = None
        self.diffusion_map = None
        self.current_sd_map = None
        self.simulated_stack = None
        self.fit_results = None
        self.result_queue = queue.Queue()
        self.progress_queue = None
        # Check if modules loaded successfully
        if not all([simRICS, export_rics, rics_fit]):
            self.show_module_error()
            return

        # Create main interface
        self.setup_gui()

    def show_module_error(self):
        """Show error if modules couldn't be loaded"""
        error_frame = ttk.Frame(self.root)
        error_frame.pack(fill='both', expand=True, padx=20, pady=20)

        error_label = ttk.Label(error_frame, 
                               text="Error: Could not load RICS modules!\n\n"
                                    "Please ensure the following files are in the same directory:\n"
                                    "• simRICS.py\n"
                                    "• export_rics.py\n"  
                                    "• rics_fit.py\n\n"
                                    "Then restart the application.",
                               font=('Arial', 12),
                               foreground='red',
                               justify='center')
        error_label.pack(expand=True)

    def setup_gui(self):
        """Setup the main GUI interface"""
        # Configure grid weights for the root to make widgets expandable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)  # For status bar
        self.root.grid_columnconfigure(0, weight=1)
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky='nsew')

        # Create tabs
        self.create_simulation_tab()
        self.create_rics_export_tab()  
        self.create_fitting_tab()
        self.create_results_tab()


        self.status_var = tk.StringVar()
        self.status_var.set("Ready - All modules loaded successfully")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=1, column=0, sticky='w')
        self.button = ttk.Button(self.root, text="Toggle Dark Mode", command=sv_ttk.toggle_theme)
        self.button.grid(row=1, column=0, sticky='e')

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, sticky='ew')
        
        
     
        # self.status_var = tk.StringVar()
        # self.status_var.set("Ready - All modules loaded successfully")
        # self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        # self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        # # self.progress_var = tk.DoubleVar()
        # # self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        # # self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
        # # self.status_bar.lift()
        # # self.progress_bar.lift()
         

        
    def create_simulation_tab(self):
        """Create the image simulation tab using simRICS module"""
        sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(sim_frame, text="Image Simulation")

        # Parameters frame
        params_frame = ttk.LabelFrame(sim_frame, text="Simulation Parameters", padding=10)
        params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Image parameters
        row = 0
        ttk.Label(params_frame, text="Image Shape (pixels):").grid(row=row, column=0, sticky='w', pady=2)
        self.img_width = tk.StringVar(value="256")
        self.img_height = tk.StringVar(value="256")
        width_frame = ttk.Frame(params_frame)
        width_frame.grid(row=row, column=1, pady=2)
        ttk.Entry(width_frame, textvariable=self.img_height, width=8).pack(side=tk.LEFT)
        ttk.Label(width_frame, text=" x ").pack(side=tk.LEFT)
        ttk.Entry(width_frame, textvariable=self.img_width, width=8).pack(side=tk.LEFT)

        row += 1
        ttk.Label(params_frame, text="Number of cores").grid(row=row, column=0, sticky='w', pady=2)
        self.n_cpu = tk.StringVar(value="4")
        ttk.Entry(params_frame, textvariable=self.n_cpu, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(params_frame, text="Number of frames:").grid(row=row, column=0, sticky='w', pady=2)
        self.n_frames = tk.StringVar(value="25")
        ttk.Entry(params_frame, textvariable=self.n_frames, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(params_frame, text="Pixel dwell time (μs):").grid(row=row, column=0, sticky='w', pady=2)
        self.pixel_dwell = tk.StringVar(value="50")
        ttk.Entry(params_frame, textvariable=self.pixel_dwell, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(params_frame, text="Pixel size (nm):").grid(row=row, column=0, sticky='w', pady=2)
        self.pixel_size = tk.StringVar(value="20")
        ttk.Entry(params_frame, textvariable=self.pixel_size, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(params_frame, text="Brightness (kHz):").grid(row=row, column=0, sticky='w', pady=2)
        self.brightness = tk.StringVar(value="2000")
        ttk.Entry(params_frame, textvariable=self.brightness, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(params_frame, text="Number of particles:").grid(row=row, column=0, sticky='w', pady=2)
        self.n_particles = tk.StringVar(value="250")
        ttk.Entry(params_frame, textvariable=self.n_particles, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(params_frame, text="Diffusion coeff X (μm²/s):").grid(row=row, column=0, sticky='w', pady=2)
        self.diff_x = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=self.diff_x, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(params_frame, text="Diffusion coeff Y (μm²/s):").grid(row=row, column=0, sticky='w', pady=2)
        self.diff_y = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=self.diff_y, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(params_frame, text="Rotation (degrees):").grid(row=row, column=0, sticky='w', pady=2)
        self.rotation = tk.StringVar(value="0")
        ttk.Entry(params_frame, textvariable=self.rotation, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(params_frame, text="Background:").grid(row=row, column=0, sticky='w', pady=2)
        self.background = tk.StringVar(value="0")
        ttk.Entry(params_frame, textvariable=self.background, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(params_frame, text="PSF sigma (pixels):").grid(row=row, column=0, sticky='w', pady=2)
        self.psf_sigma = tk.StringVar(value="5")
        ttk.Entry(params_frame, textvariable=self.psf_sigma, width=15).grid(row=row, column=1, pady=2)

        # Simulation type
        row += 1
        ttk.Label(params_frame, text="Simulation type:").grid(row=row, column=0, sticky='w', pady=2)
        self.sim_type = tk.StringVar(value="isotropic")
        sim_combo = ttk.Combobox(params_frame, textvariable=self.sim_type, 
                                values=["isotropic", "anisotropic", "anisotropic_rotated"],
                                width=12)
        sim_combo.grid(row=row, column=1, pady=2)

        # Output path
        row += 1
        ttk.Label(params_frame, text="Output path:").grid(row=row, column=0, sticky='w', pady=2)
        path_frame = ttk.Frame(params_frame)
        path_frame.grid(row=row, column=1, columnspan=2, pady=2, sticky='ew')
        self.output_path = tk.StringVar(value="./simulation_output.tif")
        ttk.Entry(path_frame, textvariable=self.output_path, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="Browse", command=self.browse_output_path).pack(side=tk.RIGHT)

        # Buttons
        row += 1
        button_frame = ttk.Frame(params_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Existing", command=self.load_simulation).pack(side=tk.LEFT, padx=5)

        # Display frame for simulation
        
        display_frame = ttk.LabelFrame(sim_frame, text="Simulation Preview", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Matplotlib figure for simulation display
        self.sim_fig = Figure(figsize=(6,6), dpi=100, facecolor = 'gray')
        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, display_frame)
        self.sim_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Navigation toolbar
        sim_toolbar = NavigationToolbar2Tk(self.sim_canvas, display_frame)
        sim_toolbar.update()

    def create_rics_export_tab(self):
        """Create the RICS export tab using export_rics module"""
        export_frame = ttk.Frame(self.notebook)
        self.notebook.add(export_frame, text="RICS Export")

        # Parameters frame
        export_params = ttk.LabelFrame(export_frame, text="Export Parameters", padding=10)
        export_params.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        row = 0
        # Input file
        ttk.Label(export_params, text="Input file:").grid(row=row, column=0, sticky='w', pady=2)
        input_frame = ttk.Frame(export_params)
        input_frame.grid(row=row, column=1, columnspan=2, pady=2, sticky='ew')
        self.input_file = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_file, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Browse", command=self.browse_input_file).pack(side=tk.RIGHT)

        row += 1
        ttk.Label(export_params, text="Input folder (batch analysis): ").grid(row=row, column=0, sticky='w', pady=2)
        batch_input_frame = ttk.Frame(export_params)
        batch_input_frame.grid(row=row, column=1, columnspan=2, pady=2, sticky='ew')
        self.batch_input_folder = tk.StringVar()
        ttk.Entry(batch_input_frame, textvariable=self.batch_input_folder, width=25).grid(row=row, column=1, pady=2)
        ttk.Button(batch_input_frame, text="Browse", command=self.browse_batch_input_folder).grid(row=row, column=2, pady=2)

        row += 1
        ttk.Label(export_params, text="Channel to use:").grid(row=row, column=0, sticky='w', pady=2)
        self.channel = tk.StringVar(value=0)
        model_combo = ttk.Combobox(export_params, textvariable=self.channel, 
                                   values=[0,1], width=12)
        model_combo.grid(row=row, column=1, pady=2)
        row += 1
        ttk.Label(export_params, text="Crop factor:").grid(row=row, column=0, sticky='w', pady=2)
        self.crop_factor = tk.StringVar(value="0.5")
        ttk.Entry(export_params, textvariable=self.crop_factor, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(export_params, text="Window size (odd):").grid(row=row, column=0, sticky='w', pady=2)
        self.window_size = tk.StringVar(value="3")
        ttk.Entry(export_params, textvariable=self.window_size, width=15).grid(row=row, column=1, pady=2)

        row += 1
        self.correct_drift = tk.BooleanVar()
        ttk.Checkbutton(export_params, text="Correct drift", variable=self.correct_drift).grid(row=row, column=0, columnspan=2, sticky='w', pady=2)

        # Buttons
        row += 1
        export_button_frame = ttk.Frame(export_params)
        export_button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(export_button_frame, text="Export RICS", command=self.export_rics).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_button_frame, text="Load RICS", command=self.load_rics).pack(side=tk.LEFT, padx=5)

        # Display frame for RICS
        
        rics_display_frame = ttk.LabelFrame(export_frame, text="RICS Maps")
        rics_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Matplotlib figure for RICS display
        self.rics_fig = Figure(figsize=(6,6), dpi=100, facecolor = 'gray')
        self.rics_canvas = FigureCanvasTkAgg(self.rics_fig, rics_display_frame)
        self.rics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Navigation toolbar
        rics_toolbar = NavigationToolbar2Tk(self.rics_canvas, rics_display_frame)
        rics_toolbar.update()

        

    def browse_metadata_file(self):
        filename = filedialog.askopenfilename(
            title="Select metadata CZI file",
            filetypes=[("CZI files", "*.czi"), ("All files", "*.*")]
        )
        if filename:
            self.file_for_metadata.set(filename)
            

    def create_fitting_tab(self):
        """Create the fitting tab using rics_fit module"""
        fit_frame = ttk.Frame(self.notebook)
        self.notebook.add(fit_frame, text="RICS Fitting")

        # Parameters frame
        # Fitting Parameters Frame (top)
        fit_params = ttk.LabelFrame(fit_frame, text="Fitting Parameters", padding=10)
        fit_params.grid(row = 0, column = 0, sticky = "nw", padx = 5, pady = 5)
        # fit_params.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Diffusion Map Fitting Parameters Frame (below)
        diff_fit_params = ttk.LabelFrame(fit_frame, text='Diffusion Map Fitting Parameters', padding=10)
        diff_fit_params.grid(row=1, column=0, sticky="nw", padx=5, pady=5)
        # diff_fit_params.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        fit_frame.grid_columnconfigure(0, weight=0)

        row = 0
        # RICS map file
        ttk.Label(fit_params, text="RICS map file:").grid(row=row, column=0, sticky='w', pady=2)
        rics_input_frame = ttk.Frame(fit_params)
        rics_input_frame.grid(row=row, column=1, columnspan=2, pady=2, sticky='ew')
        self.rics_file = tk.StringVar()
        ttk.Entry(rics_input_frame, textvariable=self.rics_file, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(rics_input_frame, text="Browse", command=self.browse_rics_file).pack(side=tk.RIGHT)
        ttk.Label(diff_fit_params, text="Input file for Diffusion Map").grid(row=row, column=0, sticky='w', pady=2)
        self.input_file_diff_map = tk.StringVar()
        ttk.Entry(diff_fit_params, textvariable=self.input_file_diff_map, width=25).grid(row=row, column=1, pady=2)
        ttk.Button(diff_fit_params, text="Browse", command=self.browse_input_file_diff_map).grid(row=row, column=2, pady=2)

        # parameters from a czi file
        row +=1
        ttk.Label(fit_params, text="Input folder (batch analysis): ").grid(row=row, column=0, sticky='w', pady=2)
        batch_fit_frame = ttk.Frame(fit_params)
        batch_fit_frame.grid(row=row, column=1, columnspan=2, pady=2, sticky='ew')
        self.batch_fit_folder = tk.StringVar()
        ttk.Entry(batch_fit_frame, textvariable=self.batch_fit_folder, width=25).grid(row=row, column=1, pady=2)
        ttk.Button(batch_fit_frame, text="Browse", command=self.browse_batch_fit_folder).grid(row=row, column=2, pady=2)
        ttk.Label(diff_fit_params, text="Window Size (pixels):").grid(row=row, column=0, sticky='w', pady=2)
        self.window_size_diff_map = tk.StringVar(value="32")
        ttk.Entry(diff_fit_params, textvariable=self.window_size_diff_map, width=15).grid(row=row, column=1, pady=2)

        row+=1
        ttk.Label(fit_params, text="Results file:").grid(row=row, column=0, sticky='w', pady=2)
        self.saving_path = tk.StringVar(value="./results_csv.csv")
        ttk.Entry(fit_params, textvariable=self.saving_path, width=25).grid(row=row, column=1, pady=2)
        ttk.Button(fit_params, text="Browse", command=self.browse_save_path).grid(row=row, column=2, pady=2)


        # Microscope parameters
        row += 1
        ttk.Label(fit_params, text="Pixel size (nm):").grid(row=row, column=0, sticky='w', pady=2)
        self.fit_pixel_size = tk.StringVar(value="20")
        ttk.Entry(fit_params, textvariable=self.fit_pixel_size, width=15).grid(row=row, column=1, pady=2)
        ttk.Label(diff_fit_params, text="Offset (pixels):").grid(row=row, column=0, sticky='w', pady=2)
        self.offset_diff_map = tk.StringVar(value="16")
        ttk.Entry(diff_fit_params, textvariable=self.offset_diff_map, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(fit_params, text="Pixel dwell (μs):").grid(row=row, column=0, sticky='w', pady=2)
        self.fit_pixel_dwell = tk.StringVar(value="50")
        ttk.Entry(fit_params, textvariable=self.fit_pixel_dwell, width=15).grid(row=row, column=1, pady=2)
        ttk.Label(diff_fit_params, text="Channel").grid(row=row, column=0, sticky='w', pady=2)
        self.channel_to_use_diff_map = tk.StringVar(value=0)
        model_combo = ttk.Combobox(diff_fit_params, textvariable=self.channel_to_use_diff_map, 
                                   values=[0,1], width=12)
        model_combo.grid(row=row, column=1, pady=2)
        
        row += 1
        ttk.Label(fit_params, text="Line time (ms):").grid(row=row, column=0, sticky='w', pady=2)
        self.fit_line_time = tk.StringVar(value="12.8")
        ttk.Entry(fit_params, textvariable=self.fit_line_time, width=15).grid(row=row, column=1, pady=2)
        diff_fit_button_frame = ttk.Frame(diff_fit_params)
        diff_fit_button_frame.grid(row=row, column=0, columnspan=1, pady=10)
        ttk.Button(diff_fit_button_frame, text="Generate Diffusion Map", command=self.run_diffusion_map).pack(side=tk.LEFT, padx=5)

        row += 1
        ttk.Label(fit_params, text="PSF size XY (μm):").grid(row=row, column=0, sticky='w', pady=2)
        self.fit_psf_xy = tk.StringVar(value="0.2")
        ttk.Entry(fit_params, textvariable=self.fit_psf_xy, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(fit_params, text="PSF aspect ratio:").grid(row=row, column=0, sticky='w', pady=2)
        self.fit_psf_aspect = tk.StringVar(value="4.985423166")
        ttk.Entry(fit_params, textvariable=self.fit_psf_aspect, width=15).grid(row=row, column=1, pady=2)

        # Crop factors for fitting
        row += 1
        ttk.Label(fit_params, text="Crop factor fast:").grid(row=row, column=0, sticky='w', pady=2)
        self.fit_crop_fast = tk.StringVar(value="0.5")
        ttk.Entry(fit_params, textvariable=self.fit_crop_fast, width=15).grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(fit_params, text="Crop factor slow:").grid(row=row, column=0, sticky='w', pady=2)
        self.fit_crop_slow = tk.StringVar(value="0.5")
        ttk.Entry(fit_params, textvariable=self.fit_crop_slow, width=15).grid(row=row, column=1, pady=2)

        # Diffusion model
        row += 1
        ttk.Label(fit_params, text="Diffusion model:").grid(row=row, column=0, sticky='w', pady=2)
        self.diffusion_model = tk.StringVar(value="2Ddiff")
        model_combo = ttk.Combobox(fit_params, textvariable=self.diffusion_model, 
                                   values=["2Ddiff", "3Ddiff"], width=12)
        model_combo.grid(row=row, column=1, pady=2)

        row+=1
        ttk.Label(fit_params, text="Channel").grid(row=row, column=0, sticky='w', pady=2)
        self.channel_to_use = tk.StringVar(value=0)
        model_combo = ttk.Combobox(fit_params, textvariable=self.channel_to_use, 
                                   values=[0,1], width=12)
        model_combo.grid(row=row, column=1, pady=2)
        

        # Fitting buttons
        row += 1
        fit_button_frame = ttk.Frame(fit_params)
        fit_button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        ttk.Button(fit_button_frame, text="Run 2D/3D Fitting", command=self.run_fitting).pack(side=tk.LEFT, padx=5)
        self.fit_1d_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fit_button_frame, text="1D Fast Axis Fit", variable=self.fit_1d_var).pack(side=tk.LEFT, padx=5)        
        

        
        # Display frame for fitting results
        
        fit_display_frame = ttk.LabelFrame(fit_frame, text="Fitting Results", padding=10)
        fit_display_frame.grid(row=0, column=1, rowspan=2, sticky="ne", padx=5, pady=5)

        # fit_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        fit_frame.grid_columnconfigure(1, weight=1)
        fit_frame.grid_rowconfigure(0, weight=1)
        fit_frame.grid_rowconfigure(1, weight=1)
        # Matplotlib figure for fitting display
        self.fit_fig = Figure(figsize=(10,10), dpi=100, facecolor = 'gray')
        self.fit_canvas = FigureCanvasTkAgg(self.fit_fig, fit_display_frame)
        toolbar_frame = ttk.LabelFrame(fit_frame, padding=10)
        toolbar_frame.grid(row=1, column=1, sticky="sw", padx=5, pady=5)
        self.fit_canvas.get_tk_widget().pack(side = tk.BOTTOM,fill=tk.X, expand=True)
        # toolbar_frame.grid_columnconfigure(1, weight=0)
        # toolbar_frame.grid_rowconfigure(0, weight=0)
        # Navigation toolbar
        fit_toolbar = NavigationToolbar2Tk(self.fit_canvas, toolbar_frame)
        fit_toolbar.update()
        

    def create_results_tab(self):
        """Create the results and log tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results & Logs")

        # Results text area
        results_label_frame = ttk.LabelFrame(results_frame, text="Analysis Results & Logs", padding=10)
        results_label_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_label_frame, height=25, width=100, font=('Consolas', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Buttons for results
        results_button_frame = ttk.Frame(results_label_frame)
        results_button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(results_button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_button_frame, text="Export All Plots", command=self.export_plots).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_button_frame, text="Save Session", command=self.save_session).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_button_frame, text="Load Session", command=self.load_session).pack(side=tk.LEFT, padx=5)

    def log_message(self, message):
        """Add a message to the log with timestamp"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        """Clear the log"""
        self.results_text.delete(1.0, tk.END)

    def browse_output_path(self):
        """Browse for output path"""
        filename = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)

    def browse_input_file(self):
        """Browse for input file"""
        filename = filedialog.askopenfilename(
            title="Select input image stack",
            filetypes=[("All files", "*.*"), ("CZI files", "*.czi"),("TIFF files", "*.tif") ]
        )
        if filename:
            self.input_file.set(filename)
    def browse_save_path(self):
        filename = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.saving_path.set(filename)
    def browse_batch_input_folder(self):
        filepath = filedialog.askdirectory(
            title="Select directory for batch input",
        )
        if filepath:
            self.batch_input_folder.set(filepath)

    def browse_batch_fit_folder(self):
        filepath = filedialog.askdirectory(
            title="Select directory for batch input",
        )
        if filepath:
            self.batch_fit_folder.set(filepath)

    def browse_input_file_diff_map(self):
        """Browse for input file"""
        filename = filedialog.askopenfilename(
            title="Select input image stack",
            filetypes=[("All files", "*.*"), ("CZI files", "*.czi"),("TIFF files", "*.tif") ]
        )
        if filename:
            self.input_file_diff_map.set(filename)
    def browse_rics_file(self):
        """Browse for RICS map file"""
        filename = filedialog.askopenfilename(
            title="Select RICS map file",
            filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
        )
        if filename:
            self.rics_file.set(filename)
    def parallel_rics_raster(
        self,
        img_shape,
        n_frames,
        pixel_dwell_time_us,      # μs
        pixel_size_nm,             # nm
        brightness_khz,          # kHz per molecule
        avg_n_particles,
        diffusion_um2_s,          # μm²/s
        background,
        psf_sigma_px,
        output_path,
        cpu_n
    ):
        n_particles = avg_n_particles

        pool = multiprocessing.Pool(processes=cpu_n)
        print('Cores used: '+str(cpu_n))
        
        # Prepare arguments for each frame, give each a random seed to avoid identical results
        frame_args = [
            (
                idx,
                img_shape,
                pixel_dwell_time_us,
                pixel_size_nm,
                brightness_khz,
                n_particles,
                diffusion_um2_s,
                background,
                psf_sigma_px,
                np.random.randint(0, 1000000),
            )
            for idx in range(n_frames)
        ]
        stack = []
        
        i = 0
        for result in tqdm(pool.imap(simRICS.simulate_single_frame, frame_args), total=n_frames):
            stack.append(result)
            i+=1
            progress = (i / n_frames) * 100
            self.progress_queue.put(progress)
        pool.close()
        pool.join()

        stack = np.stack(stack, axis=0)
        tifffile.imwrite(output_path, stack, photometric='minisblack')
        print(f"Saved parallel simulated raster scan to {output_path}")
        return stack
        
    def parallel_rics_raster_aniso(
        self,
        img_shape,
        n_frames,
        pixel_dwell_time_us,      # μs
        pixel_size_nm,             # nm
        brightness_khz,          # kHz per molecule
        avg_n_particles,
        diffusion_um2_s_x,          # μm²/s
        diffusion_um2_s_y,          # μm²/s   
        background,
        psf_sigma_px,
        output_path,
        cpu_n
    ):
        n_particles = avg_n_particles
        
        pool = multiprocessing.Pool(processes=cpu_n)

        # Prepare arguments for each frame, give each a random seed to avoid identical results
        frame_args = [
            (
                idx,
                img_shape,
                pixel_dwell_time_us,
                pixel_size_nm,
                brightness_khz,
                n_particles,
                diffusion_um2_s_x,
                diffusion_um2_s_y,
                background,
                psf_sigma_px,
                np.random.randint(0, 1000000),
            )
            for idx in range(n_frames)
        ]
        stack = []
        for result in tqdm(pool.imap(simRICS.simulate_single_frame_aniso, frame_args), total=n_frames):
            stack.append(result)
            
            i+=1
            progress = (i / n_frames) * 100
            self.progress_queue.put(progress)
        pool.close()
        pool.join()

        stack = np.stack(stack, axis=0)
        tifffile.imwrite(output_path, stack, photometric='minisblack')
        print(f"Saved parallel simulated raster scan to {output_path}")
        return stack

    def parallel_rics_raster_aniso_rotated(
        self,
        img_shape,
        n_frames,
        pixel_dwell_time_us,      # μs
        pixel_size_nm,             # nm
        brightness_khz,          # kHz per molecule
        avg_n_particles,
        diffusion_um2_s_x,          # μm²/s
        diffusion_um2_s_y,          # μm²/s   
        rotation_deg,
        background,
        psf_sigma_px,
        output_path,
        cpu_n
    ):
        n_particles = avg_n_particles
        
        pool = multiprocessing.Pool(processes=cpu_n)
        params = dict(
            
            img_shape = img_shape,
            n_frames = n_frames,
            pixel_dwell_time_us = pixel_dwell_time_us,
            pixel_size_nm = pixel_size_nm,
            brightness_khz = brightness_khz,
            avg_n_particles = n_particles,
            diffusion_um2_s_x = diffusion_um2_s_x,
            diffusion_um2_s_y = diffusion_um2_s_y,
            rotation_deg = rotation_deg,
            background = background,
            psf_sigma_px = psf_sigma_px,
            output_path = output_path
            )
        txt_path = os.path.splitext(output_path)[0] + ".txt"
        with open(txt_path, 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
        print(f"Simulation parameters written to {txt_path}")
        # Prepare arguments for each frame, give each a random seed to avoid identical results
        frame_args = [
            (
                idx,
                img_shape,
                pixel_dwell_time_us,
                pixel_size_nm,
                brightness_khz,
                n_particles,
                diffusion_um2_s_x,
                diffusion_um2_s_y,
                rotation_deg,
                background,
                psf_sigma_px,
                np.random.randint(0, 1000000),
            )
        

            for idx in range(n_frames)
        ]
        stack = []
        for result in tqdm(pool.imap(simRICS.simulate_single_frame_aniso_rotated, frame_args), total=n_frames):
            stack.append(result)
            
            i+=1
            progress = (i / n_frames) * 100
            self.progress_queue.put(progress)
        pool.close()
        pool.join()

        stack = np.stack(stack, axis=0)
        tifffile.imwrite(output_path, stack, photometric='minisblack')
        print(f"Saved parallel simulated raster scan to {output_path}")
        return stack
    def run_simulation(self):
        """Run image simulation using simRICS module"""
        if not simRICS:
            messagebox.showerror("Error", "Simulation module not loaded!")
            return
    
        self.log_message("Starting simulation...")
        self.status_var.set("Running simulation...")
        # Show progress bar at simulation start
        self.status_bar.update_idletasks()  # Force redraw

        self.progress_queue = multiprocessing.Queue()
        # self.progress_queue.put_nowait(0)
        # print(self.progress_queue.get_nowait())
        def worker():
            try:
                # print("Inside simulation worker")  # Confirm worker start
                sim_type = self.sim_type.get()
                img_shape = (int(self.img_height.get()), int(self.img_width.get()))
                n_frames = int(self.n_frames.get())
                pixel_dwell_time_us = float(self.pixel_dwell.get())
                pixel_size_nm = float(self.pixel_size.get())
                brightness_khz = float(self.brightness.get())
                avg_n_particles = int(self.n_particles.get())
                diffusion_um2_s_x = float(self.diff_x.get())
                diffusion_um2_s_y = float(self.diff_y.get())
                rotation_deg = float(self.rotation.get()) if hasattr(self, 'rotation') else 0  # For rotated sim
                background = float(self.background.get())
                psf_sigma_px = float(self.psf_sigma.get())
                output_path = self.output_path.get()
                n_cpu = int(self.n_cpu.get())
                if n_cpu >= multiprocessing.cpu_count():
                    n_cpu = int(0.8 * int(multiprocessing.cpu_count()))
                else:
                    pass
                if sim_type == "isotropic":
                    avg_diffusion = (diffusion_um2_s_x + diffusion_um2_s_y) / 2
                    self.simulated_stack = self.parallel_rics_raster(
                        img_shape,
                        n_frames,
                        pixel_dwell_time_us,
                        pixel_size_nm,
                        brightness_khz,
                        avg_n_particles,
                        avg_diffusion,
                        background,
                        psf_sigma_px,
                        output_path,
                        n_cpu
                        
                    )
                elif sim_type == "anisotropic":
                    self.simulated_stack = self.parallel_rics_raster_aniso(
                        img_shape,
                        n_frames,
                        pixel_dwell_time_us,
                        pixel_size_nm,
                        brightness_khz,
                        avg_n_particles,
                        diffusion_um2_s_x,
                        diffusion_um2_s_y,
                        background,
                        psf_sigma_px,
                        output_path,
                        n_cpu
                        
                    )
                elif sim_type == "anisotropic_rotated":
                    self.simulated_stack = self.parallel_rics_raster_aniso_rotated(
                        img_shape,
                        n_frames,
                        pixel_dwell_time_us,
                        pixel_size_nm,
                        brightness_khz,
                        avg_n_particles,
                        diffusion_um2_s_x,
                        diffusion_um2_s_y,
                        rotation_deg,
                        background,
                        psf_sigma_px,
                        output_path,
                        n_cpu
                        
                    )
                else:
                    raise ValueError(f"Unknown simulation type: {sim_type}")
                # print("Simulation worker finished successfully")
                self.root.after(0, self.update_simulation_display)
                self.root.after(0, lambda: self.status_var.set("Simulation completed"))
                self.root.after(0, lambda:self.log_message("Simulation completed"))
                self.root.after(0, lambda: self.progress_bar.pack_forget())
                self.status_bar.update_idletasks()

            except Exception as e:
                import traceback
                self.log_message(f"Exception in simulation worker: {str(e)}")
                self.log_message(traceback.format_exc())
                self.root.after(0, lambda: self.status_var.set("Error"))
                self.root.after(0, lambda: self.progress_bar.pack_forget())  # Hide progress bar on error
            
        threading.Thread(target=worker, daemon=True).start()
        # Start polling the progress queue in a separate thread to keep GUI responsive
        def poll_queue():
            progress = 0
            while progress<=100:
                try:
                    progress = self.progress_queue.get(block = False)
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    self.root.after(0, lambda: self.progress_bar.update_idletasks())
                    if progress == 100:
                        self.root.after(0, lambda: self.progress_bar.pack_forget())
                        self.root.after(0, lambda: self.progress_bar.update_idletasks())
                        break
                except:
                    pass

        threading.Thread(target=poll_queue, daemon=True).start()
        

        

  
                
    def update_simulation_display(self):
        """Update the simulation display with multiple views"""
        if self.simulated_stack is not None:
            self.sim_fig.clear()

            # Create subplots
            gs = gridspec.GridSpec(2, 2, figure=self.sim_fig)

            # First frame
            ax1 = self.sim_fig.add_subplot(gs[0, 0])
            ax1.imshow(self.simulated_stack[0], cmap='gray')
            ax1.set_title('First Frame')
            ax1.axis('off')

            # Last frame
            ax2 = self.sim_fig.add_subplot(gs[0, 1])
            ax2.imshow(self.simulated_stack[-1], cmap='gray')
            ax2.set_title('Last Frame')
            ax2.axis('off')

            # Mean projection
            ax3 = self.sim_fig.add_subplot(gs[1, 0])
            mean_img = np.mean(self.simulated_stack, axis=0)
            ax3.imshow(mean_img, cmap='gray')
            ax3.set_title('Time-averaged Image')
            ax3.axis('off')

            # Intensity over time at central region of the image
            # Calculate crop boundaries (0.25 means the crop covers 25% of the image size)
            crop_fraction = 0.25
            ny, nx = self.simulated_stack.shape[1], self.simulated_stack.shape[2]
            crop_ny = int(crop_fraction * ny)
            crop_nx = int(crop_fraction * nx)
            
            # Determine the starting and ending indices of the crop centered in the image
            start_y = (ny - crop_ny) // 2
            end_y = start_y + crop_ny
            start_x = (nx - crop_nx) // 2
            end_x = start_x + crop_nx
            
            # Extract the crop for all frames
            cropped_region = self.simulated_stack[:, start_y:end_y, start_x:end_x]
            
            # Calculate the average intensity over the crop for each frame
            intensity_trace = cropped_region.mean(axis=(1, 2))
            
            # Plot the intensity trace
            ax4 = self.sim_fig.add_subplot(gs[1, 1])
            ax4.plot(intensity_trace, 'b-')
            ax4.set_title('Average Intensity vs Frame (Centered Crop)')
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Average Intensity')
            ax4.grid(True)


            self.sim_canvas.draw()

    def load_simulation(self):
        """Load existing simulation"""
        filename = filedialog.askopenfilename(
            title="Select simulation file",
            filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.simulated_stack = tifffile.imread(filename)
                self.log_message(f"Loaded simulation from {filename}")
                self.log_message(f"Stack shape: {self.simulated_stack.shape}")
                self.update_simulation_display()
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {str(e)}")

    def export_rics(self):
        """Export RICS map using export_rics module"""
        if not export_rics:
            messagebox.showerror("Error", "Export module not loaded!")
            return

        if not self.input_file.get() and not self.batch_input_folder.get():
            messagebox.showwarning("Warning", "Please select an input file first or select a folder for batch processing")
            return


        self.log_message("Starting RICS export...")
        self.status_var.set("Exporting RICS...")

        # Create a thread for RICS export
        export_thread = Thread(target=self._export_rics_thread)
        export_thread.daemon = True
        export_thread.start()



    def _export_rics_thread(self):
        """Thread function for RICS export using your export_rics module"""
        if not self.input_file.get():
            files = get_files_from_folder(self.batch_input_folder.get(), '.czi', '')
            for file in files:
                try:
                    input_file = file
                    channel = int(self.channel.get())
                    crop_factor = float(self.crop_factor.get())
                    window_size = int(self.window_size.get())

                    self.log_message(f"Processing file: {input_file}")
                    self.log_message(f"Channel: {channel}, Crop factor: {crop_factor}, Window size: {window_size}")

                    # # Temporarily modify the module's global variables
                    # original_channel = export_rics.channel_to_use
                    # original_crop = export_rics.crop_factor
                    # original_window = export_rics.window_size
                    # original_drift = export_rics.correct_drift

                    export_rics.channel_to_use = channel
                    export_rics.crop_factor = crop_factor
                    export_rics.window_size = window_size
                    export_rics.correct_drift = self.correct_drift.get()

                    # Process the file using your module's function
                    ext = os.path.splitext(input_file)[1].lower()

                    if ext == '.czi':
                        from pylibCZIrw import czi as pyczi
                        with pyczi.open_czi(input_file) as czidoc:
                            total_bounding_box = czidoc.total_bounding_box
                            n_frames = total_bounding_box['T'][1]

                        RICS_map, sd_map, all_frames, corrected_stack = export_rics.process_all_frames_czi(
                            input_file, n_frames, channel, window_size
                        )
                    elif ext in ['.tif', '.tiff']:
                        stack = tifffile.imread(input_file)
                        cropped_img = export_rics.crop_center(stack, crop_factor=crop_factor)
                        n_frames = cropped_img.shape[0]

                        if self.correct_drift.get():
                            cropped_img = export_rics.drift_correct(cropped_img)

                        RICS_map, sd_map, all_frames, corrected_stack = export_rics.process_all_frames_tiff(
                            cropped_img, n_frames, channel, window_size
                        )
                    else:
                        raise ValueError(f"Unsupported file format: {ext}")

                    # Store results
                    self.current_rics_map = RICS_map
                    self.current_sd_map = sd_map
                    self.current_image_stack = all_frames
                    self.current_corrected_stack = corrected_stack
                    # Save RICS maps
                    rics_output = os.path.splitext(input_file)[0] + '_RICScorr.tif'
                    sd_output = os.path.splitext(input_file)[0] + '_RICSunc.tif'

                    tifffile.imwrite(rics_output, RICS_map, photometric='minisblack')
                    tifffile.imwrite(sd_output, sd_map, photometric='minisblack')

                    self.log_message(f"RICS map saved to: {rics_output}")
                    self.log_message(f"Standard deviation map saved to: {sd_output}")
                    self.log_message(f"RICS map shape: {RICS_map.shape}")

                    # # Restore original values
                    # export_rics.channel_to_use = original_channel
                    # export_rics.crop_factor = original_crop
                    # export_rics.window_size = original_window
                    # export_rics.correct_drift = original_drift

                    # Update display
                    self.root.after(0, self.update_rics_display)

                except Exception as e:
                    self.log_message(f"RICS export error: {str(e)}")
                    import traceback
                    self.log_message(f"Traceback: {traceback.format_exc()}")
                finally:
                    self.root.after(0, lambda: self.status_var.set("Ready"))
        elif self.input_file.get():
            try:
                    input_file = self.input_file.get()
                    channel = int(self.channel.get())
                    crop_factor = float(self.crop_factor.get())
                    window_size = int(self.window_size.get())

                    self.log_message(f"Processing file: {input_file}")
                    self.log_message(f"Channel: {channel}, Crop factor: {crop_factor}, Window size: {window_size}")

                    # # Temporarily modify the module's global variables
                    # original_channel = export_rics.channel_to_use
                    # original_crop = export_rics.crop_factor
                    # original_window = export_rics.window_size
                    # original_drift = export_rics.correct_drift

                    export_rics.channel_to_use = channel
                    export_rics.crop_factor = crop_factor
                    export_rics.window_size = window_size
                    export_rics.correct_drift = self.correct_drift.get()

                    # Process the file using your module's function
                    ext = os.path.splitext(input_file)[1].lower()

                    if ext == '.czi':
                        from pylibCZIrw import czi as pyczi
                        with pyczi.open_czi(input_file) as czidoc:
                            total_bounding_box = czidoc.total_bounding_box
                            n_frames = total_bounding_box['T'][1]

                        RICS_map, sd_map, all_frames, corrected_stack = export_rics.process_all_frames_czi(
                            input_file, n_frames, channel, window_size
                        )
                    elif ext in ['.tif', '.tiff']:
                        stack = tifffile.imread(input_file)
                        cropped_img = export_rics.crop_center(stack, crop_factor=crop_factor)
                        n_frames = cropped_img.shape[0]

                        if self.correct_drift.get():
                            cropped_img = export_rics.drift_correct(cropped_img)

                        RICS_map, sd_map, all_frames, corrected_stack = export_rics.process_all_frames_tiff(
                            cropped_img, n_frames, channel, window_size
                        )
                    else:
                        raise ValueError(f"Unsupported file format: {ext}")

                    # Store results
                    self.current_rics_map = RICS_map
                    self.current_sd_map = sd_map
                    self.current_image_stack = all_frames
                    self.current_corrected_stack = corrected_stack
                    # Save RICS maps
                    rics_output = os.path.splitext(input_file)[0] + '_RICScorr.tif'
                    sd_output = os.path.splitext(input_file)[0] + '_RICSunc.tif'

                    tifffile.imwrite(rics_output, RICS_map, photometric='minisblack')
                    tifffile.imwrite(sd_output, sd_map, photometric='minisblack')

                    self.log_message(f"RICS map saved to: {rics_output}")
                    self.log_message(f"Standard deviation map saved to: {sd_output}")
                    self.log_message(f"RICS map shape: {RICS_map.shape}")

                    # # Restore original values
                    # export_rics.channel_to_use = original_channel
                    # export_rics.crop_factor = original_crop
                    # export_rics.window_size = original_window
                    # export_rics.correct_drift = original_drift

                    # Update display
                    self.root.after(0, self.update_rics_display)

            except Exception as e:
                self.log_message(f"RICS export error: {str(e)}")
                import traceback
                self.log_message(f"Traceback: {traceback.format_exc()}")
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))


    
        

    def update_rics_display(self):
        """Update the RICS display using your plotting function"""


        if self.current_rics_map is not None:
            self.rics_fig.clear()

            # Use your existing plotting workflow
            try:
                export_rics.plot_rics_workflow(
                    self.current_image_stack, 
                    self.current_corrected_stack,  
                    self.current_rics_map, 
                    self.current_sd_map, 
                    "gui_display"
                )
                # The plot_rics_workflow function creates its own figure, so we need to recreate for our canvas
            except:
                pass

            center_y = self.current_rics_map.shape[0] // 2
            center_x = self.current_rics_map.shape[1] // 2
            self.current_rics_map[center_y, center_x] = 0.0
            self.current_sd_map[center_y, center_x] = 0.0


            # crop_factor_fast_ax = 0.5
            # crop_factor_slow_ax = 0.5
            # floor_fast_ax = int(np.floor(self.current_rics_map.shape[1] * (1 - crop_factor_fast_ax) * 0.5))
            # ceil_fast_ax = int(np.floor(self.current_rics_map.shape[1] * 0.5 * (1 + crop_factor_fast_ax)))
            # floor_slow_ax = int(np.floor(self.current_rics_map.shape[0] * (1 - crop_factor_slow_ax) * 0.5))
            # ceil_slow_ax = int(np.floor(self.current_rics_map.shape[0] * 0.5 * (1 + crop_factor_slow_ax)))
            # self.cropped_rics_map = self.current_rics_map[floor_slow_ax:ceil_slow_ax,
            #                     floor_fast_ax:ceil_fast_ax] 
            # self.cropped_sd_map = self.current_sd_map[floor_slow_ax:ceil_slow_ax,
            #                     floor_fast_ax:ceil_fast_ax]
            
            # Create our own display
            gs = gridspec.GridSpec(2, 3, figure=self.rics_fig, width_ratios=[1, 1, 2])

            # Raw image
            if self.current_image_stack is not None:
                ax1 = self.rics_fig.add_subplot(gs[0, 0])
                ax1.imshow(self.current_image_stack[0], cmap='gray')
                ax1.set_title("Raw Image (Frame 0)")
                ax1.axis('off')
            # Corrected image
            if self.current_corrected_stack is not None:
                ax2 = self.rics_fig.add_subplot(gs[0, 1])
                ax2.imshow(self.current_corrected_stack[0], cmap='gray')
                ax2.set_title("Corrected Image (Frame 0)")
                ax2.axis('off')

            # RICS map
            ax3 = self.rics_fig.add_subplot(gs[1, 0])
            im3 = ax3.imshow(self.current_rics_map, cmap='jet')
            ax3.set_title("RICS Map")
            ax3.axis('off')
            self.rics_fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # Standard deviation map
            if self.current_sd_map is not None:
                ax4 = self.rics_fig.add_subplot(gs[1, 1])
                im4 = ax4.imshow(self.current_sd_map, cmap='jet')
                ax4.set_title("Uncertainty Map")
                ax4.axis('off')
                self.rics_fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

            # 3D view
            ax5 = self.rics_fig.add_subplot(gs[:, 2], projection='3d')
            X = np.arange(self.current_rics_map.shape[1])
            Y = np.arange(self.current_rics_map.shape[0])
            X, Y = np.meshgrid(X, Y)
            ax5.plot_surface(X, Y, self.current_rics_map, cmap='jet', alpha=0.8)
            ax5.set_title('RICS Map 3D')
            ax5.view_init(elev=20, azim=90)

            
            
        
            self.rics_canvas.draw()

    def load_rics(self):
        """Load existing RICS map"""
        filename = filedialog.askopenfilename(
            title="Select RICS map file",
            filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.current_rics_map = tifffile.imread(filename)
                self.rics_file.set(filename)
                self.log_message(f"Loaded RICS map from {filename}")
                self.log_message(f"RICS map shape: {self.current_rics_map.shape}")

                # Try to load corresponding uncertainty map
                sd_filename = filename.replace('_RICScorr.tif', '_RICSunc.tif')
                if os.path.exists(sd_filename):
                    self.current_sd_map = tifffile.imread(sd_filename)
                    self.log_message(f"Also loaded uncertainty map: {sd_filename}")

                self.update_rics_display()
            except Exception as e:
                messagebox.showerror("Error", f"Could not load RICS file: {str(e)}")

    def run_fitting(self):
        """Run RICS fitting using rics_fit module"""
        if not rics_fit:
            messagebox.showerror("Error", "Fitting module not loaded!")
            return

        if not self.rics_file.get() and not self.batch_fit_folder.get():
            messagebox.showwarning("Warning", "Please load a RICS map first")
            return

        self.log_message("Starting RICS fitting...")
        self.status_var.set("Running fitting...")

        # Create a thread for fitting
        fit_thread = Thread(target=self._run_fitting_thread)
        fit_thread.daemon = True
        fit_thread.start()

    def _run_fitting_thread(self):
        """Thread function for fitting using your rics_fit module"""
        if not self.rics_file.get() and self.batch_fit_folder.get():
            files = get_files_from_folder(self.batch_fit_folder.get(), '.tif', 'RICScorr')
            for file in files:
                self.current_file = file
                try:
                    # Load RICS map if needed
                    
                    import os
                    self.current_rics_map = tifffile.imread(file)
                    self.log_message(f"Loaded RICS map for fitting: {file}")
                    root, ext = os.path.splitext(file)

                    metadata_file = os.path.split(root)[1].split('_')
                    metadata_file = metadata_file[:-1]
                    metadata_file = '_'.join(metadata_file) + '.czi'
                    metadata_path = os.path.join(os.path.split(root)[0], metadata_file)
                    if self.current_rics_map is None:
                        raise ValueError("No RICS map available for fitting")

                    # Extract fitting parameters
                    if metadata_path:
                        import inspect_metadata as im  # ensure this import is available
                    
                        if not os.path.isfile(metadata_path):
                            self.log_message(f"Metadata file {metadata_path} not found. Using given parameters.")
                            pixel_size_um = float(self.fit_pixel_size.get()) * 1e-3
                            pixel_time_s = float(self.fit_pixel_dwell.get()) * 1e-6
                            line_time_s = float(self.fit_line_time.get()) * 1e-3
                        else:
                            channel_to_use = int(self.channel_to_use.get())
                            with pyczi.open_czi(metadata_path) as czidoc:
                                Pixel_size_nm, Pixel_dwell_time_us, line_time_ms = im.get_metadata(czidoc, channel_to_use)
                            self.log_message(f"Extracted metadata from {metadata_path}: px size={Pixel_size_nm}nm, dwell={Pixel_dwell_time_us}us, line time={line_time_ms}ms")
                            pixel_size_um = float(Pixel_size_nm) * 1e-3
                            pixel_time_s = float(Pixel_dwell_time_us) * 1e-6
                            line_time_s = float(line_time_ms) * 1e-3
                    else:
                        pixel_size_um = float(self.fit_pixel_size.get()) * 1e-3
                        pixel_time_s = float(self.fit_pixel_dwell.get()) * 1e-6
                        line_time_s = float(self.fit_line_time.get()) * 1e-3

                    
                    psf_size_xy_um = float(self.fit_psf_xy.get())
                    psf_aspect_ratio = float(self.fit_psf_aspect.get())
                    diffusion_model = self.diffusion_model.get()

                    self.log_message(f"Fitting parameters:")
                    self.log_message(f"  Pixel size: {pixel_size_um*1000:.1f} nm")
                    self.log_message(f"  Pixel dwell time: {pixel_time_s*1e6:.1f} μs")
                    self.log_message(f"  Line time: {line_time_s*1e3:.1f} ms")
                    self.log_message(f"  PSF size XY: {psf_size_xy_um:.3f} μm")
                    self.log_message(f"  Model: {diffusion_model}")

                    # Crop RICS map according to settings
                    rics_map = self.current_rics_map
                    crop_fast = float(self.fit_crop_fast.get())
                    crop_slow = float(self.fit_crop_slow.get())

                    floor_fast_ax = int(np.floor(rics_map.shape[1] * (1 - crop_fast) * 0.5))
                    ceil_fast_ax = int(np.floor(rics_map.shape[1] * 0.5 * (1 + crop_fast)))
                    floor_slow_ax = int(np.floor(rics_map.shape[0] * (1 - crop_slow) * 0.5))
                    ceil_slow_ax = int(np.floor(rics_map.shape[0] * 0.5 * (1 + crop_slow)))

                    rics_map = rics_map[floor_slow_ax:ceil_slow_ax, floor_fast_ax:ceil_fast_ax]

                    # Zero center pixel
                    center_y = rics_map.shape[0] // 2
                    center_x = rics_map.shape[1] // 2
                    rics_map[center_y, center_x] = 0.0

                    self.log_message(f"Cropped RICS map shape: {rics_map.shape}")

                    # Create RICS fitter using your module
                    fitter = rics_fit.RICS_fit(
                        RICS_map=rics_map,
                        pixel_size_um=pixel_size_um,
                        pixel_time_s=pixel_time_s,
                        line_time_s=line_time_s,
                        psf_size_xy_um=psf_size_xy_um,
                        psf_aspect_ratio=psf_aspect_ratio
                    )

                    # Run fitting based on model choice
                    if diffusion_model == "2Ddiff":
                        fit_params, model, residual = fitter.run_2Ddiff_fit()
                        diffusion_coeff = fit_params['diff_coeff'].value
                        amplitude = fit_params['amplitude'].value
                        offset = fit_params['offset'].value
                        N = float(0.5 / amplitude)
                        self.log_message("2D Diffusion fitting results:")
                        self.log_message(f"  Diffusion coefficient: {diffusion_coeff:.3f} μm²/s")
                        self.log_message(f"  Amplitude: {amplitude:.6f}")
                        self.log_message(f"  Offset: {offset:.6f}")
                        fit_results = []
                        fit_results.append({'filepath': file,
                                'Particle Number': N,
                                'Diffusion Coefficient': diffusion_coeff,
                                'residual': np.mean(residual)
                                })
                        results_df = pd.DataFrame(fit_results)
                        output_csv = self.saving_path.get()
                        results_df.to_csv(output_csv, index=False, mode = 'a')
                        

                    else:  # 3Ddiff
                        fit_params, model, residual = fitter.run_3Ddiff_fit()
                        diffusion_coeff = fit_params['diff_coeff'].value
                        amplitude = fit_params['amplitude'].value
                        offset = fit_params['offset'].value
                        N = float(0.35 / amplitude)

                        self.log_message("3D Diffusion fitting results:")
                        self.log_message(f"  Diffusion coefficient: {diffusion_coeff:.3f} μm²/s")
                        self.log_message(f"  Amplitude: {amplitude:.6f}")
                        self.log_message(f"  Offset: {offset:.6f}")
                        fit_results = []
                        fit_results.append({'filepath': file,
                                'Particle Number': N,
                                'Diffusion Coefficient': diffusion_coeff,
                                'residual': np.mean(residual)
                                })
                        results_df = pd.DataFrame(fit_results)
                        output_csv = self.saving_path.get()
                        results_df.to_csv(output_csv, index=False, mode = 'a')

                    self.fit_results = {
                        'rics_map': rics_map,
                        'model': model,
                        'residual': residual,
                        'diffusion_coeff': diffusion_coeff,
                        'amplitude': amplitude,
                        'offset': offset,
                        'fit_params': fit_params,
                        'fitter': fitter,
                        'model_type': diffusion_model
                    }
                    self.current_rics_map = None # reset the RICS map
                    # Update display
                    self.root.after(0, self.update_fitting_display)
                    if hasattr(self, 'fit_1d_var') and self.fit_1d_var.get():
                        # Perform 1D fitting when checked
                        self.run_1d_fitting()
                    else:
                        pass
                except Exception as e:
                    self.log_message(f"Fitting error: {str(e)}")
                    import traceback
                    self.log_message(f"Traceback: {traceback.format_exc()}")
                finally:
                    self.root.after(0, lambda: self.status_var.set("Ready"))
        elif self.rics_file.get():
            file = self.rics_file.get()
            try:
                    # Load RICS map if needed
                    
                    import os
                    self.current_rics_map = tifffile.imread(file)
                    self.log_message(f"Loaded RICS map for fitting: {file}")
                    root, ext = os.path.splitext(file)
                    metadata_file = os.path.split(root)[1].split('_')
                    metadata_file = metadata_file[:-1]
                    metadata_file = '_'.join(metadata_file) + '.czi'
                    metadata_path = os.path.join(os.path.split(root)[0], metadata_file)
                    if self.current_rics_map is None:
                        raise ValueError("No RICS map available for fitting")

                    # Extract fitting parameters
                    if metadata_path:
                        import inspect_metadata as im  # ensure this import is available
                    
                        if not os.path.isfile(metadata_path):
                            self.log_message(f"Metadata file {metadata_path} not found. Using given parameters.")
                            pixel_size_um = float(self.fit_pixel_size.get()) * 1e-3
                            pixel_time_s = float(self.fit_pixel_dwell.get()) * 1e-6
                            line_time_s = float(self.fit_line_time.get()) * 1e-3
                        else:
                            channel_to_use = int(self.channel_to_use.get())
                            with pyczi.open_czi(metadata_path) as czidoc:
                                Pixel_size_nm, Pixel_dwell_time_us, line_time_ms = im.get_metadata(czidoc, channel_to_use)
                            self.log_message(f"Extracted metadata from {metadata_path}: px size={Pixel_size_nm}nm, dwell={Pixel_dwell_time_us}us, line time={line_time_ms}ms")
                            pixel_size_um = float(Pixel_size_nm) * 1e-3
                            pixel_time_s = float(Pixel_dwell_time_us) * 1e-6
                            line_time_s = float(line_time_ms) * 1e-3
                    else:
                        pixel_size_um = float(self.fit_pixel_size.get()) * 1e-3
                        pixel_time_s = float(self.fit_pixel_dwell.get()) * 1e-6
                        line_time_s = float(self.fit_line_time.get()) * 1e-3

                    
                    psf_size_xy_um = float(self.fit_psf_xy.get())
                    psf_aspect_ratio = float(self.fit_psf_aspect.get())
                    diffusion_model = self.diffusion_model.get()

                    self.log_message(f"Fitting parameters:")
                    self.log_message(f"  Pixel size: {pixel_size_um*1000:.1f} nm")
                    self.log_message(f"  Pixel dwell time: {pixel_time_s*1e6:.1f} μs")
                    self.log_message(f"  Line time: {line_time_s*1e3:.1f} ms")
                    self.log_message(f"  PSF size XY: {psf_size_xy_um:.3f} μm")
                    self.log_message(f"  Model: {diffusion_model}")

                    # Crop RICS map according to settings
                    rics_map = self.current_rics_map
                    crop_fast = float(self.fit_crop_fast.get())
                    crop_slow = float(self.fit_crop_slow.get())

                    floor_fast_ax = int(np.floor(rics_map.shape[1] * (1 - crop_fast) * 0.5))
                    ceil_fast_ax = int(np.floor(rics_map.shape[1] * 0.5 * (1 + crop_fast)))
                    floor_slow_ax = int(np.floor(rics_map.shape[0] * (1 - crop_slow) * 0.5))
                    ceil_slow_ax = int(np.floor(rics_map.shape[0] * 0.5 * (1 + crop_slow)))

                    rics_map = rics_map[floor_slow_ax:ceil_slow_ax, floor_fast_ax:ceil_fast_ax]

                    # Zero center pixel
                    center_y = rics_map.shape[0] // 2
                    center_x = rics_map.shape[1] // 2
                    rics_map[center_y, center_x] = 0.0

                    self.log_message(f"Cropped RICS map shape: {rics_map.shape}")

                    # Create RICS fitter using your module
                    fitter = rics_fit.RICS_fit(
                        RICS_map=rics_map,
                        pixel_size_um=pixel_size_um,
                        pixel_time_s=pixel_time_s,
                        line_time_s=line_time_s,
                        psf_size_xy_um=psf_size_xy_um,
                        psf_aspect_ratio=psf_aspect_ratio
                    )

                    # Run fitting based on model choice
                    if diffusion_model == "2Ddiff":
                        fit_params, model, residual = fitter.run_2Ddiff_fit()
                        diffusion_coeff = fit_params['diff_coeff'].value
                        amplitude = fit_params['amplitude'].value
                        offset = fit_params['offset'].value
                        N = float(0.5 / amplitude)
                        self.log_message("2D Diffusion fitting results:")
                        self.log_message(f"  Diffusion coefficient: {diffusion_coeff:.3f} μm²/s")
                        self.log_message(f"  Amplitude: {amplitude:.6f}")
                        self.log_message(f"  Offset: {offset:.6f}")
                        fit_results = []
                        fit_results.append({'filepath': file,
                                'Particle Number': N,
                                'Diffusion Coefficient': diffusion_coeff,
                                'residual': np.mean(residual)
                                })
                        results_df = pd.DataFrame(fit_results)
                        output_csv = self.saving_path.get()
                        results_df.to_csv(output_csv, index=False, mode = 'a')
                        

                    else:  # 3Ddiff
                        fit_params, model, residual = fitter.run_3Ddiff_fit()
                        diffusion_coeff = fit_params['diff_coeff'].value
                        amplitude = fit_params['amplitude'].value
                        offset = fit_params['offset'].value
                        N = float(0.35 / amplitude)

                        self.log_message("3D Diffusion fitting results:")
                        self.log_message(f"  Diffusion coefficient: {diffusion_coeff:.3f} μm²/s")
                        self.log_message(f"  Amplitude: {amplitude:.6f}")
                        self.log_message(f"  Offset: {offset:.6f}")
                        fit_results = []
                        fit_results.append({'filepath': file,
                                'Particle Number': N,
                                'Diffusion Coefficient': diffusion_coeff,
                                'residual': np.mean(residual)
                                })
                        results_df = pd.DataFrame(fit_results)
                        output_csv = self.saving_path.get()
                        results_df.to_csv(output_csv, index=False, mode = 'a')

                    self.fit_results = {
                        'rics_map': rics_map,
                        'model': model,
                        'residual': residual,
                        'diffusion_coeff': diffusion_coeff,
                        'amplitude': amplitude,
                        'offset': offset,
                        'fit_params': fit_params,
                        'fitter': fitter,
                        'model_type': diffusion_model
                    }
                    self.current_rics_map = None # reset the RICS map
                    # Update display
                    self.root.after(0, self.update_fitting_display)
                    if hasattr(self, 'fit_1d_var') and self.fit_1d_var.get():
                        # Perform 1D fitting when checked
                        self.run_1d_fitting()
                    else:
                        pass
            except Exception as e:
                self.log_message(f"Fitting error: {str(e)}")
                import traceback
                self.log_message(f"Traceback: {traceback.format_exc()}")
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))

    def run_1d_fitting(self):
        """Run 1D fitting on fast axis using your module"""
        if not rics_fit:
            messagebox.showerror("Error", "Fitting module not loaded!")
            return

        self.log_message("Running 1D fast axis fitting...")
        self.status_var.set("Running 1D fitting...")

        fit_thread = Thread(target=self._run_1d_fitting_thread)
        fit_thread.daemon = True
        fit_thread.start()

    def _run_1d_fitting_thread(self):
        """Thread for 1D fitting"""
        try:
            file = self.current_file

            fitter = self.fit_results['fitter']

            # Run 1D fit
            fit_params_1D, model_1D, residual_1D = fitter.fast_axis_diff_fit()

            diffusion_coeff_1D = fit_params_1D['diff_coeff'].value
            amplitude_1D = fit_params_1D['amplitude'].value
            offset_1D = fit_params_1D['offset'].value

            self.log_message("1D Fast axis fitting results:")
            self.log_message(f"  Diffusion coefficient (1D): {diffusion_coeff_1D:.3f} μm²/s")
            self.log_message(f"  Amplitude (1D): {amplitude_1D:.6f}")
            self.log_message(f"  Offset (1D): {offset_1D:.6f}")

            # Store 1D results
            self.fit_results['fit_params_1D'] = fit_params_1D
            self.fit_results['model_1D'] = model_1D
            self.fit_results['residual_1D'] = residual_1D
            self.fit_results['diffusion_coeff_1D'] = diffusion_coeff_1D
            fit_results = []
            fit_results.append({'filepath': file,
                    'Diffusion Coefficient': diffusion_coeff_1D,
                    'residual': np.mean(residual_1D)
                    })
            results_df = pd.DataFrame(fit_results)
            output_csv = self.saving_path.get()
            root, ext = os.path.splitext(output_csv)
            root = root + '_1D'
            output_csv = root + ext
            results_df.to_csv(output_csv, index=False, mode = 'a')
            # Update display
            self.root.after(0, self.update_fitting_display)

        except Exception as e:
            self.log_message(f"1D Fitting error: {str(e)}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))

    def run_diffusion_map(self):
        if not rics_fit:
            messagebox.showerror("Error", "Fitting module not loaded!")
            return

        if not self.input_file_diff_map.get():
            messagebox.showwarning("Warning", "Please select an input file first")
            return

        self.log_message("Starting Diff Map export...")
        self.status_var.set("Exporting Diff Map...")
        self.status_bar.update_idletasks()  # Force redraw

        

        # Create a thread for RICS export
        export_thread = Thread(target=self._run_diffusion_map_thread)
        export_thread.daemon = True
        export_thread.start()


        

    def _run_diffusion_map_thread(self):
        
        try:

            self.log_message("Generating diffusion map from current intensity stack...")
            # Extract fitting parameters
            input_file_diff_map = self.input_file_diff_map.get()
            channel_to_use = int(self.channel_to_use_diff_map.get())
            import inspect_metadata as im  # ensure this import is available
            with pyczi.open_czi(input_file_diff_map) as czidoc:
                        Pixel_size_nm, Pixel_dwell_time_us, line_time_ms = im.get_metadata(czidoc, channel_to_use)
            self.log_message(f"Extracted metadata from {input_file_diff_map}: px size={Pixel_size_nm}nm, dwell={Pixel_dwell_time_us}us, line time={line_time_ms}ms")
            pixel_size_um = Pixel_size_nm * 1e-3
            pixel_time_s = Pixel_dwell_time_us * 1e-6
            line_time_s = line_time_ms * 1e-3
            self.fit_pixel_size = Pixel_size_nm
            self.fit_pixel_dwell = Pixel_dwell_time_us
            self.fit_line_time = line_time_ms
            psf_size_xy_um = float(self.fit_psf_xy.get())
            psf_aspect_ratio = float(self.fit_psf_aspect.get())
            diffusion_model = self.diffusion_model.get()
            all_frames = []
            input_file_diff_map = self.input_file_diff_map.get()
            with pyczi.open_czi(input_file_diff_map) as czidoc:
                total_bounding_box = czidoc.total_bounding_box
                n_frames = total_bounding_box['T'][1]
            for i_frame in range(n_frames):
                frame_data = export_rics.read_frame(self.input_file_diff_map.get(), i_frame, 0)
                all_frames.append(frame_data)
            stack = np.stack(all_frames, axis = 0)
            window_size_diff_map = int(self.window_size_diff_map.get())
            offset_diff_map = int(self.offset_diff_map.get())

            Dmap, Nmap, Bmap, fast_axis_data_list, modelmap_fast_axis_list = self.compute_local_diffusion_map(
                stack, pixel_size_um, pixel_time_s, line_time_s,
                psf_size_xy_um, psf_aspect_ratio, window_size=window_size_diff_map, offset=offset_diff_map, model=diffusion_model, min_valid_pixels = 0.5, input_file_diff_map = input_file_diff_map
            )
            self.diffusion_map = Dmap
            self.B_map = Bmap
            self.N_map = Nmap
            self.RICS_map_fast_axis_list = fast_axis_data_list
            self.modelmap_fast_axis_list = modelmap_fast_axis_list
            self.root.after(0, self.update_fitting_display)
            diff_map_output = os.path.splitext(input_file_diff_map)[0] + '_diff_map.tif'
            tifffile.imwrite(diff_map_output, Dmap, photometric='minisblack')

            self.log_message(f"Diffusion map saved to: {diff_map_output}")
            
        except Exception as e:
            self.log_message(f"Diffusion map error: {str(e)}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))

    



    def compute_local_diffusion_map(self, stack, pixelsize_um, pixeltime_s, linetime_s,
                                psf_xy_um=0.2, psf_aspect=5.0,
                                window_size=32, offset=16,
                                model='2Ddiff', min_valid_pixels=0.5, input_file_diff_map = None):
        """
        Grid-based diffusion map from image stack, reproducing PAM grid fitting.
        Each block of the image stack is fitted to return spatial maps of D, amplitude, and brightness.
        """

        def poll_queue():
            progress = 0
            while progress<=100:
                try:
                    progress = self.progress_queue.get(block = False)
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    self.root.after(0, lambda: self.progress_bar.update_idletasks())
                    if progress == 100:
                        self.root.after(0, lambda: self.progress_bar.pack_forget())
                        self.root.after(0, lambda: self.progress_bar.update_idletasks())
                        break
                except:
                    pass

        threading.Thread(target=poll_queue, daemon=True).start()

        h, w = stack.shape[-2:] # get the height and width of the image 
        nx = (w - window_size) // offset + 1 # number of windows in x 
        ny = (h - window_size) // offset + 1 # number of windows in y
        Dmap = np.full((h, w), np.nan) # create NaN images with the same size
        Nmap = np.full((h, w), np.nan)
        Bmap = np.full((h, w), np.nan) 
        block_args = []
        for iy in range(ny):
            for ix in range(nx):
                y0 = iy * offset
                x0 = ix * offset
                block = stack[:, y0:y0+window_size, x0:x0+window_size]
                block_args.append((
                    block.copy(), y0, x0,
                    pixelsize_um, pixeltime_s, linetime_s,
                    psf_xy_um, psf_aspect, model, input_file_diff_map
                ))
        total = len(block_args)
        # Start polling the progress queue in a separate thread to keep GUI responsive
        self.root.after(0, lambda: self.progress_bar.pack_forget())
        self.status_bar.update_idletasks()
        self.progress_queue = multiprocessing.Queue()
        pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
        results = []
        i = 0
        for result in tqdm(pool.imap(process_block_diff_map, block_args), total=total):
            results.append(result)
            i+=1
            progress = (i/total)*100
            self.progress_queue.put(progress)

        
        # Initialize output maps
        Dmap = np.full((h, w), np.nan)
        Nmap = np.full((h, w), np.nan)
        Bmap = np.full((h, w), np.nan)
        RICS_map_fast_axis_list = []
        modelmap_fast_axis_list = []
        # Fill maps from results
        for y0, x0, D, amp, brightness, RICS_map_fast_axis, modelmap_fast_axis in results:

            Dmap[y0:y0+window_size, x0:x0+window_size] = D
            Nmap[y0:y0+window_size, x0:x0+window_size] = amp
            Bmap[y0:y0+window_size, x0:x0+window_size] = brightness
            RICS_map_fast_axis_list.append(RICS_map_fast_axis)
            modelmap_fast_axis_list.append(modelmap_fast_axis)
        
        Dmap = scipy.ndimage.median_filter(Dmap, size=3)
        Nmap = scipy.ndimage.median_filter(Nmap, size=3)
        Bmap = scipy.ndimage.median_filter(Bmap, size=3)

        

        

        return Dmap, Nmap, Bmap, RICS_map_fast_axis_list, modelmap_fast_axis_list

    def update_fitting_display(self):
        """Update the fitting results display using your plotting functions"""
        if self.fit_results is not None:
            self.fit_fig.clear()


            # Use your existing plotting function
            try:
                # This will create a separate figure - we'll recreate for our canvas
                rics_fit.plot_fitting_workflow(
                    self.fit_results['rics_map'],
                    self.fit_results['model'], 
                    self.fit_results['residual'],
                    "gui_display"
                )
            except:
                pass

            # Create our own display matching your layout
            if 'model_1D' in self.fit_results:
                # If 1D fit is available, show both 2D/3D and 1D results
                gs = gridspec.GridSpec(3, 4, figure=self.fit_fig, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1])
            else:
                # Only 2D/3D results
                gs = gridspec.GridSpec(2, 4, figure=self.fit_fig, width_ratios=[1, 1, 1, 1])

            # Create coordinate arrays for 3D plots
            X = np.arange(self.fit_results['rics_map'].shape[1])
            Y = np.arange(self.fit_results['rics_map'].shape[0])
            X, Y = np.meshgrid(X, Y)

            # Original RICS map
            ax1 = self.fit_fig.add_subplot(gs[0, 0], projection='3d')
            ax1.plot_surface(X, Y, self.fit_results['rics_map'], cmap='jet', alpha=0.8)
            ax1.set_title('RICS Data')
            ax1.view_init(elev=20, azim=90)

            # Model
            ax2 = self.fit_fig.add_subplot(gs[0, 1], projection='3d')
            ax2.plot_surface(X, Y, self.fit_results['model'], cmap='jet', alpha=0.8)
            ax2.set_title('RICS Fit')
            ax2.view_init(elev=20, azim=90)

            # Residual
            ax3 = self.fit_fig.add_subplot(gs[0, 2], projection='3d')
            ax3.plot_surface(X, Y, self.fit_results['residual'], cmap='jet', alpha=0.8)
            ax3.set_title('Residuals')
            ax3.view_init(elev=20, azim=90)

            # 1D cross-section comparison
            ax4 = self.fit_fig.add_subplot(gs[1, :])
            center = self.fit_results['rics_map'].shape[0] // 2
            x_axis = np.arange(self.fit_results['rics_map'].shape[1]) - self.fit_results['rics_map'].shape[1]//2

            ax4.plot(x_axis, self.fit_results['rics_map'][center, :], 'ko-', 
                    label='Data (Fast axis)', markersize=4, linewidth=1)
            ax4.plot(x_axis, self.fit_results['model'][center, :], 'r-', 
                    label=f'{self.fit_results["model_type"]} Fit (D={self.fit_results["diffusion_coeff"]:.3f} μm²/s)', 
                    linewidth=2)

            # Add 1D fit if available
            if 'model_1D' in self.fit_results:
                ax4.plot(x_axis, self.fit_results['model_1D'], 'g--', 
                        label=f'1D Fit (D={self.fit_results["diffusion_coeff_1D"]:.3f} μm²/s)', 
                        linewidth=2)

            ax4.set_xlabel('Pixel lag')
            ax4.set_ylabel('Correlation')
            ax4.set_title('1D Cross-section Fits')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # If 1D residuals available, show them
            if 'residual_1D' in self.fit_results:
                ax5 = self.fit_fig.add_subplot(gs[2, :])
                ax5.plot(x_axis, self.fit_results['residual_1D'], 'g.-', alpha=0.7, label='1D Residuals')
                ax5.axhline(0, color='k', linestyle='--', alpha=0.5)
                ax5.set_xlabel('Pixel lag')
                ax5.set_ylabel('Residuals')
                ax5.set_title('1D Fit Residuals')
                ax5.legend()
                ax5.grid(True, alpha=0.3)

            if self.diffusion_map is not None:
                median_val = np.nanmedian(self.diffusion_map) # Calculate median of the diffusion map (excluding NaNs if present)
                # Define the range
                vmin = 0.5 * median_val
                vmax = 2.0 * median_val
                ax6 = self.fit_fig.add_subplot(gs[0, 3])
                im6 = ax6.imshow(self.diffusion_map, cmap='jet', vmin = vmin, vmax = vmax)
                ax6.set_title("Diffusion Map")
                ax6.axis('off')
                self.fit_fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

            self.fit_fig.tight_layout()
            self.fit_canvas.draw()
        elif self.diffusion_map is not None and self.B_map is not None and self.N_map is not None:
            self.fit_fig.clear()
            gs = gridspec.GridSpec(2, 3, figure=self.fit_fig, width_ratios = [1,1,1])

            median_val = np.nanmedian(self.diffusion_map) # Calculate median of the diffusion map (excluding NaNs if present)
            # Define the range
            vmin = 0.5 * median_val
            vmax = 2.0 * median_val
            ax6 = self.fit_fig.add_subplot(gs[0, 0])
            im6 = ax6.imshow(self.diffusion_map, cmap='jet', vmin = vmin, vmax = vmax)
            ax6.set_title("Diffusion Map")
            ax6.axis('off')
            self.fit_fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

            median_val = np.nanmedian(self.B_map) # Calculate median of the diffusion map (excluding NaNs if present)
            # Define the range
            vmin = 0.5 * median_val
            vmax = 2.0 * median_val
            ax7 = self.fit_fig.add_subplot(gs[0, 1])
            im7 = ax7.imshow(self.B_map, cmap='jet', vmin = vmin, vmax = vmax)
            ax7.set_title("Brightness Map")
            ax7.axis('off')
            self.fit_fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

            median_val = np.nanmedian(self.N_map) # Calculate median of the diffusion map (excluding NaNs if present)
            # Define the range
            vmin = 0.5 * median_val
            vmax = 2.0 * median_val
            ax8 = self.fit_fig.add_subplot(gs[0, 2])
            im8 = ax8.imshow(self.N_map, cmap='jet', vmin = vmin, vmax = vmax)
            ax8.set_title("Number Map")
            ax8.axis('off')
            self.fit_fig.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

            ax9 = self.fit_fig.add_subplot(gs[1, 1])
            ax10 = self.fit_fig.add_subplot(gs[1, 2])

            for i in range(len(self.RICS_map_fast_axis_list)):
                fast_axis_data = self.RICS_map_fast_axis_list[i]
                modelmap_fast_axis = self.modelmap_fast_axis_list[i]
                n_points = len(fast_axis_data)
                center = n_points // 2
                x_lag = (np.arange(n_points) - center) * self.fit_pixel_size * 1e-3
                
                # Plot all datasets on the same set of axes
                ax9.plot(x_lag, fast_axis_data, 'o-')
                ax10.plot(x_lag, modelmap_fast_axis, '--')

            
            ax9.set_ylabel('Autocorrelation G(Δx)')
            ax9.set_title('RICS 1D Autocorrelation (Fast Axis)')
            # ax9.legend()
            ax9.grid(True)

            ax10.set_ylabel('Autocorrelation G(Δx)')
            ax10.set_title('RICS Fit (Fast Axis)')
            # ax10.legend()
            ax10.grid(True)


            self.fit_fig.tight_layout()
            self.fit_canvas.draw()

    

    def save_results(self):
        """Save analysis results"""
        filename = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.results_text.get(1.0, tk.END))
            self.log_message(f"Results saved to {filename}")

    def save_session(self):
        """Save current session parameters"""
        filename = filedialog.asksaveasfilename(
            title="Save session",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                session_data = {
                    'simulation_params': {
                        'img_width': self.img_width.get(),
                        'img_height': self.img_height.get(),
                        'n_frames': self.n_frames.get(),
                        'pixel_dwell': self.pixel_dwell.get(),
                        'pixel_size': self.pixel_size.get(),
                        'brightness': self.brightness.get(),
                        'n_particles': self.n_particles.get(),
                        'diff_x': self.diff_x.get(),
                        'diff_y': self.diff_y.get(),
                        'rotation': self.rotation.get(),
                        'background': self.background.get(),
                        'psf_sigma': self.psf_sigma.get(),
                        'sim_type': self.sim_type.get(),
                        'output_path': self.output_path.get()
                    },
                    'export_params': {
                        'input_file': self.input_file.get(),
                        'channel': self.channel.get(),
                        'crop_factor': self.crop_factor.get(),
                        'window_size': self.window_size.get(),
                        'correct_drift': self.correct_drift.get()
                    },
                    'fitting_params': {
                        'rics_file': self.rics_file.get(),
                        'fit_pixel_size': self.fit_pixel_size.get(),
                        'fit_pixel_dwell': self.fit_pixel_dwell.get(),
                        'fit_line_time': self.fit_line_time.get(),
                        'fit_psf_xy': self.fit_psf_xy.get(),
                        'fit_psf_aspect': self.fit_psf_aspect.get(),
                        'fit_crop_fast': self.fit_crop_fast.get(),
                        'fit_crop_slow': self.fit_crop_slow.get(),
                        'diffusion_model': self.diffusion_model.get()
                    }
                }

                with open(filename, 'w') as f:
                    json.dump(session_data, f, indent=2)

                self.log_message(f"Session saved to {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Could not save session: {str(e)}")

    def load_session(self):
        """Load session parameters"""
        filename = filedialog.askopenfilename(
            title="Load session",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    session_data = json.load(f)

                # Load simulation parameters
                if 'simulation_params' in session_data:
                    sim_params = session_data['simulation_params']
                    self.img_width.set(sim_params.get('img_width', '256'))
                    self.img_height.set(sim_params.get('img_height', '256'))
                    self.n_frames.set(sim_params.get('n_frames', '100'))
                    self.pixel_dwell.set(sim_params.get('pixel_dwell', '50'))
                    self.pixel_size.set(sim_params.get('pixel_size', '20'))
                    self.brightness.set(sim_params.get('brightness', '2000'))
                    self.n_particles.set(sim_params.get('n_particles', '250'))
                    self.diff_x.set(sim_params.get('diff_x', '100'))
                    self.diff_y.set(sim_params.get('diff_y', '10'))
                    self.rotation.set(sim_params.get('rotation', '90'))
                    self.background.set(sim_params.get('background', '0'))
                    self.psf_sigma.set(sim_params.get('psf_sigma', '5'))
                    self.sim_type.set(sim_params.get('sim_type', 'anisotropic_rotated'))
                    self.output_path.set(sim_params.get('output_path', './simulation_output.tif'))

                # Load export parameters
                if 'export_params' in session_data:
                    export_params = session_data['export_params']
                    self.input_file.set(export_params.get('input_file', ''))
                    self.channel.set(export_params.get('channel', '0'))
                    self.crop_factor.set(export_params.get('crop_factor', '0.5'))
                    self.window_size.set(export_params.get('window_size', '3'))
                    self.correct_drift.set(export_params.get('correct_drift', False))

                # Load fitting parameters
                if 'fitting_params' in session_data:
                    fit_params = session_data['fitting_params']
                    self.rics_file.set(fit_params.get('rics_file', ''))
                    self.fit_pixel_size.set(fit_params.get('fit_pixel_size', '20'))
                    self.fit_pixel_dwell.set(fit_params.get('fit_pixel_dwell', '50'))
                    self.fit_line_time.set(fit_params.get('fit_line_time', '12.8'))
                    self.fit_psf_xy.set(fit_params.get('fit_psf_xy', '0.2'))
                    self.fit_psf_aspect.set(fit_params.get('fit_psf_aspect', '4.985423166'))
                    self.fit_crop_fast.set(fit_params.get('fit_crop_fast', '0.5'))
                    self.fit_crop_slow.set(fit_params.get('fit_crop_slow', '0.5'))
                    self.diffusion_model.set(fit_params.get('diffusion_model', '2Ddiff'))

                self.log_message(f"Session loaded from {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Could not load session: {str(e)}")

    def export_plots(self):
        """Export all plots"""
        directory = filedialog.askdirectory(title="Select directory for plot export")
        if directory:
            try:
                plots_saved = 0
                if hasattr(self, 'sim_fig') and self.simulated_stack is not None:
                    self.sim_fig.savefig(os.path.join(directory, 'simulation_results.png'), 
                                        dpi=300, bbox_inches='tight', facecolor='white')
                    plots_saved += 1

                if hasattr(self, 'rics_fig') and self.current_rics_map is not None:
                    self.rics_fig.savefig(os.path.join(directory, 'rics_analysis.png'), 
                                         dpi=300, bbox_inches='tight', facecolor='white')
                    plots_saved += 1

                if hasattr(self, 'fit_fig') and self.fit_results is not None:
                    self.fit_fig.savefig(os.path.join(directory, 'fitting_results.png'), 
                                        dpi=300, bbox_inches='tight', facecolor='white')
                    plots_saved += 1

                self.log_message(f"Exported {plots_saved} plots to {directory}")

            except Exception as e:
                messagebox.showerror("Error", f"Could not export plots: {str(e)}")


def process_block_diff_map(args):
    """
    Worker function: processes one block and returns (y0, x0, D, amp)
    """
    block, y0, x0, pixelsize_um, pixeltime_s, linetime_s, psf_xy_um, psf_aspect, model, input_file_diff_map = args   
    if np.count_nonzero(~np.isnan(block)) < 0.5 * block.size:
        print("Some blocks were found invalid")
        return (y0, x0, np.nan, np.nan, np.nan)

    
    try:
        # import rics_fit
        RICS_map, sd_map, stack, corrected_stack = export_rics.process_all_frames_tiff(block, block.shape[0], 0, window_size = 3)
        center_slow_ax = RICS_map.shape[0] // 2
        center_fast_ax = RICS_map.shape[1] // 2

        RICS_map[center_slow_ax, center_fast_ax] = 0
        RICS_map_fast_axis = RICS_map[center_slow_ax,:]

        
        fitter = rics_fit.RICS_fit(RICS_map, pixelsize_um, pixeltime_s, linetime_s, psf_xy_um, psf_aspect)
        if model == '3Ddiff':
            params, modelmap, res = fitter.run_3Ddiff_fit()
        else:
            params, modelmap, res = fitter.run_2Ddiff_fit()
        D = params['diff_coeff'].value
        amp = params['amplitude'].value

        modelmap_fast_axis = modelmap[center_slow_ax,:]
        # D = 0
        # amp = 0
        
    except Exception as e:
        import traceback
        print("Exception in process_block:", e)
        print(traceback.format_exc())
        D, amp = np.nan, np.nan
    brightness = np.std(block)
    return (y0, x0, D, amp, brightness, RICS_map_fast_axis, modelmap_fast_axis)

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
def on_closing():
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
        root.quit()
    


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    multiprocessing.set_start_method('spawn', force = True)

    root = tk.Tk()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = ModularRICSGUI(root)
    root.mainloop()

    


