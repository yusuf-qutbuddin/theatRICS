
#!/usr/bin/env python3
"""
Created on Friday 3rd October 2025

@author: yusufqq

Modular RICS Analysis GUI Application
Imports existing RICS modules and provides a unified interface
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sv_ttk
import numpy as np
from path import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import traceback
import matplotlib
matplotlib.use('agg')
from pylibCZIrw import czi as pyczi
import multiprocessing
import queue
import tifffile
import scipy.ndimage
import json
import pandas as pd
from theatrics.workers.sfcs_worker import sfcs_process_main_curvefit
from theatrics.workers.export_worker import export_rics_process_main
from theatrics.workers.fit_worker import fit_rics_process_main
from theatrics.workers.sim_worker import simulate_rics_process_main
from theatrics.workers.diffmap_worker import diffusion_map_process_main
from theatrics.workers.fcsfit_worker import fcsfit_process_main
from theatrics.workers.frap_worker import frap_process_main
from theatrics.workers.vesicle_worker import vesicle_process_main


from theatrics.vesicle import detection as vesicle_detection
from theatrics.fcsfit import calculations as calculate
from theatrics.utils.file_utils import get_files_from_folder
from theatrics.utils.mp_utils import clamp_workers
from theatrics.frap import analysis as frap_analysis




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
        if not all([sfcs_process_main_curvefit, export_rics_process_main, fit_rics_process_main, simulate_rics_process_main, diffusion_map_process_main, get_files_from_folder, clamp_workers]):
            self.show_module_error()
            return

        # Create main interface
        self.setup_gui()

    def show_module_error(self):
        """Show error if modules couldn't be loaded"""
        error_frame = ttk.Frame(self.root)
        error_frame.pack(fill='both', expand=True, padx=20, pady=20)

        error_label = ttk.Label(error_frame, 
                               text="Error: Could not load modules!\n\n"
                                    "Please ensure the following files are in the same directory:\n"
                                    "• simRICS.py\n"
                                    "• export_rics.py\n"  
                                    "• rics_fit.py\n\n"
                                    "• SFCS_module.py\n\n"
                                    
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
        self.create_SFCS_tab()
        self.create_fcs_fit_tab()
        self.create_frap_tab()
        self.create_vesicle_tab()
        self.create_results_tab()


        self.status_var = tk.StringVar()
        self.status_var.set("Ready - All modules loaded successfully")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=1, column=0, sticky='w')
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, sticky="e", padx=10)
        self.button = ttk.Button(button_frame, text="Toggle Dark Mode", command=sv_ttk.toggle_theme)
        self.button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(button_frame, text="Cancel Running Task", command=self.cancel_current_task)
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        self.restart_button = ttk.Button(button_frame, text="Restart", command=self.restart_application)
        self.restart_button.pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, sticky='ew')
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
     
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
        self.export_rics_browse_entry = ttk.Entry(input_frame, textvariable=self.input_file, width=25)
        self.export_rics_browse_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.export_rics_browse_btn =ttk.Button(input_frame, text="Browse", command=self.browse_input_file)
        self.export_rics_browse_btn.pack(side=tk.RIGHT)
        self.register_busy_widget(self.export_rics_browse_entry)
        self.register_busy_widget(self.export_rics_browse_btn)
        row += 1
        ttk.Label(export_params, text="Input folder (batch analysis): ").grid(row=row, column=0, sticky='w', pady=2)
        batch_input_frame = ttk.Frame(export_params)
        batch_input_frame.grid(row=row, column=1, columnspan=2, pady=2, sticky='ew')
        self.batch_input_folder = tk.StringVar()
        self.export_rics_browse_batch_entry = ttk.Entry(batch_input_frame, textvariable=self.batch_input_folder, width=25)
        self.export_rics_browse_batch_entry.grid(row=row, column=1, pady=2)
        self.export_rics_browse_batch_btn = ttk.Button(batch_input_frame, text="Browse", command=self.browse_batch_input_folder)
        self.export_rics_browse_batch_btn.grid(row=row, column=2, pady=2)
        self.register_busy_widget(self.export_rics_browse_batch_btn)
        self.register_busy_widget(self.export_rics_browse_batch_entry)
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
        self.export_rics_btn = ttk.Button(export_button_frame, text="Export RICS", command=self.export_rics)
        self.export_rics_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(export_button_frame, text="Load RICS", command=self.load_rics).pack(side=tk.LEFT, padx=5)
        self.register_busy_widget(self.export_rics_btn)

        
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

    def create_SFCS_tab(self):
        """Create the SFCS tab using SFCS module"""
        SFCS_frame = ttk.Frame(self.notebook)
        self.notebook.add(SFCS_frame, text="SFCS")

        # Parameters frame
        SFCS_params = ttk.LabelFrame(SFCS_frame, text="SFCS Parameters", padding=10)
        SFCS_params.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Configure grid weights for proper expansion
        SFCS_params.grid_columnconfigure(1, weight=1)

        row = 0

        # Input file
        ttk.Label(SFCS_params, text="Input file:").grid(row=row, column=0, sticky='w', pady=2)
        input_frame = ttk.Frame(SFCS_params)  # Renamed to avoid name conflict
        input_frame.grid(row=row, column=1, columnspan=2, pady=2, sticky='ew')
        self.sfcs_input_file = tk.StringVar()
        self.sfcs_entry = ttk.Entry(input_frame, textvariable=self.sfcs_input_file, width=25)
        self.sfcs_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sfcs_browse_btn = ttk.Button(input_frame, text="Browse", command=self.browse_sfcs_input_file)
        self.sfcs_browse_btn.pack(side=tk.RIGHT)
        self.register_busy_widget(self.sfcs_entry)
        self.register_busy_widget(self.sfcs_browse_btn)

        row += 1
        ttk.Label(SFCS_params, text="Channel to use:").grid(row=row, column=0, sticky='w', pady=2)
        channel_frame = ttk.Frame(SFCS_params)  # Fixed: Use correct parent
        channel_frame.grid(row=row, column=1, columnspan=2, pady=2, sticky='ew')
        self.sfcs_channel = tk.StringVar(value="0")
        model_combo = ttk.Combobox(channel_frame, textvariable=self.sfcs_channel,  # Fixed parent
                                   values=["0", "1"], width=12)  # String values for Combobox
        model_combo.pack(side=tk.LEFT)  # Fixed: proper packing

        row += 1
        self.correct_bleach = tk.BooleanVar()
        ttk.Checkbutton(SFCS_params, text="Bleach Correction", variable=self.correct_bleach).grid(row=row, column=0, columnspan=2, sticky='w', pady=2)

        row += 1
        ttk.Label(SFCS_params, text="Number of cores").grid(row=row, column=0, sticky='w', pady=2)
        cpu_frame = ttk.Frame(SFCS_params)  # Fixed: Use correct parent
        cpu_frame.grid(row=row, column=1, columnspan=2, pady=2, sticky='ew')
        self.n_cpu = tk.StringVar(value="4")
        ttk.Entry(cpu_frame, textvariable=self.n_cpu, width=15).pack(side=tk.LEFT)  # Fixed parent

        # Buttons
        row += 1
        SFCS_button_frame = ttk.Frame(SFCS_params)
        SFCS_button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        self.run_sfcs_btn = ttk.Button(SFCS_button_frame, text="Correlate", command=self.run_SFCS)
        self.run_sfcs_btn.pack(side=tk.LEFT, padx=5)
        self.register_busy_widget(self.run_sfcs_btn)

        # Display frame for SFCS
        SFCS_display_frame = ttk.LabelFrame(SFCS_frame, text="SFCS curves")
        SFCS_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Matplotlib figure for SFCS display
        self.SFCS_fig = Figure(figsize=(6, 6), dpi=100, facecolor='none')  # Changed to 'none' for better theming
        self.SFCS_canvas = FigureCanvasTkAgg(self.SFCS_fig, SFCS_display_frame)
        self.SFCS_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Navigation toolbar
        SFCS_toolbar = NavigationToolbar2Tk(self.SFCS_canvas, SFCS_display_frame)
        SFCS_toolbar.update()

    def browse_metadata_file(self):
        filename = filedialog.askopenfilename(
            title="Select metadata CZI file",
            filetypes=[("CZI files", "*.czi"), ("All files", "*.*")]
        )
        if filename:
            self.file_for_metadata.set(filename)
            
    def create_fcs_fit_tab(self):
        fcs_frame = ttk.Frame(self.notebook)
        self.notebook.add(fcs_frame, text="FCS Fitting")

        params_frame = ttk.LabelFrame(fcs_frame, text="FCS Fit Parameters", padding=10)
        params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        row = 0
        ttk.Label(params_frame, text="Single CSV:").grid(row=row, column=0, sticky="w")
        self.fcsfit_csv = tk.StringVar()
        e = ttk.Entry(params_frame, textvariable=self.fcsfit_csv, width=28)
        e.grid(row=row, column=1, sticky="ew")
        b = ttk.Button(params_frame, text="Browse", command=self.browse_fcsfit_csv)
        b.grid(row=row, column=2, padx=5)
        self.register_busy_widget(e)
        self.register_busy_widget(b)

        row += 1
        ttk.Label(params_frame, text="Batch folder:").grid(row=row, column=0, sticky="w")
        self.fcsfit_folder = tk.StringVar()
        e2 = ttk.Entry(params_frame, textvariable=self.fcsfit_folder, width=28)
        e2.grid(row=row, column=1, sticky="ew")
        b2 = ttk.Button(params_frame, text="Browse", command=self.browse_fcsfit_folder)
        b2.grid(row=row, column=2, padx=5)
        self.register_busy_widget(e2)
        self.register_busy_widget(b2)


        row += 1
        ttk.Label(params_frame, text="Batch file pattern:").grid(row=row, column=0, sticky="w")
        self.fcsfit_pattern = tk.StringVar(value="*_ACF_ch0_ar_bg.csv")
        pattern_entry = ttk.Entry(params_frame, textvariable=self.fcsfit_pattern, width=28)
        pattern_entry.grid(row=row, column=1, sticky="ew")
        self.register_busy_widget(pattern_entry)
        row += 1
        ttk.Label(params_frame, text="Model:").grid(row=row, column=0, sticky="w")
        self.fcsfit_model = tk.StringVar(value="g3diffCal")
        cmb = ttk.Combobox(
            params_frame,
            textvariable=self.fcsfit_model,
            values=[
                    "g3diffCal", "g3diffBlinkCal", "g3diff", "g3diffOffset", "g3diffBlink", "g3diffBlinkOffset", 
                    "g3diffDoubleBlink", "g3lorentzianZ", "g3lorentzianZCal", "g3anomalousDiff", "g3anomalousDiffBlink", 
                    "g3diffTwoComponents", "g3diffTwoComponentsBlink", "siFCS", "siFCSTwoComponents", "g3diffMEMFCS", 
                    "g3diffLargeParticles", "g2diff", "g2diffTwoComponents", "g2diffOffset", "g2diffBlink","g2diffSFCS",

            ],
            width=22,
        )
        cmb.grid(row=row, column=1, sticky="w")
        self.register_busy_widget(cmb)
        # somewhere in create_fcs_fit_tab after you create the model combobox:
        self.fcs_initparams_host = ttk.LabelFrame(params_frame, text="Initial parameters (model-dependent)", padding=10)
        self.fcs_initparams_host.grid(row=row+1, column=0, columnspan=3, sticky="ew", pady=10)
        self.fcs_param_vars = {}       # key -> tk.StringVar (current visible ones)
        self.fcs_param_cache = {}      # key -> last user-entered string (persists across model switches)

        # rebuild once initially
        self._rebuild_fcs_param_editor()

        # bind combobox change
        # bind combobox change (visibility will work once calib frame exists)
        cmb.bind("<<ComboboxSelected>>", lambda e: (self._rebuild_fcs_param_editor(), self._update_fcs_calibration_visibility()))

        row += 2
        ttk.Label(params_frame, text="Tau min (s):").grid(row=row, column=0, sticky="w")
        self.fcsfit_tau_min = tk.StringVar(value="1e-6")
        ttk.Entry(params_frame, textvariable=self.fcsfit_tau_min, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Tau max (s):").grid(row=row, column=0, sticky="w")
        self.fcsfit_tau_max = tk.StringVar(value="1.0")
        ttk.Entry(params_frame, textvariable=self.fcsfit_tau_max, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="PSF radius (µm):").grid(row=row, column=0, sticky="w")
        self.fcsfit_psf_radius = tk.StringVar(value="0.25")
        ttk.Entry(params_frame, textvariable=self.fcsfit_psf_radius, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="PSF aspect ratio:").grid(row=row, column=0, sticky="w")
        self.fcsfit_psf_ar = tk.StringVar(value="5.0")
        ttk.Entry(params_frame, textvariable=self.fcsfit_psf_ar, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Experiment T (°C):").grid(row=row, column=0, sticky="w")
        self.fcsfit_expt_T = tk.StringVar(value="30")
        ttk.Entry(params_frame, textvariable=self.fcsfit_expt_T, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        self.fcsfit_calib_frame = ttk.LabelFrame(params_frame, text="Calibration parameters", padding=8)
        self.fcsfit_calib_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        
        r2 = 0
        ttk.Label(self.fcsfit_calib_frame, text="Given D (µm²/s):").grid(row=r2, column=0, sticky="w")
        self.fcsfit_givenD = tk.StringVar(value="435")
        ttk.Entry(self.fcsfit_calib_frame, textvariable=self.fcsfit_givenD, width=10).grid(row=r2, column=1, sticky="w")
        # now that calib frame exists, set correct initial visibility
        self._update_fcs_calibration_visibility()
        r2 += 1
        ttk.Label(self.fcsfit_calib_frame, text="Given D temp (°C):").grid(row=r2, column=0, sticky="w")
        self.fcsfit_givenD_T = tk.StringVar(value="25")
        ttk.Entry(self.fcsfit_calib_frame, textvariable=self.fcsfit_givenD_T, width=10).grid(row=r2, column=1, sticky="w")

        row += 1
        btn_frame = ttk.Frame(params_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)

        self.fcsfit_run_btn = ttk.Button(btn_frame, text="Run Fit", command=self.run_fcsfit)
        self.fcsfit_run_btn.pack(side=tk.LEFT, padx=5)
        self.register_busy_widget(self.fcsfit_run_btn)

        # Right side: log/plot placeholder (reuse Results tab for logs)
        display_frame = ttk.LabelFrame(fcs_frame, text="Info", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(
            display_frame,
            text="FCS fitting outputs are saved to a Results/ folder next to each input CSV.\nSee the Results & Logs tab for progress.",
            justify="left"
        ).pack(anchor="nw")
        self.fcsfit_fig = Figure(figsize=(8, 6), dpi=100, facecolor="gray")
        self.fcsfit_canvas = FigureCanvasTkAgg(self.fcsfit_fig, display_frame)
        self.fcsfit_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.fcsfit_canvas, display_frame)
        toolbar.update()
    def _rebuild_fcs_param_editor(self):
        # clear previous widgets
        for w in self.fcs_initparams_host.winfo_children():
            w.destroy()

        defaults = self.fcs_default_initial_params()
        model = self.fcsfit_model.get()
        keys_map = self.fcs_model_param_keys()
        keys = keys_map.get(model, ["N", "tau diffusion"])  # fallback

        self.fcs_param_vars = {}

        # grid
        self.fcs_initparams_host.grid_columnconfigure(1, weight=1)

        for r, key in enumerate(keys):
            ttk.Label(self.fcs_initparams_host, text=key).grid(row=r, column=0, sticky="w", pady=2)

            # prefer cached user input if present, else default
            initial_text = self.fcs_param_cache.get(key, str(defaults.get(key, "")))
            v = tk.StringVar(value=initial_text)

            e = ttk.Entry(self.fcs_initparams_host, textvariable=v, width=18)
            e.grid(row=r, column=1, sticky="ew", pady=2)

            # when user edits, keep it in cache so switching models doesn't lose it
            def _make_tracer(k, var):
                def _tr(*_):
                    self.fcs_param_cache[k] = var.get()
                return _tr

            v.trace_add("write", _make_tracer(key, v))

            self.fcs_param_vars[key] = v
            self.register_busy_widget(e)

    def _update_fcs_calibration_visibility(self):
        if not hasattr(self, "fcsfit_calib_frame"):
            return

        model = self.fcsfit_model.get()
        is_cal = "Cal" in model

        if is_cal:
            self.fcsfit_calib_frame.grid()
        else:
            self.fcsfit_calib_frame.grid_remove()
    def browse_fcsfit_csv(self):
        fn = filedialog.askopenfilename(
            title="Select FCS correlation CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if fn:
            self.fcsfit_csv.set(fn)

    def browse_fcsfit_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing correlation CSVs")
        if folder:
            self.fcsfit_folder.set(folder)

    def create_frap_tab(self):
        frap_frame = ttk.Frame(self.notebook)
        self.notebook.add(frap_frame, text="FRAP")

        params_frame = ttk.LabelFrame(frap_frame, text="FRAP Parameters", padding=10)
        params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        row = 0
        ttk.Label(params_frame, text="Single CZI:").grid(row=row, column=0, sticky="w")
        self.frap_czi = tk.StringVar()
        e1 = ttk.Entry(params_frame, textvariable=self.frap_czi, width=28)
        e1.grid(row=row, column=1, sticky="ew")
        b1 = ttk.Button(params_frame, text="Browse", command=self.browse_frap_czi)
        b1.grid(row=row, column=2, padx=5)
        self.register_busy_widget(e1)
        self.register_busy_widget(b1)

        row += 1
        ttk.Label(params_frame, text="Batch folder:").grid(row=row, column=0, sticky="w")
        self.frap_folder = tk.StringVar()
        e2 = ttk.Entry(params_frame, textvariable=self.frap_folder, width=28)
        e2.grid(row=row, column=1, sticky="ew")
        b2 = ttk.Button(params_frame, text="Browse", command=self.browse_frap_folder)
        b2.grid(row=row, column=2, padx=5)
        self.register_busy_widget(e2)
        self.register_busy_widget(b2)

        row += 1
        ttk.Label(params_frame, text="Pattern:").grid(row=row, column=0, sticky="w")
        self.frap_pattern = tk.StringVar(value="*FRAP*.czi")
        ttk.Entry(params_frame, textvariable=self.frap_pattern, width=18).grid(row=row, column=1, sticky="w")

        row += 1
        self.frap_imaging_bleach = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Imaging bleach correction", variable=self.frap_imaging_bleach).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )

        row += 1
        ttk.Label(params_frame, text="Fallback pixel size (µm):").grid(row=row, column=0, sticky="w")
        self.frap_pixel_size = tk.StringVar(value="")
        ttk.Entry(params_frame, textvariable=self.frap_pixel_size, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Initial D (px²/frame):").grid(row=row, column=0, sticky="w")
        self.frap_init_D = tk.StringVar(value="200")
        ttk.Entry(params_frame, textvariable=self.frap_init_D, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="D lower bound:").grid(row=row, column=0, sticky="w")
        self.frap_D_lb = tk.StringVar(value="100")
        ttk.Entry(params_frame, textvariable=self.frap_D_lb, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="D upper bound:").grid(row=row, column=0, sticky="w")
        self.frap_D_ub = tk.StringVar(value="1000")
        ttk.Entry(params_frame, textvariable=self.frap_D_ub, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Number of ROIs:").grid(row=row, column=0, sticky="w")
        self.frap_n_rois = tk.StringVar(value="")
        ttk.Entry(params_frame, textvariable=self.frap_n_rois, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Control ROI index (0-based):").grid(row=row, column=0, sticky="w")
        self.frap_ctrl_idx = tk.StringVar(value="")
        ttk.Entry(params_frame, textvariable=self.frap_ctrl_idx, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        btn_frame = ttk.Frame(params_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)

        self.frap_run_btn = ttk.Button(btn_frame, text="Run FRAP", command=self.run_frap)
        self.frap_run_btn.pack(side=tk.LEFT, padx=5)
        self.register_busy_widget(self.frap_run_btn)

        display_frame = ttk.LabelFrame(frap_frame, text="FRAP Display", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.frap_fig = Figure(figsize=(9, 7), dpi=100, facecolor="white")
        self.frap_canvas = FigureCanvasTkAgg(self.frap_fig, display_frame)
        self.frap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.frap_canvas, display_frame)
        toolbar.update()


    def browse_frap_czi(self):
        fn = filedialog.askopenfilename(
            title="Select FRAP CZI file",
            filetypes=[("CZI files", "*.czi"), ("All files", "*.*")]
        )
        if fn:
            self.frap_czi.set(fn)


    def browse_frap_folder(self):
        folder = filedialog.askdirectory(title="Select FRAP batch folder")
        if folder:
            self.frap_folder.set(folder)


    

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
        self.fit_browse_entry = ttk.Entry(rics_input_frame, textvariable=self.rics_file, width=25)
        self.fit_browse_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.fit_browse_btn = ttk.Button(rics_input_frame, text="Browse", command=self.browse_rics_file)
        self.fit_browse_btn.pack(side=tk.RIGHT)
        self.register_busy_widget(self.fit_browse_entry)
        self.register_busy_widget(self.fit_browse_btn)
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
        self.fit_browse_batch_entry = ttk.Entry(batch_fit_frame, textvariable=self.batch_fit_folder, width=25)
        self.fit_browse_batch_entry.grid(row=row, column=1, pady=2)
        self.fit_browse_batch_btn = ttk.Button(batch_fit_frame, text="Browse", command=self.browse_batch_fit_folder)
        self.fit_browse_batch_btn.grid(row=row, column=2, pady=2)
        self.register_busy_widget(self.fit_browse_batch_entry)
        self.register_busy_widget(self.fit_browse_batch_btn)
        ttk.Label(diff_fit_params, text="Window Size (pixels):").grid(row=row, column=0, sticky='w', pady=2)
        self.window_size_diff_map = tk.StringVar(value="32")
        ttk.Entry(diff_fit_params, textvariable=self.window_size_diff_map, width=15).grid(row=row, column=1, pady=2)

        row+=1
        ttk.Label(fit_params, text="Results file:").grid(row=row, column=0, sticky='w', pady=2)
        self.saving_path = tk.StringVar(value="./results_csv.csv")
        self.fit_browse_save_entry = ttk.Entry(fit_params, textvariable=self.saving_path, width=25)
        self.fit_browse_save_entry.grid(row=row, column=1, pady=2)
        self.fit_browse_save_btn = ttk.Button(fit_params, text="Browse", command=self.browse_save_path)
        self.fit_browse_save_btn.grid(row=row, column=2, pady=2)
        self.register_busy_widget(self.fit_browse_batch_entry)
        self.register_busy_widget(self.fit_browse_batch_btn)

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
                                   values=["2Ddiff", "3Ddiff", "2comp2Ddiff"], width=12)
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

    def browse_sfcs_input_file(self):
        """Browse for input file"""
        filename = filedialog.askopenfilename(
            title="Select input image stack",
            filetypes=[("All files", "*.*"), ("CZI files", "*.czi"),("TIFF files", "*.tif") ]
        )
        if filename:
            self.sfcs_input_file.set(filename)
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
    
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------Simulation RICS GUI-------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------           

    def run_simulation(self):
        if self._is_worker_running("sim_proc"):
            messagebox.showwarning("Warning", "Simulation is already running.")
            return
        if not self.output_path.get():
            messagebox.showwarning("Warning", "Please set an output path.")
            return

        
        self.log_message("Starting simulation (worker)...")
        self.status_var.set("Running simulation...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        sim_type = self.sim_type.get()
        img_shape = (self._safe_int(self.img_height, "Image height", 256),
                     self._safe_int(self.img_width, "Image width", 256))
        n_frames = self._safe_int(self.n_frames, "Number of frames", 25)

        cpu_n = clamp_workers(self.n_cpu.get(), max_fraction=0.8, hard_cap=64)

        # Build params dict
        params = dict(
            sim_type=sim_type,
            img_shape=img_shape,
            n_frames=n_frames,
            pixel_dwell_time_us=self._safe_float(self.pixel_dwell, "Pixel dwell", 50.0),
            pixel_size_nm=self._safe_float(self.pixel_size, "Pixel size", 20.0),
            brightness_khz=self._safe_float(self.brightness, "Brightness", 2000.0),
            n_particles=self._safe_int(self.n_particles, "Number of particles", 250),
            background=self._safe_float(self.background, "Background", 0.0),
            psf_sigma_px=self._safe_float(self.psf_sigma, "PSF sigma", 5.0),
            output_path=self.output_path.get(),
            cpu_n=cpu_n,
        )
        Dx = self._safe_float(self.diff_x, "Diffusion X", 10.0)
        Dy = self._safe_float(self.diff_y, "Diffusion Y", 10.0)

        if sim_type == "isotropic":
            params["diffusion_um2_s"] = 0.5 * (Dx + Dy)
        else:
            params["diffusion_um2_s_x"] = Dx
            params["diffusion_um2_s_y"] = Dy

        if sim_type == "anisotropic_rotated":
            params["rotation_deg"] = float(self.rotation.get())

        self.sim_queue = multiprocessing.Queue()
        self.sim_cancel_event = multiprocessing.Event()

        self.sim_proc = multiprocessing.Process(
            target=simulate_rics_process_main,
            args=(params, self.sim_queue, self.sim_cancel_event),
            daemon=False
        )
        self.sim_proc.start()
        self._poll_sim_queue()
    def _poll_sim_queue(self):
        try:
            while True:
                msg_type, payload = self.sim_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    self.progress_var.set(float(payload))
                    self.root.update_idletasks()
                elif msg_type == "cancelled":
                    self.log_message("sim cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return
                elif msg_type == "done":
                    self.set_ui_busy(False)
                    out_path = payload["output_path"]
                    self.log_message(f"Simulation saved: {out_path}  shape={payload['shape']}")

                    # load for preview
                    self.simulated_stack = tifffile.imread(out_path)
                    self.update_simulation_display()

                    self.status_var.set("Simulation completed")
                    self.progress_bar.grid_remove()
                    
                    return

                elif msg_type == "error":
                    self.log_message(payload)
                    self.status_var.set("Error")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    messagebox.showerror("Simulation Error", "Simulation failed. See log.")
                    return

        except queue.Empty:
            pass

        if self.sim_proc is not None and not self.sim_proc.is_alive():
            self.status_var.set("Error")
            self.log_message("Simulation worker terminated unexpectedly.")
            self.progress_bar.grid_remove()
            self.set_ui_busy(False)
            return

        self.root.after(50, self._poll_sim_queue)

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
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------SFCS GUI-------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------           
    def _poll_sfcs_queue(self):
        try:
            while True:
                msg_type, payload = self.sfcs_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    self.progress_var.set(float(payload))
                    self.root.update_idletasks()
                elif msg_type == "cancelled":
                    self.log_message("SFCS cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return
                elif msg_type == "done":
                    self.frame_data = payload["frame_data"]
                    self.aligned_data = payload["aligned_data"]
                    self.intensity_traces = payload["intensity_traces"]
                    if "correct_intensity_traces" in payload:
                        self.log_message("Bleach corrected...")
                        self.correct_intensity_traces = payload["correct_intensity_traces"]
                    self.G = payload["G"]
                    self.G_std = payload["G_std"]

                    self.status_var.set("Correlation completed")
                    self.log_message("Correlation completed")
                    self.update_SFCS_display()
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return

                elif msg_type == "error":
                    self.status_var.set("Error")
                    self.log_message(payload)
                    self.progress_bar.grid_remove()
                    messagebox.showerror("SFCS Error", "SFCS failed. See log.")
                    self.set_ui_busy(False)
                    return

        except queue.Empty:
            
            pass

        # if process died unexpectedly
        if self.sfcs_proc is not None and not self.sfcs_proc.is_alive():
            self.status_var.set("Error")
            self.log_message("SFCS process terminated unexpectedly.")
            self.progress_bar.grid_remove()
            self.set_ui_busy(False)
            return

        self.root.after(50, self._poll_sfcs_queue)
    def run_SFCS(self):
        if self._is_worker_running("sfcs_proc"):
            messagebox.showwarning("Warning", "SFCS is already running.")
            return

        if not self.sfcs_input_file.get():
            messagebox.showwarning("Warning", "Please select an input file first")
            return

        self.log_message("Starting SFCS...")
        self.status_var.set("Running correlation...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()  # since you used grid for it

        cpu_n = self._safe_int(self.n_cpu, "CPU cores", 4)
        if cpu_n >= multiprocessing.cpu_count():
            cpu_n = max(1, int(0.8 * multiprocessing.cpu_count()))

        self.sfcs_queue = multiprocessing.Queue()
        
        self.sfcs_cancel_event = multiprocessing.Event()

        # non-daemon is REQUIRED because worker will create a Pool
        self.sfcs_proc = multiprocessing.Process(
            target=sfcs_process_main_curvefit,
            args=(self.sfcs_input_file.get(), int(self.sfcs_channel.get()), cpu_n, self.sfcs_queue, self.sfcs_cancel_event, self.correct_bleach.get()),
            kwargs=dict(chunk_lines=500, max_workers=64),  # tune these
            daemon=False
        )
        self.sfcs_proc.start()

        self._poll_sfcs_queue()

    def update_SFCS_display(self):
        """Update the SFCS display with multiple views: original, aligned, intensity, and correlation"""
        if hasattr(self, 'frame_data') and self.frame_data is not None:
            self.SFCS_fig.clear()

            # Create subplots: 2x2 grid
            gs = gridspec.GridSpec(2, 2, figure=self.SFCS_fig, hspace=0.3, wspace=0.3)

            # 1. Original frame data - cropped to 100 lines (y-axis)
            ax1 = self.SFCS_fig.add_subplot(gs[0, 0])
            if self.frame_data.ndim == 3:  # (time, y, x)
                ny = self.frame_data.shape[1]
                crop_start = max(0, (ny - 100) // 2)
                crop_end = min(ny, crop_start + 100)
                cropped_original = self.frame_data[0, crop_start:crop_end, :]
            else:  # 2D case
                ny = self.frame_data.shape[0]
                crop_start = max(0, (ny - 100) // 2)
                crop_end = min(ny, crop_start + 100)
                cropped_original = self.frame_data[crop_start:crop_end, :]

            im1 = ax1.imshow(cropped_original, cmap='gray')
            ax1.set_title('Original Frame (100 lines crop)')
            ax1.axis('off')
            self.SFCS_fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # 2. Aligned data - cropped to 100 lines
            ax2 = self.SFCS_fig.add_subplot(gs[0, 1])
            if hasattr(self, 'aligned_data') and self.aligned_data is not None:
                if self.aligned_data.ndim == 3:
                    ny = self.aligned_data.shape[1]
                    crop_start = max(0, (ny - 100) // 2)
                    crop_end = min(ny, crop_start + 100)
                    cropped_aligned = self.aligned_data[0, crop_start:crop_end, :]
                else:
                    ny = self.aligned_data.shape[0]
                    crop_start = max(0, (ny - 100) // 2)
                    crop_end = min(ny, crop_start + 100)
                    cropped_aligned = self.aligned_data[crop_start:crop_end, :]

                im2 = ax2.imshow(cropped_aligned, cmap='gray')
                ax2.set_title('Aligned Frame (100 lines crop)')
                ax2.axis('off')
                self.SFCS_fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            else:
                ax2.text(0.5, 0.5, 'No aligned data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Aligned Frame')
                ax2.axis('off')

            # 3. Intensity trace from central crop region
            ax3 = self.SFCS_fig.add_subplot(gs[1, 0])
            intensity_traces = self.intensity_traces
            correct_intensity_traces = self.correct_intensity_traces
            ax3.plot(intensity_traces, 'b-', linewidth=1.5, label = "intensity", alpha = 0.5)
            ax3.plot(correct_intensity_traces, 'r--', linewidth=1.5, label = "bleach corrected intensity", alpha = 0.5)

            ax3.set_title('Average Intensity vs Frame')
            ax3.set_xlabel('Frame')
            ax3.set_ylabel('Intensity')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            # 4. SFCS Correlation curve - updated plotting style
            ax4 = self.SFCS_fig.add_subplot(gs[1, 1])
            if hasattr(self, 'G') and self.G is not None:
                if hasattr(self, 'G_std') and self.G_std is not None:
                    # Main correlation curve with uncertainty band
                    ax4.semilogx(self.G[:, 0], self.G[:, 1], 'k-', linewidth=2, label='G(τ)')
                    ax4.fill_between(self.G[:, 0],
                                     self.G[:, 1] - self.G_std,
                                     self.G[:, 1] + self.G_std,
                                     alpha=0.3, color='gray', label='±1σ (Wohland)')
                    ax4.set_xlabel('Lag time (s)')
                    ax4.set_ylabel('G(τ)')
                    ax4.set_title('SFCS Autocorrelation')
                    ax4.grid(True, alpha=0.3)
                    ax4.legend()
                else:
                    # Just the main curve if no std available
                    ax4.semilogx(self.G[:, 0], self.G[:, 1], 'k-', linewidth=2, label='G(τ)')
                    ax4.set_xlabel('Lag time (s)')
                    ax4.set_ylabel('G(τ)')
                    ax4.set_title('SFCS Autocorrelation')
                    ax4.grid(True, alpha=0.3)
                    ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('SFCS Correlation')

            self.SFCS_canvas.draw()
            root, ext = os.path.splitext(str(self.sfcs_input_file.get()))
            self.SFCS_fig.savefig(root+"_correlation.svg",dpi=300, bbox_inches='tight', facecolor='white')
            
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------RICS Export GUI-------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------  
    def _poll_export_queue(self):
        try:
            while True:
                msg_type, payload = self.export_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    self.progress_var.set(float(payload))
                    self.root.update_idletasks()
                elif msg_type == "cancelled":
                    self.log_message("export cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return
                elif msg_type == "done":
                    self.set_ui_busy(False)
                    rics_path = payload["rics_output"]
                    sd_path = payload["sd_output"]

                    # Load results from disk (recommended: do not pass arrays via queue)
                    self.current_rics_map = tifffile.imread(rics_path)
                    self.current_sd_map = tifffile.imread(sd_path)

                    self.rics_file.set(rics_path)
                    self.log_message(f"RICS map saved to: {rics_path}")
                    self.log_message(f"Uncertainty map saved to: {sd_path}")
                    self.log_message(f"RICS map shape: {self.current_rics_map.shape}")

                    # If you want to display raw/corrected frame previews, either:
                    #  - have the worker also save them, and load here, OR
                    #  - set these to None (display will just show maps)
                    self.current_image_stack = tifffile.imread(payload["tiff_output"])
                    self.current_corrected_stack = tifffile.imread(payload["corrected_tiff_output"])

                    self.update_rics_display()

                    self.status_var.set("Ready")
                    self.progress_bar.grid_remove()
                    return

                elif msg_type == "error":
                    self.set_ui_busy(False)
                    self.status_var.set("Error")
                    self.log_message(payload)
                    self.progress_bar.grid_remove()
                    messagebox.showerror("RICS Export Error", "Export failed. See log for traceback.")
                    return

        except queue.Empty:
            pass


        # process died unexpectedly
        if self.export_proc is not None and not self.export_proc.is_alive():
            self.set_ui_busy(False)
            self.status_var.set("Error")
            self.log_message("Export worker terminated unexpectedly.")
            self.progress_bar.grid_remove()
            return

        self.root.after(50, self._poll_export_queue)
    
    def export_rics(self):
        if self._is_worker_running("export_proc"):
            messagebox.showwarning("Warning", "RICS export is already running.")
            return
        """Export RICS map using export worker (process-based)."""
        if not self.input_file.get() and not self.batch_input_folder.get():
            messagebox.showwarning("Warning", "Please select an input file or a folder for batch processing")
            return

        # Batch mode (optional: see next section)
        if not self.input_file.get() and self.batch_input_folder.get():
            self._start_export_rics_batch()
            return

        # Single file
        input_file = self.input_file.get()

        self.log_message("Starting RICS export (worker)...")
        self.status_var.set("Exporting RICS...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        params = dict(
            input_file=input_file,
            channel=self._safe_int(self.channel, "Channel", 0),
            crop_factor=self._safe_float(self.crop_factor, "Crop factor", 0.5),
            window_size=self._safe_int(self.window_size, "Window size", 3),
            correct_drift=bool(self.correct_drift.get()),
        )

        self.export_queue = multiprocessing.Queue()
        self.export_cancel_event = multiprocessing.Event()


        self.export_proc = multiprocessing.Process(
            target=export_rics_process_main,
            args=(params, self.export_queue, self.export_cancel_event),
            daemon=False
        )
        self.export_proc.start()
        self._poll_export_queue()


    def _poll_export_queue_batch(self):
        try:
            while True:
                msg_type, payload = self.export_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    # progress within this file (0..100) -> map to batch fraction
                    per_file = float(payload) / 100.0
                    done_files = self._batch_export_index - 1
                    total_files = len(self._batch_export_files)
                    overall = 100.0 * (done_files + per_file) / total_files
                    self.progress_var.set(overall)
                    self.root.update_idletasks()

                elif msg_type == "cancelled":
                    self.log_message("export cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return
                elif msg_type == "done":
                    self.set_ui_busy(False)
                    self.log_message(f"Saved: {payload['rics_output']}")
                    # next file
                    self._run_next_export_in_batch()
                    return

                elif msg_type == "error":
                    self.set_ui_busy(False)
                    self.status_var.set("Error")
                    self.log_message(payload)
                    self.progress_bar.grid_remove()
                    messagebox.showerror("Batch Export Error", "A file failed. See log.")
                    return

        except queue.Empty:
            pass

        if self.export_proc is not None and not self.export_proc.is_alive():
            self.set_ui_busy(False)
            self.status_var.set("Error")
            self.log_message("Export worker terminated unexpectedly (batch).")
            self.progress_bar.grid_remove()
            return

        self.root.after(50, self._poll_export_queue_batch)

    def _start_export_rics_batch(self):
        files = get_files_from_folder(self.batch_input_folder.get(), ".czi", "")
        if not files:
            messagebox.showwarning("Warning", "No .czi files found in the selected folder.")
            return

        self._batch_export_files = files
        self._batch_export_index = 0

        self.log_message(f"Starting batch RICS export for {len(files)} files...")
        self.status_var.set("Batch exporting RICS...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        self._run_next_export_in_batch()

    def _run_next_export_in_batch(self):
        if self._batch_export_index >= len(self._batch_export_files):
            self.log_message("Batch export completed.")
            self.status_var.set("Ready")
            self.progress_bar.grid_remove()
            return

        input_file = self._batch_export_files[self._batch_export_index]
        self._batch_export_index += 1

        self.log_message(f"[{self._batch_export_index}/{len(self._batch_export_files)}] Exporting: {input_file}")

        params = dict(
            input_file=input_file,
            channel=int(self.channel.get()),
            crop_factor=float(self.crop_factor.get()),
            window_size=int(self.window_size.get()),
            correct_drift=bool(self.correct_drift.get()),
        )

        self.export_queue = multiprocessing.Queue()
        self.export_cancel_event = multiprocessing.Event()
        self.export_proc = multiprocessing.Process(
            target=export_rics_process_main,
            args=(params, self.export_queue,self.export_cancel_event),
            daemon=False
        )
        self.export_proc.start()
        self._poll_export_queue_batch()






    
        

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


            
            # Create our own display
            gs = gridspec.GridSpec(2, 3, figure=self.rics_fig, width_ratios=[1, 1, 2])

            # Raw image
            if self.current_image_stack is not None:
                ax1 = self.rics_fig.add_subplot(gs[0, 0])
                ax1.imshow(self.current_image_stack, cmap='gray')
                ax1.set_title("Raw Image (Frame 0)")
                ax1.axis('off')
            # Corrected image
            if self.current_corrected_stack is not None:
                ax2 = self.rics_fig.add_subplot(gs[0, 1])
                ax2.imshow(self.current_corrected_stack, cmap='gray')
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


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------RICS Fitting GUI-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- 

    def run_fitting(self):
        if self._is_worker_running("fit_proc"):
            messagebox.showwarning("Warning", "RICS fitting is already running.")
            return

        if not self.rics_file.get() and not self.batch_fit_folder.get():
            messagebox.showwarning("Warning", "Please load a RICS map first or select a batch folder")
            return

        self.log_message("Starting RICS fitting (worker)...")
        self.status_var.set("Running fitting...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        # build params
        base_params = dict(
            saving_path=self.saving_path.get(),
            crop_fast=self._safe_float(self.fit_crop_fast, "Crop fast", 0.5),
            crop_slow=self._safe_float(self.fit_crop_slow, "Crop slow", 0.5),
            diffusion_model=self.diffusion_model.get(),
            channel_to_use=self._safe_int(self.channel_to_use, "Channel", 0),
            fit_pixel_size_nm=self._safe_float(self.fit_pixel_size, "Pixel size", 20.0),
            fit_pixel_dwell_us=self._safe_float(self.fit_pixel_dwell, "Pixel dwell", 50.0),
            fit_line_time_ms=self._safe_float(self.fit_line_time, "Line time", 12.8),
            psf_size_xy_um=self._safe_float(self.fit_psf_xy, "PSF XY", 0.2),
            psf_aspect_ratio=self._safe_float(self.fit_psf_aspect, "PSF aspect", 5.0),
            do_fit_1d=bool(self.fit_1d_var.get()),
        )

        if self.rics_file.get():
            params = dict(base_params, mode="single", rics_file=self.rics_file.get())
        else:
            files = get_files_from_folder(self.batch_fit_folder.get(), ".tif", "RICScorr")
            params = dict(base_params, mode="batch", rics_files=files)

        self.fit_queue = multiprocessing.Queue()
        
        self.fit_cancel_event = multiprocessing.Event()

        self.fit_proc = multiprocessing.Process(
            target=fit_rics_process_main,
            args=(params, self.fit_queue, self.fit_cancel_event),
            daemon=False
        )
        self.fit_proc.start()
        self._poll_fit_queue()

    def _poll_fit_queue(self):
        try:
            while True:
                msg_type, payload = self.fit_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    self.progress_var.set(float(payload))
                    self.root.update_idletasks()
                elif msg_type == "cancelled":
                    self.log_message("fit cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return
                elif msg_type == "file_start":
                    self.set_ui_busy(True)
                    self.log_message(f"Fitting {payload['index']}/{payload['total']}: {payload['file']}")

                elif msg_type == "file_done":
                    self.set_ui_busy(True)
                    summary = payload["summary"]
                    npz_path = payload["npz_path"]
                    self.log_message(f"Done: D={summary.get('D', summary.get('D1'))}  file={summary['filepath']}")
                    self.current_file = summary["filepath"]
                    # For batch you might not want to update plots each time; optional:
                    self._load_fit_npz_and_update(npz_path, summary)
                    

                elif msg_type == "done":
                    self.set_ui_busy(False)
                    # single done has summary/npz_path; batch done just has n_total
                    
                    if "npz_path" in payload:
                        self._load_fit_npz_and_update(payload["npz_path"], payload["summary"])
                        summary = payload["summary"]
                        self.current_file = summary["filepath"]
                    self.status_var.set("Ready")
                    self.log_message("Fit completed")
                    self.progress_bar.grid_remove()
                    return

                elif msg_type == "error":
                    self.set_ui_busy(False)
                    self.status_var.set("Error")
                    self.log_message(payload)
                    self.progress_bar.grid_remove()
                    messagebox.showerror("Fitting Error", "Fitting failed. See log.")
                    return

        except queue.Empty:
            pass

        if self.fit_proc is not None and not self.fit_proc.is_alive():
            self.set_ui_busy(False)
            self.status_var.set("Error")
            self.log_message("Fit worker terminated unexpectedly.")
            self.progress_bar.grid_remove()
            return

        self.root.after(50, self._poll_fit_queue)

    def _load_fit_npz_and_update(self, npz_path, summary):
        data = np.load(npz_path, allow_pickle=False)
        rics_map = data["rics_map"]
        model = data["model"]
        residual = data["residual"]
        model_1D = data["model_1D"]
        residual_1D = data["residual_1D"]

        # rebuild fit_results dict expected by update_fitting_display()
        self.fit_results = {
            "rics_map": rics_map,
            "model": model,
            "residual": residual,
            "model_type": summary["model"],
            # add the keys update_fitting_display() uses:
            "diffusion_coeff": summary.get("D", np.nan),
            "diffusion_coeff1": summary.get("D1", np.nan),
            "diffusion_coeff2": summary.get("D2", np.nan),
            "amplitude": np.nan,
            "offset": summary.get("offset", np.nan),
        }
        if model_1D.size:
            self.fit_results["model_1D"] = model_1D
        if residual_1D.size:
            self.fit_results["residual_1D"] = residual_1D
            self.fit_results["diffusion_coeff_1D"] = summary.get("D_1D", np.nan)

        self.update_fitting_display()

    

    

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
                gs = gridspec.GridSpec(4, 4, figure=self.fit_fig, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1, 1])
            else:
                # Only 2D/3D results
                gs = gridspec.GridSpec(3, 4, figure=self.fit_fig, width_ratios=[1, 1, 1, 1])

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
            ax4a = self.fit_fig.add_subplot(gs[1, :])
            center = self.fit_results['rics_map'].shape[0] // 2
            x_axis = np.arange(self.fit_results['rics_map'].shape[1]) - self.fit_results['rics_map'].shape[1]//2

            ax4a.plot(x_axis, self.fit_results['rics_map'][center, :], 'ko-',
                    label='Data (Fast axis)', markersize=4, linewidth=1)
            ax4a.plot(x_axis, self.fit_results['model'][center, :], 'r-',
                    label=f'{self.fit_results["model_type"]} Fit (D={self.fit_results["diffusion_coeff"]:.3f} μm²/s)', 
                    linewidth=2)
            ax4b = self.fit_fig.add_subplot(gs[2, :])
            center = self.fit_results['rics_map'].shape[1] // 2
            x_axis = np.arange(self.fit_results['rics_map'].shape[0]) - self.fit_results['rics_map'].shape[0] // 2

            ax4b.plot(x_axis, self.fit_results['rics_map'][:, center], 'ko-',
                     label='Data (Fast axis)', markersize=4, linewidth=1)
            ax4b.plot(x_axis, self.fit_results['model'][:, center], 'r-',
                     label=f'{self.fit_results["model_type"]} Fit (D={self.fit_results["diffusion_coeff"]:.3f} μm²/s)',
                     linewidth=2)

            # Add 1D fit if available
            if 'model_1D' in self.fit_results:
                ax4a.plot(x_axis, self.fit_results['model_1D'], 'g--',
                        label=f'1D Fit (D={self.fit_results["diffusion_coeff_1D"]:.3f} μm²/s)', 
                        linewidth=2)

            ax4a.set_xlabel('Pixel lag')
            ax4a.set_ylabel('Correlation')
            ax4a.set_title('1D Cross-section Fits (fast)')
            ax4a.legend()
            ax4a.grid(True, alpha=0.3)
            ax4b.set_xlabel('Pixel lag')
            ax4b.set_ylabel('Correlation')
            ax4b.set_title('1D Cross-section Fits (slow)')
            ax4b.legend()
            ax4b.grid(True, alpha=0.3)
            # If 1D residuals available, show them
            if 'residual_1D' in self.fit_results:
                ax5 = self.fit_fig.add_subplot(gs[3, :])
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


            # root, ext = os.path.splitext(self.current_file)  # ext == ".tif"
            # fit_name = root + "_fit.png"
            # self.log_message(self, str(fit_name))
            # if self.batch_fit_folder.get():
            #     fit_name = os.path.join(self.batch_fit_folder.get(), fit_name)
            #
            # self.fit_fig.savefig(fit_name, dpi=300)

            self.fit_fig.tight_layout()
            self.fit_canvas.draw()

            if self.batch_fit_folder.get():
                root, ext = os.path.splitext(self.current_file)
                root = root + "_fit.svg"
                self.fit_fig.savefig(os.path.join(self.batch_fit_folder.get(), root),
                                 dpi=300, bbox_inches='tight', facecolor='white')
            elif self.rics_file.get():
                root, ext = os.path.splitext(self.rics_file.get())
                root = root + "_fit.svg"
                self.fit_fig.savefig(root,
                                     dpi=300, bbox_inches='tight', facecolor='white')
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
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------FCS Fitting GUI-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def fcs_default_initial_params(self) -> dict:
        # legacy defaults
        return {
            "N": 0.5,
            "tau diffusion": 1e-4,

            "Radius of the particle": 0.1,  # um

            "delta": 20,
            "F_Blink": 0.2,

            "delta1": 500,
            "delta2": 5,
            "F_B1": 0.2,
            "Sigma_F": 0.5,

            "f1": 0.25,
            "rho_D": 1e1,
            "rho_B": 2e3,

            "Gamma": 200,
            "Alpha": 1,

            "G0": 1e-4,
            "tau characteristic decay": 5,

            "G0_1": 1e-4,
            "G0_2": 1e-4,
            "tau characteristic decay short": 30,
            "tau characteristic decay long": 1000,

            "tau_D limits": [-7, -1],
            "number of diffusion components": 200,
            "number of iterations": 20000,
            # in fcs_default_initial_params()
            "offset": 0.0,
        }
    def _parse_param_value(self, s: str):
        s = s.strip()
        if s == "":
            return None

        # list like [-7, -1]
        if s.startswith("[") and s.endswith("]"):
            import ast
            return ast.literal_eval(s)

        # numeric
        try:
            if any(c in s for c in (".", "e", "E")):
                return float(s)
            return int(s)
        except ValueError:
            return s

    def fcs_model_param_keys(self) -> dict:
        """
        Model -> list of parameter keys to display in the GUI.
        Keys must exist in fcs_default_initial_params().
        """
        common = ["N", "tau diffusion"]

        return {
            # 2D
            "g2diff": common,
            "g2diffSFCS": common,
            "g2diffOffset": common + ["offset"],  # if you support offset for g2
            "g2diffBlink": common + ["delta", "F_Blink"],

            # 3D
            "g3diff": common,
            "g3diffOffset": common + ["offset"],
            "g3diffLargeParticles": common + ["Radius of the particle"],
            "g3diffBlink": common + ["delta", "F_Blink"],
            "g3diffBlinkOffset": common + ["delta", "F_Blink", "offset"],
            "g3diffDoubleBlink": common + ["delta1", "delta2", "F_B1", "Sigma_F"],
            "g3lorentzianZ": common,
            "g3lorentzianZCal": common + ["PSF aspect ratio"],  # only if you use this as an initial param in that model
            "g3anomalousDiff": ["N", "Gamma", "Alpha"],
            "g3anomalousDiffBlink": ["N", "Gamma", "Alpha", "delta", "F_Blink"],

            # Two components
            "g3diffTwoComponents": ["N", "tau diffusion", "f1", "rho_D"],
            "g2diffTwoComponents": ["N", "tau diffusion", "f1", "rho_D"],
            "g3diffTwoComponentsBlink": ["N", "tau diffusion", "f1", "rho_D", "rho_B", "F_Blink"],

            # Single exponential
            "siFCS": ["G0", "tau characteristic decay"],
            "siFCSTwoComponents": ["G0_1", "G0_2", "tau characteristic decay short", "tau characteristic decay long"],

            # MEMFCS
            "g3diffMEMFCS": ["tau_D limits", "number of diffusion components", "number of iterations"],
        }

    def run_fcsfit(self):
        if self._is_worker_running("fcsfit_proc"):
            messagebox.showwarning("Warning", "FCS fitting is already running.")
            return
        csv_path = self.fcsfit_csv.get().strip()
        folder = self.fcsfit_folder.get().strip()

        if not csv_path and not folder:
            messagebox.showwarning("Warning", "Select a single CSV or a batch folder.")
            return

        mode = "single" if csv_path else "batch"

        tau_min = self._safe_float(self.fcsfit_tau_min, "Tau min", 1e-6)
        tau_max = self._safe_float(self.fcsfit_tau_max, "Tau max", 1.0)


        initial_params = dict(self.fcs_default_initial_params())

        # apply only currently visible editor values
        for key, var in self.fcs_param_vars.items():
            val = self._parse_param_value(var.get())
            if val is not None:
                initial_params[key] = val

        # compatibility for legacy code
        if "F_Blink" in initial_params and "F_B" not in initial_params:
            initial_params["F_B"] = initial_params["F_Blink"]
        if "F_B" in initial_params and "F_Blink" not in initial_params:
            initial_params["F_Blink"] = initial_params["F_B"]

        model = self.fcsfit_model.get()

        kwargs = dict(
            fitting_model=model,
            tau_domain=(tau_min, tau_max),
            user_tau_domain=True,
            psf_radius_um=self._safe_float(self.fcsfit_psf_radius, "PSF radius", 0.25),
            psf_aspect_ratio=self._safe_float(self.fcsfit_psf_ar, "PSF aspect ratio", 5.0),
            experiment_T=self._safe_float(self.fcsfit_expt_T, "Experiment T", 30.0),
            BG_value=0.0,
            user_initial_params=True,
            initial_params=initial_params,
            goodness_of_fit_criterion=["instant_correlation_runsstest"],
            figure_display_delay=0.001,
        )

        # Calibration models only
        if "Cal" in model:
            kwargs["given_D"] = (
                float(self.fcsfit_givenD.get()),
                float(self.fcsfit_givenD_T.get())
            )
        else:
            # still provide a harmless default so downstream code won't fail
            kwargs["given_D"] = (435.0, 25.0)

        self.log_message(f"Starting FCS fit ({mode})...")
        self.status_var.set("Running FCS fitting...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        self.fcsfit_queue = multiprocessing.Queue()
        self.fcsfit_cancel_event = multiprocessing.Event()

        params = {"mode": mode, "kwargs": kwargs}
        if mode == "single":
            params["kwargs"]["csv_path"] = csv_path
        else:
            params["folder"] = folder
            params["pattern"] = self.fcsfit_pattern.get().strip()

        self.fcsfit_proc = multiprocessing.Process(
            target=fcsfit_process_main,
            args=(params, self.fcsfit_queue, self.fcsfit_cancel_event),
            daemon=False
        )
        self.fcsfit_proc.start()
        self._poll_fcsfit_queue()
    def update_fcs_fit_display(self, res: dict, write_summary: bool = True):
        """
        Plot into the embedded matplotlib canvas (like rics_fit) AND
        export SVG/CSVs into Results/ next to the input file.

        If write_summary=False, only per-file outputs are written, not the summary CSV.
        """
        self.fcsfit_fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fcsfit_fig)

        fitting_model = res["fitting_model"]
        base_path = res["base_path"]  # no ".csv"

        tau = np.asarray(res["tau"], dtype=float)
        G = np.asarray(res["G"], dtype=float)
        sigma = np.asarray(res["sigma_G"], dtype=float)
        pred = np.asarray(res["ccPrediction"], dtype=float)
        wr = np.asarray(res["weighted_r"], dtype=float)

        ax00 = self.fcsfit_fig.add_subplot(gs[0, 0])
        ax00.semilogx(tau, G, "r", label="G observed")
        ax00.semilogx(tau, pred, "g", label="G fit")
        ax00.fill_between(tau, G - sigma, G + sigma, color="b", alpha=0.2, label="±σ")
        ax00.set_xlabel("τ (s)")
        ax00.set_ylabel("G(τ)")
        ax00.legend()
        ax00.grid(True, alpha=0.3)

        ax01 = self.fcsfit_fig.add_subplot(gs[0, 1])
        ax01.semilogx(tau, wr, "b")
        ax01.axhline(0, color="k", lw=1, alpha=0.5)
        ax01.set_xlabel("τ (s)")
        ax01.set_ylabel("weighted residual")
        ax01.grid(True, alpha=0.3)

        ax10 = self.fcsfit_fig.add_subplot(gs[1, 0])
        reIMSD = None
        if fitting_model not in ["siFCS", "siFCSTwoComponents", "g3diffMEMFCS"]:
            aR = res.get("PSF_aspect_ratio", None)
            N = res.get("N", None)
            offset = res.get("offset", 0.0)

            if aR is not None and N is not None:
                reIMSD = calculate.iMSD_calc(tau, float(aR), float(N), pred, float(offset))
                ax10.loglog(tau, reIMSD)
                ax10.set_ylabel("iMSD")
            else:
                ax10.loglog(tau, G, "r", label="G observed")
                ax10.loglog(tau, pred, "g", label="G fit")
                ax10.set_ylabel("G(τ)")
        else:
            ax10.loglog(tau, G, "r", label="G observed")
            ax10.loglog(tau, pred, "g", label="G fit")
            ax10.set_ylabel("G(τ)")

        ax10.set_xlabel("τ (s)")
        ax10.grid(True, alpha=0.3)

        ax11 = self.fcsfit_fig.add_subplot(gs[1, 1])
        finite = np.isfinite(wr)
        ax11.hist(wr[finite], bins=40, density=True)
        ax11.set_xlabel("weighted residual")
        ax11.set_ylabel("density")

        self.fcsfit_fig.tight_layout()
        self.fcsfit_canvas.draw()

        # per-file outputs next to each file
        edit_path = self.fcs_make_edit_path(base_path, fitting_model)

        self.fcsfit_fig.savefig(edit_path + ".svg", dpi=300, bbox_inches="tight", facecolor="white")

        cc_fits_df = pd.DataFrame({"tau": tau, "G": G, "sigma G": sigma, "cc Fit": pred})
        cc_fits_df.to_csv(edit_path + ".csv", header=True, index=False)

        if reIMSD is not None:
            iMSD_df = pd.DataFrame({"tau": tau, "iMSD": reIMSD})
            iMSD_df.to_csv(edit_path + "_iMSD.csv", header=True, index=False)

        if write_summary:
            summary_csv = os.path.join(os.path.dirname(edit_path), f"{fitting_model}_fit_summary.csv")
            estimate = res.get("estimate_data", {})
            row = {}
            for k, v in estimate.items():
                if v == [None]:
                    continue
                if isinstance(v, list) and len(v) == 1:
                    row[k] = v[0]
                else:
                    row[k] = v

            row["Filename"] = base_path
            df = pd.DataFrame([row])

            if not os.path.exists(summary_csv):
                df.to_csv(summary_csv, header=True, index=False)
            else:
                df.to_csv(summary_csv, mode="a", header=False, index=False)

        self.log_message(f"Saved FCS outputs to: {os.path.dirname(edit_path)}")

    def fcs_make_edit_path(self, base_path: str, fitting_model: str) -> str:
        # base_path is without ".csv"
        edit_path = base_path + "_" + fitting_model
        head, tail = os.path.split(edit_path)
        results_dir = os.path.join(head, "Results")
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, tail)  # without extension
    def _poll_fcsfit_queue(self):
        try:
            while True:
                msg_type, payload = self.fcsfit_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    self.progress_var.set(float(payload))

                elif msg_type == "cancelled":
                    self.log_message("FCS fitting cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return

                elif msg_type == "file_done":
                    res = payload["res"] if isinstance(payload, dict) and "res" in payload else payload
                    self.log_message(f"Finished: {res.get('base_path')}")

                elif msg_type == "done":
                    # batch payload
                    if isinstance(payload, dict) and "summary" in payload and "last_res" in payload:
                        if payload["last_res"] is not None:
                            self.update_fcs_fit_display(payload["last_res"], write_summary=False)

                        summary = payload["summary"]
                        self.log_message(f"Batch summary CSV: {summary.get('summary_csv')}")
                        self.log_message(f"Processed {summary.get('n_ok')}/{summary.get('n_total')} files")
                        if summary.get("n_failed", 0) > 0:
                            self.log_message(f"Failed files: {summary.get('failed')}")
                    else:
                        # single-file payload
                        res = payload["res"] if isinstance(payload, dict) and "res" in payload else payload
                        self.update_fcs_fit_display(res, write_summary=True)

                        for w in res.get("warnings", []):
                            self.log_message(f"FCS warning: {w}")

                    self.set_ui_busy(False)
                    self.status_var.set("Ready")
                    self.progress_bar.grid_remove()
                    self.log_message("FCS fitting done.")
                    return

                elif msg_type == "error":
                    self.set_ui_busy(False)
                    self.status_var.set("Error")
                    self.progress_bar.grid_remove()
                    self.log_message(payload)
                    messagebox.showerror("FCS Fit Error", "FCS fitting failed. See log.")
                    return

        except queue.Empty:
            pass

        if self.fcsfit_proc is not None and not self.fcsfit_proc.is_alive():
            self.set_ui_busy(False)
            self.status_var.set("Error")
            self.progress_bar.grid_remove()
            self.log_message("FCS fit worker terminated unexpectedly.")
            return

        self.root.after(50, self._poll_fcsfit_queue)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------FRAP GUI-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- 



    def run_frap(self):
        if self._is_worker_running("frap_proc"):
            messagebox.showwarning("Warning", "FRAP analysis is already running.")
            return
        czi_path = self.frap_czi.get().strip()
        folder = self.frap_folder.get().strip()

        if not czi_path and not folder:
            messagebox.showwarning("Warning", "Select a single CZI or a batch folder.")
            return

        mode = "single" if czi_path else "batch"

        px_text = self.frap_pixel_size.get().strip()
        pixel_size_um = float(px_text) if px_text else None
        n_rois_text = self.frap_n_rois.get().strip()
        ctrl_idx_text = self.frap_ctrl_idx.get().strip()

        config = {
            "frap_pattern": self.frap_pattern.get(),
            "pixel_size_um": pixel_size_um,
            "imaging_bleach": bool(self.frap_imaging_bleach.get()),
            "n_rois": int(n_rois_text) if n_rois_text else None,
            "ctrl_idx": int(ctrl_idx_text) if ctrl_idx_text else None,
            "init": {
                "F_0": None,
                "f_bl": None,
                "f_mob": None,
                 "D": self._safe_float(self.frap_init_D, "Initial D", 200.0),
                "t_b": None,
            },
            "bounds": {
                "F_0": [0, None],
                "f_bl": [0, 1.0],
                "f_mob": [0, 1.2],
                "D": [self._safe_float(self.frap_D_lb, "D lower bound", 100.0),
                      self._safe_float(self.frap_D_ub, "D upper bound", 1000.0)],
                "t_b": [None, None],
            },
            "d_search_decades": 3,
            "outlier_z": 3.5,
        }

        self.log_message(f"Starting FRAP ({mode})...")
        self.status_var.set("Running FRAP...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        self.frap_queue = multiprocessing.Queue()
        self.frap_cancel_event = multiprocessing.Event()

        params = {"mode": mode, "config": config}
        if mode == "single":
            params["czi_path"] = czi_path
        else:
            params["folder"] = folder
            params["pattern"] = self.frap_pattern.get()

        self.frap_proc = multiprocessing.Process(
            target=frap_process_main,
            args=(params, self.frap_queue, self.frap_cancel_event),
            daemon=False
        )
        self.frap_proc.start()
        self._poll_frap_queue()


    def update_frap_display(self, res: dict):
        from matplotlib import gridspec

        self.frap_fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.frap_fig, hspace=0.44, wspace=0.33,
                               left=0.08, right=0.97, top=0.93, bottom=0.09)

        axA = self.frap_fig.add_subplot(gs[0, 0])
        axB = self.frap_fig.add_subplot(gs[0, 1])
        axC = self.frap_fig.add_subplot(gs[1, 0])
        axD = self.frap_fig.add_subplot(gs[1, 1])

        stem = res["stem"]
        t_all = np.asarray(res["t_all"], dtype=float)
        dt = float(res["dt"])
        bleach_frame = int(res["bleach_frame"])
        raw_traces = [np.asarray(x, dtype=float) for x in res["raw_traces"]]
        norm_traces = [np.asarray(x, dtype=float) for x in res["norm_traces"]]
        ctrl_idx = int(res["ctrl_idx"])
        frap_idxs = list(res["frap_idxs"])
        fit_results = res["fit_results"]
        colors = res["roi_colors"]
        imaging_bleach = bool(res["imaging_bleach"])

        tb_s = bleach_frame * dt
        T = t_all[-1] - t_all[0]

        self.frap_fig.suptitle(
            f"{stem}   [{' + imaging bleach' if imaging_bleach else 'no imaging bleach'}]",
            fontsize=12, fontweight='bold', color='#1A1A2E', y=0.99
        )

        def _style(ax, title, xl, yl):
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=9)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=7, color='#1A1A2E')
            ax.set_xlabel(xl, fontsize=10)
            ax.set_ylabel(yl, fontsize=10)

        def _vline(ax):
            ax.axvline(tb_s, color='#BBBBBB', lw=1.0, ls='--', zorder=0)
            yl = ax.get_ylim()
            ax.text(tb_s + T * 0.01, yl[1] * 0.98, 'bleach',
                    fontsize=7, color='#999999', va='top')

        _style(axA, 'Raw intensity (pre-normalisation)', 'Time [s]', 'Mean intensity [counts]')
        axA.plot(t_all, raw_traces[ctrl_idx], color='#AAAAAA', lw=1.5, ls='--',
                 label=f'ROI {ctrl_idx+1} — control')
        for k, fi in enumerate(frap_idxs):
            axA.plot(t_all, raw_traces[fi], color=colors[k], lw=2, label=f'ROI {fi+1} — FRAP')
        axA.autoscale(axis='y')
        _vline(axA)
        axA.legend(fontsize=8, frameon=False)

        _style(axB, 'Normalised fit', 'Time [s]', 'Normalised intensity [counts]')
        xs = np.linspace(0, len(t_all) - 1, 1000)
        ts = xs * dt
        for k, fi in enumerate(frap_idxs):
            if fit_results[k] is None:
                continue
            popt = fit_results[k][0]
            axB.plot(t_all, norm_traces[fi], 'o', color=colors[k], ms=2.5, alpha=0.35)
            axB.plot(ts, frap_analysis.evaluate_model(xs, popt, imaging_bleach), '-', color=colors[k], lw=2.2,
                     label=(f'ROI {fi+1}: Mobile fraction={popt[4]:.2f}, '
                            f't½={fit_results[k][2]:.2f} s'))
            if imaging_bleach:
                F_0, t_b = popt[2], popt[6]
                axB.plot(ts, F_0 * np.exp(-xs / t_b), ':', color=colors[k], lw=1.1, alpha=0.5)
        axB.autoscale(axis='y')
        _vline(axB)
        axB.legend(fontsize=8, frameon=False)

        _style(axC, 'Bleach corrected Fit', 'Time [s]', 'Intensity [counts]')
        x_rel = np.linspace(0.01, len(t_all) - 1 - bleach_frame, 600)
        t_rel = x_rel * dt
        x_abs = x_rel + bleach_frame

        for k, fi in enumerate(frap_idxs):
            if fit_results[k] is None:
                continue
            popt = fit_results[k][0]
            x_0f, R_f, F_0f, f_bl_f, f_mob_f, D_f = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
            t_b_f = popt[6] if imaging_bleach else np.inf

            post_idx = np.arange(bleach_frame, len(t_all), dtype=float)
            t_post = (post_idx - bleach_frame) * dt
            raw_post = norm_traces[fi][bleach_frame:].astype(float)
            ib_post = F_0f * np.exp(-post_idx / t_b_f)
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = (raw_post / (ib_post + 1e-9) - (1.0 - f_bl_f)) / (f_bl_f + 1e-9)

            axC.plot(t_post, np.clip(corr, -0.3, f_mob_f * 1.4), 'o', color=colors[k], ms=2.5, alpha=0.4)

            S_fit = frap_analysis._soumpasis(x_abs, x_0f, R_f, D_f)
            axC.plot(t_rel, f_mob_f * S_fit, '-', color=colors[k], lw=2.2,
                     label=(f'ROI {fi+1}: Mobile fraction={f_mob_f:.2f}, '
                            f'f_bl={f_bl_f:.2f}, t½={fit_results[k][2]:.2f} s'))
            axC.axhline(f_mob_f, color=colors[k], lw=0.8, ls='--', alpha=0.45)
            axC.axvline(fit_results[k][2], color=colors[k], lw=0.8, ls=':', alpha=0.65)

        axC.set_xlim(left=0)
        max_fm = max((fit_results[k][0][4] for k in range(len(frap_idxs)) if fit_results[k] is not None), default=1.0)
        axC.set_ylim(-0.3, max_fm * 1.3)
        axC.legend(fontsize=8, frameon=False)

        _style(axD, 'Fit residuals', 'Time since bleach (s)', 'Residuals')
        axD.axhline(0, color='#888888', lw=1.0)
        all_r = []
        for k, fi in enumerate(frap_idxs):
            if fit_results[k] is None:
                continue
            popt = fit_results[k][0]
            post_i = np.arange(bleach_frame, len(t_all), dtype=float)
            t_post = (post_i - bleach_frame) * dt
            pre_mu = float(np.nanmean(norm_traces[fi][:bleach_frame])) + 1e-9
            resid = (norm_traces[fi][bleach_frame:].astype(float) - frap_analysis.evaluate_model(post_i, popt, imaging_bleach)) / pre_mu
            all_r.extend(resid.tolist())
            win = max(3, len(resid) // 15)
            if len(resid) > win * 2:
                rm = np.convolve(resid, np.ones(win) / win, mode='valid')
                axD.plot(t_post[win // 2: win // 2 + len(rm)], rm, '-', color=colors[k], lw=1.8, label=f'ROI {fi+1}')

        axD.set_xlim(left=0)
        ylim = max(0.12, np.percentile(np.abs(all_r), 99) * 1.3) if all_r else 0.12
        axD.set_ylim(-ylim, ylim)
        axD.legend(fontsize=8, frameon=False)

        self.frap_canvas.draw()

        # save SVG from GUI figure only
        czi_path = Path(res["czi_path"])
        svg_path = czi_path.with_name(czi_path.stem + "_FRAP_overview.svg")
        self.frap_fig.savefig(svg_path, dpi=300, bbox_inches='tight', facecolor='white')
        self.log_message(f"Saved FRAP SVG: {svg_path}")


    def _poll_frap_queue(self):
        try:
            while True:
                msg_type, payload = self.frap_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    self.progress_var.set(float(payload))

                elif msg_type == "cancelled":
                    self.log_message("FRAP cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return

                elif msg_type == "file_done":
                    self.update_frap_display(payload)
                    self.log_message(f"Finished FRAP: {payload.get('czi_path')}")

                elif msg_type == "done":
                    
                    if isinstance(payload, dict) and "last_res" in payload:
                        if payload["last_res"] is not None:
                            self.update_frap_display(payload["last_res"])
                            res = payload["last_res"]
                            self.log_message(f"ROIs found: {res.get('n_rois_in_metadata', '?')}, "
                                             f"control: {res.get('ctrl_idx_source', 'auto')}")
                        self.log_message(f"FRAP batch summary: {payload}")
                    else:
                        self.update_frap_display(payload)
                        self.log_message(f"ROIs found: {payload.get('n_rois_in_metadata', '?')}, "
                                         f"control: {payload.get('ctrl_idx_source', 'auto')}")

                    self.set_ui_busy(False)
                    self.status_var.set("Ready")
                    self.progress_bar.grid_remove()
                    self.log_message("FRAP done.")
                    return

                elif msg_type == "error":
                    self.set_ui_busy(False)
                    self.status_var.set("Error")
                    self.progress_bar.grid_remove()
                    self.log_message(payload)
                    messagebox.showerror("FRAP Error", "FRAP analysis failed. See log.")
                    return

        except queue.Empty:
            pass

        if self.frap_proc is not None and not self.frap_proc.is_alive():
            self.set_ui_busy(False)
            self.status_var.set("Error")
            self.progress_bar.grid_remove()
            self.log_message("FRAP worker terminated unexpectedly.")
            return

        self.root.after(50, self._poll_frap_queue)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------Diffusion Map GUI-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- 

    
    def run_diffusion_map(self):
        if self._is_worker_running("diffmap_proc"):
            messagebox.showwarning("Warning", "Diffusion map is already running.")
            return
        if not self.input_file_diff_map.get():
            messagebox.showwarning("Warning", "Please select an input file first")
            return

        self.log_message("Starting Diffusion Map (worker)...")
        self.status_var.set("Generating diffusion map...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        cpu_n = clamp_workers(self.n_cpu.get(), max_fraction=0.8, hard_cap=64)

        params = dict(
            input_file=self.input_file_diff_map.get(),
            channel=self._safe_int(self.channel_to_use_diff_map, "Channel", 0),
            psf_xy_um=self._safe_float(self.fit_psf_xy, "PSF XY", 0.2),
            psf_aspect_ratio=self._safe_float(self.fit_psf_aspect, "PSF aspect", 5.0),
            window_size=self._safe_int(self.window_size_diff_map, "Window size", 32),
            offset=self._safe_int(self.offset_diff_map, "Offset", 16),
            diffusion_model=self.diffusion_model.get(),
            cpu_n=cpu_n,
        )

        self.diffmap_queue = multiprocessing.Queue()
        self.diffmap_cancel_event = multiprocessing.Event()

        self.diffmap_proc = multiprocessing.Process(
            target=diffusion_map_process_main,
            args=(params, self.diffmap_queue, self.diffmap_cancel_event),
            daemon=False
        )
        self.diffmap_proc.start()
        self._poll_diffmap_queue()

    def _poll_diffmap_queue(self):
        try:
            while True:
                msg_type, payload = self.diffmap_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    self.progress_var.set(float(payload))
                    self.root.update_idletasks()

                elif msg_type == "cancelled":
                    self.log_message("diffmap cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return
                elif msg_type == "done":
                    self.log_message(f"Diffusion map saved: {payload['diff_map_output']}")

                    aux = np.load(payload["aux_output"], allow_pickle=True)
                    self.diffusion_map = aux["Dmap"]
                    self.N_map = aux["Nmap"]
                    self.B_map = aux["Bmap"]
                    self.RICS_map_fast_axis_list = list(aux["fast_list"])
                    self.modelmap_fast_axis_list = list(aux["model_fast_list"])

                    # update GUI timing fields if you want
                    self.fit_pixel_size.set(str(payload["pixel_size_nm"]))
                    self.fit_pixel_dwell.set(str(payload["pixel_dwell_us"]))
                    self.fit_line_time.set(str(payload["line_time_ms"]))

                    self.update_fitting_display()

                    self.status_var.set("Ready")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return

                elif msg_type == "error":
                    self.log_message(payload)
                    self.status_var.set("Error")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    messagebox.showerror("Diffusion Map Error", "Diffusion map failed. See log.")
                    return

        except queue.Empty:
            pass

        if self.diffmap_proc is not None and not self.diffmap_proc.is_alive():
            self.status_var.set("Error")
            self.log_message("Diffusion map worker terminated unexpectedly.")
            self.progress_bar.grid_remove()
            self.set_ui_busy(False)
            return

        self.root.after(50, self._poll_diffmap_queue)
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- Vesicle Finder GUI ------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------

    def create_vesicle_tab(self):
        ves_frame = ttk.Frame(self.notebook)
        self.notebook.add(ves_frame, text="Vesicle Finder")

        params_frame = ttk.LabelFrame(ves_frame, text="Detection Parameters", padding=10)
        params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        row = 0
        ttk.Label(params_frame, text="CZI file:").grid(row=row, column=0, sticky="w")
        self.vesicle_czi = tk.StringVar()
        e1 = ttk.Entry(params_frame, textvariable=self.vesicle_czi, width=28)
        e1.grid(row=row, column=1, sticky="ew")
        b1 = ttk.Button(params_frame, text="Browse", command=self._browse_vesicle_czi)
        b1.grid(row=row, column=2, padx=5)
        self.register_busy_widget(e1)
        self.register_busy_widget(b1)

        row += 1
        ttk.Label(params_frame, text="Channel:").grid(row=row, column=0, sticky="w")
        self.vesicle_channel = tk.StringVar(value="0")
        ttk.Combobox(params_frame, textvariable=self.vesicle_channel,
                     values=["0", "1", "2", "3"], width=5).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(params_frame, text="Fallback pixel size (µm):").grid(row=row, column=0, sticky="w")
        self.vesicle_fallback_px = tk.StringVar(value="")
        ttk.Entry(params_frame, textvariable=self.vesicle_fallback_px, width=8).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(params_frame, text="Frame start:").grid(row=row, column=0, sticky="w")
        self.vesicle_frame_start = tk.StringVar(value="0")
        ttk.Entry(params_frame, textvariable=self.vesicle_frame_start, width=8).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Frame end:").grid(row=row, column=0, sticky="w")
        self.vesicle_frame_end = tk.StringVar(value="")
        ttk.Entry(params_frame, textvariable=self.vesicle_frame_end, width=8).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Frame step:").grid(row=row, column=0, sticky="w")
        self.vesicle_frame_step = tk.StringVar(value="1")
        ttk.Entry(params_frame, textvariable=self.vesicle_frame_step, width=8).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Crop margin (µm):").grid(row=row, column=0, sticky="w")
        self.vesicle_margin = tk.StringVar(value="5.0")
        ttk.Entry(params_frame, textvariable=self.vesicle_margin, width=8).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Detection method:").grid(row=row, column=0, sticky="w")
        self.vesicle_method = tk.StringVar(value="hough")
        method_cmb = ttk.Combobox(params_frame, textvariable=self.vesicle_method,
                     values=["hough", "cellpose","hough_transmitted", "otsu"], width=10)
        method_cmb.grid(row=row, column=1, sticky="w")

        # Hough-specific params (shown/hidden based on method)
        row += 1
        self.vesicle_hough_frame = ttk.LabelFrame(params_frame, text="Hough Circle Parameters", padding=5)
        self.vesicle_hough_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)

        hr = 0
        ttk.Label(self.vesicle_hough_frame, text="Min radius (µm):").grid(row=hr, column=0, sticky="w")
        self.vesicle_min_radius = tk.StringVar(value="5.0")
        ttk.Entry(self.vesicle_hough_frame, textvariable=self.vesicle_min_radius, width=8).grid(row=hr, column=1, sticky="w")

        hr += 1
        ttk.Label(self.vesicle_hough_frame, text="Max radius (µm):").grid(row=hr, column=0, sticky="w")
        self.vesicle_max_radius = tk.StringVar(value="25.0")
        ttk.Entry(self.vesicle_hough_frame, textvariable=self.vesicle_max_radius, width=8).grid(row=hr, column=1, sticky="w")

        hr += 1
        ttk.Label(self.vesicle_hough_frame, text="Radius step (µm):").grid(row=hr, column=0, sticky="w")
        self.vesicle_radius_step = tk.StringVar(value="0.5")
        ttk.Entry(self.vesicle_hough_frame, textvariable=self.vesicle_radius_step, width=8).grid(row=hr, column=1, sticky="w")

        hr += 1
        ttk.Label(self.vesicle_hough_frame, text="Canny sigma:").grid(row=hr, column=0, sticky="w")
        self.vesicle_canny_sigma = tk.StringVar(value="2.0")
        ttk.Entry(self.vesicle_hough_frame, textvariable=self.vesicle_canny_sigma, width=8).grid(row=hr, column=1, sticky="w")

        hr += 1
        ttk.Label(self.vesicle_hough_frame, text="Min distance (µm):").grid(row=hr, column=0, sticky="w")
        self.vesicle_hough_min_dist = tk.StringVar(value="10.0")
        ttk.Entry(self.vesicle_hough_frame, textvariable=self.vesicle_hough_min_dist, width=8).grid(row=hr, column=1, sticky="w")

        hr += 1
        ttk.Label(self.vesicle_hough_frame, text="Threshold:").grid(row=hr, column=0, sticky="w")
        self.vesicle_hough_thresh = tk.StringVar(value="0.3")
        ttk.Entry(self.vesicle_hough_frame, textvariable=self.vesicle_hough_thresh, width=8).grid(row=hr, column=1, sticky="w")

        # Cellpose-specific params
        row += 1
        self.vesicle_cellpose_frame = ttk.LabelFrame(params_frame, text="Cellpose Parameters", padding=5)
        self.vesicle_cellpose_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)

        cr = 0
        ttk.Label(self.vesicle_cellpose_frame, text="Model:").grid(row=cr, column=0, sticky="w")
        self.vesicle_cp_model = tk.StringVar(value="cyto3")
        ttk.Combobox(self.vesicle_cellpose_frame, textvariable=self.vesicle_cp_model,
                     values=["cyto3", "cyto2", "cyto", "nuclei"], width=10).grid(row=cr, column=1, sticky="w")

        cr += 1
        ttk.Label(self.vesicle_cellpose_frame, text="Est. diameter (µm):").grid(row=cr, column=0, sticky="w")
        self.vesicle_diameter = tk.StringVar(value="")
        ttk.Entry(self.vesicle_cellpose_frame, textvariable=self.vesicle_diameter, width=8).grid(row=cr, column=1, sticky="w")
        cr += 1
        self.vesicle_cp_gpu = tk.BooleanVar(value=False)
        gpu_cb = ttk.Checkbutton(self.vesicle_cellpose_frame, text="Use GPU",
                        variable=self.vesicle_cp_gpu)
        gpu_cb.grid(row=cr, column=0, columnspan=2, sticky="w")
        cr += 1
        self.vesicle_cp_invert = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.vesicle_cellpose_frame, text="Invert image (for ring-shaped GUVs)",
                        variable=self.vesicle_cp_invert).grid(row=cr, column=0, columnspan=2, sticky="w")
        cr += 1
        ttk.Label(self.vesicle_cellpose_frame, text="Min circularity (0-1):").grid(row=cr, column=0, sticky="w")
        self.vesicle_cp_circularity = tk.StringVar(value="0.65")
        ttk.Entry(self.vesicle_cellpose_frame, textvariable=self.vesicle_cp_circularity, width=8).grid(row=cr, column=1, sticky="w")

        cr += 1
        ttk.Label(self.vesicle_cellpose_frame, text="Max eccentricity (0-1):").grid(row=cr, column=0, sticky="w")
        self.vesicle_cp_eccentricity = tk.StringVar(value="0.5")
        ttk.Entry(self.vesicle_cellpose_frame, textvariable=self.vesicle_cp_eccentricity, width=8).grid(row=cr, column=1, sticky="w")

        cr += 1
        ttk.Label(self.vesicle_cellpose_frame, text="Min solidity (0-1):").grid(row=cr, column=0, sticky="w")
        self.vesicle_cp_solidity = tk.StringVar(value="0.85")
        ttk.Entry(self.vesicle_cellpose_frame, textvariable=self.vesicle_cp_solidity, width=8).grid(row=cr, column=1, sticky="w")
        cr += 1
        self.vesicle_cp_preprocess = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.vesicle_cellpose_frame, text="Preprocess for transmitted light",
                        variable=self.vesicle_cp_preprocess).grid(row=cr, column=0, columnspan=2, sticky="w")
        cr += 1
        self.vesicle_cp_fit_circles = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.vesicle_cellpose_frame, text="Fit circles to detected masks",
                        variable=self.vesicle_cp_fit_circles).grid(row=cr, column=0, columnspan=2, sticky="w")
        # disable if torch/CUDA not available
        try:
            import torch
            if not torch.cuda.is_available():
                gpu_cb.configure(state="disabled")
                self.vesicle_cp_gpu.set(False)
        except ImportError:
            gpu_cb.configure(state="disabled")
            self.vesicle_cp_gpu.set(False)
        # method-dependent visibility
        def _update_method_visibility(*_):
            m = self.vesicle_method.get()
            if m in ("hough", "hough_transmitted"):
                self.vesicle_hough_frame.grid()
                self.vesicle_cellpose_frame.grid_remove()
            elif m == "cellpose":
                self.vesicle_hough_frame.grid_remove()
                self.vesicle_cellpose_frame.grid()
            else:
                self.vesicle_hough_frame.grid_remove()
                self.vesicle_cellpose_frame.grid_remove()

        method_cmb.bind("<<ComboboxSelected>>", _update_method_visibility)
        _update_method_visibility()
        row += 1
        ttk.Label(params_frame, text="Min area (µm²):").grid(row=row, column=0, sticky="w")
        self.vesicle_min_area = tk.StringVar(value="200.0")
        ttk.Entry(params_frame, textvariable=self.vesicle_min_area, width=8).grid(row=row, column=1, sticky="w")
        row += 1
        btn_frame = ttk.Frame(params_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)

        self.vesicle_detect_btn = ttk.Button(btn_frame, text="Detect Vesicles", command=self._run_vesicle_detect)
        self.vesicle_detect_btn.pack(side=tk.LEFT, padx=5)
        self.register_busy_widget(self.vesicle_detect_btn)

        self.vesicle_crop_btn = ttk.Button(btn_frame, text="Crop Selected", command=self._run_vesicle_crop,
                                           state="disabled")
        self.vesicle_crop_btn.pack(side=tk.LEFT, padx=5)

        self.vesicle_crop_all_btn = ttk.Button(btn_frame, text="Export All", command=self._run_vesicle_crop_all,
                                               state="disabled")
        self.vesicle_crop_all_btn.pack(side=tk.LEFT, padx=5)

        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)

        row += 1
        ttk.Label(params_frame, text="── Membrane Straightening ──", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w"
        )

        row += 1
        ttk.Label(params_frame, text="Thickness (µm):").grid(row=row, column=0, sticky="w")
        self.vesicle_thickness = tk.StringVar(value="2.0")
        ttk.Entry(params_frame, textvariable=self.vesicle_thickness, width=8).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Intensity channel:").grid(row=row, column=0, sticky="w")
        self.vesicle_straighten_channel = tk.StringVar(value="0")
        ttk.Combobox(params_frame, textvariable=self.vesicle_straighten_channel,
                     values=["0", "1", "2", "3"], width=5).grid(row=row, column=1, sticky="w")

        row += 1
        straighten_btn_frame = ttk.Frame(params_frame)
        straighten_btn_frame.grid(row=row, column=0, columnspan=3, pady=5)

        self.vesicle_straighten_btn = ttk.Button(
            straighten_btn_frame, text="Straighten Selected",
            command=self._run_vesicle_straighten, state="disabled"
        )
        self.vesicle_straighten_btn.pack(side=tk.LEFT, padx=5)

        self.vesicle_straighten_all_btn = ttk.Button(
            straighten_btn_frame, text="Straighten All",
            command=self._run_vesicle_straighten_all, state="disabled"
        )
        self.vesicle_straighten_all_btn.pack(side=tk.LEFT, padx=5)

        # Vesicle list
        row += 1
        ttk.Label(params_frame, text="Detected vesicles (click to select):").grid(
            row=row, column=0, columnspan=3, sticky="w"
        )

        row += 1
        self.vesicle_listbox = tk.Listbox(params_frame, height=10, width=40, selectmode=tk.MULTIPLE)
        self.vesicle_listbox.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)

        # Display
        display_frame = ttk.LabelFrame(ves_frame, text="Preview", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.vesicle_fig = Figure(figsize=(8, 6), dpi=100, facecolor="gray")
        self.vesicle_canvas = FigureCanvasTkAgg(self.vesicle_fig, display_frame)
        self.vesicle_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.vesicle_canvas, display_frame)
        toolbar.update()

        # Enable click-on-image selection
        self.vesicle_canvas.mpl_connect("button_press_event", self._on_vesicle_click)

        # State
        self._vesicle_detect_result = None
        self._vesicle_selected_labels = set()

    def _browse_vesicle_czi(self):
        fn = filedialog.askopenfilename(
            title="Select CZI file",
            filetypes=[("CZI files", "*.czi"), ("All files", "*.*")]
        )
        if fn:
            self.vesicle_czi.set(fn)

    def _vesicle_common_params(self):
        diam_text = self.vesicle_diameter.get().strip()
        diameter_um = float(diam_text) if diam_text else None

        end_text = self.vesicle_frame_end.get().strip()
        frame_end = int(end_text) if end_text else None

        fallback_text = self.vesicle_fallback_px.get().strip()
        fallback_pixel_size_um = float(fallback_text) if fallback_text else None

        return dict(
            czi_path=self.vesicle_czi.get().strip(),
            channel=self._safe_int(self.vesicle_channel, "Channel", 0),
            frame_start=self._safe_int(self.vesicle_frame_start, "Frame start", 0),
            frame_end=frame_end,
            frame_step=self._safe_int(self.vesicle_frame_step, "Frame step", 1),
            crop_margin_um=self._safe_float(self.vesicle_margin, "Crop margin", 5.0),
            method=self.vesicle_method.get(),
            use_cellpose=self.vesicle_method.get() == "cellpose",
            model_type=self.vesicle_cp_model.get(),
            diameter=diameter_um,
            cellpose_gpu=bool(self.vesicle_cp_gpu.get()),
            cellpose_invert=bool(self.vesicle_cp_invert.get()),
            filter_circularity=self._safe_float(self.vesicle_cp_circularity, "Min circularity", 0.0),
            filter_eccentricity=self._safe_float(self.vesicle_cp_eccentricity, "Max eccentricity", 1.0),
            filter_solidity=self._safe_float(self.vesicle_cp_solidity, "Min solidity", 0.0),
            preprocess_transmitted=bool(self.vesicle_cp_preprocess.get()),
            fit_circles=bool(self.vesicle_cp_fit_circles.get()),
            min_area_um2=self._safe_float(self.vesicle_min_area, "Min area", 1.0),
            min_radius_um=self._safe_float(self.vesicle_min_radius, "Min radius", 1.0),
            max_radius_um=self._safe_float(self.vesicle_max_radius, "Max radius", 20.0),
            radius_step_um=self._safe_float(self.vesicle_radius_step, "Radius step", 0.5),
            canny_sigma=self._safe_float(self.vesicle_canny_sigma, "Canny sigma", 2.0),
            hough_min_distance_um=self._safe_float(self.vesicle_hough_min_dist, "Min distance", 5.0),
            hough_threshold_fraction=self._safe_float(self.vesicle_hough_thresh, "Threshold fraction", 0.3),
            fallback_pixel_size_um=fallback_pixel_size_um,
        )
    # ---- DETECT ----
    def _run_vesicle_detect(self):
        if self._is_worker_running("vesicle_proc"):
            messagebox.showwarning("Warning", "Vesicle detection is already running.")
            return

        if not self.vesicle_czi.get().strip():
            messagebox.showwarning("Warning", "Please select a CZI file.")
            return

        self.log_message("Starting vesicle detection...")
        self.status_var.set("Detecting vesicles...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        params = self._vesicle_common_params()
        params["phase"] = "detect"
        params["selected_labels"] = None

        self.vesicle_queue = multiprocessing.Queue()
        self.vesicle_cancel_event = multiprocessing.Event()

        self.vesicle_proc = multiprocessing.Process(
            target=vesicle_process_main,
            args=(params, self.vesicle_queue, self.vesicle_cancel_event),
            daemon=False,
        )
        self.vesicle_proc.start()
        self._poll_vesicle_queue()

    # ---- CROP SELECTED ----
    def _run_vesicle_crop(self):
        selected = list(self._vesicle_selected_labels)
        if not selected:
            messagebox.showwarning("Warning", "No vesicles selected. Click on the image or listbox.")
            return
        self._start_vesicle_crop(selected)

    # ---- CROP ALL ----
    def _run_vesicle_crop_all(self):
        if self._vesicle_detect_result is None:
            return
        all_labels = [v["label"] for v in self._vesicle_detect_result["vesicles"]]
        if not all_labels:
            messagebox.showwarning("Warning", "No vesicles detected.")
            return
        self._start_vesicle_crop(all_labels)

    def _start_vesicle_crop(self, labels):
        if self._is_worker_running("vesicle_proc"):
            messagebox.showwarning("Warning", "Vesicle processing is already running.")
            return

        self.log_message(f"Cropping {len(labels)} vesicle(s)...")
        self.status_var.set("Cropping vesicles...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        params = self._vesicle_common_params()
        params["phase"] = "crop"
        params["selected_labels"] = labels

        self.vesicle_queue = multiprocessing.Queue()
        self.vesicle_cancel_event = multiprocessing.Event()

        self.vesicle_proc = multiprocessing.Process(
            target=vesicle_process_main,
            args=(params, self.vesicle_queue, self.vesicle_cancel_event),
            daemon=False,
        )
        self.vesicle_proc.start()
        self._poll_vesicle_queue()

    # ---- POLL ----
    def _poll_vesicle_queue(self):
        try:
            while True:
                msg_type, payload = self.vesicle_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    self.progress_var.set(float(payload))

                elif msg_type == "cancelled":
                    self.log_message("Vesicle detection cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return

                elif msg_type == "done":
                    result = payload
                    if result["mode"] == "detect":
                        self._handle_vesicle_detect_result(result)
                    elif result["mode"] == "crop":
                        self._handle_vesicle_crop_result(result)
                    elif result["mode"] == "straighten":
                        self._handle_vesicle_straighten_result(result)

                    self.set_ui_busy(False)
                    self.status_var.set("Ready")
                    self.progress_bar.grid_remove()
                    return

                elif msg_type == "error":
                    self.set_ui_busy(False)
                    self.status_var.set("Error")
                    self.progress_bar.grid_remove()
                    self.log_message(payload)
                    messagebox.showerror("Vesicle Error", "Vesicle detection failed. See log.")
                    return

        except queue.Empty:
            pass

        if self.vesicle_proc is not None and not self.vesicle_proc.is_alive():
            self.set_ui_busy(False)
            self.status_var.set("Error")
            self.progress_bar.grid_remove()
            self.log_message("Vesicle worker terminated unexpectedly.")
            return

        self.root.after(50, self._poll_vesicle_queue)

    # ---- RESULT HANDLERS ----
    def _handle_vesicle_detect_result(self, result):
        self._vesicle_detect_result = result
        self._vesicle_selected_labels = set()
        vesicles = result["vesicles"]
        pixel_size_um = result.get("pixel_size_um", None)

        self.log_message(f"Detected {len(vesicles)} vesicle(s) in frame 0")
        self.log_message(f"Total frames in file: {result['n_total_frames']}")
        if pixel_size_um:
            self.log_message(f"Pixel size: {pixel_size_um:.4f} µm")

        if not vesicle_detection.CELLPOSE_AVAILABLE:
            self.log_message("NOTE: Cellpose not installed; used Otsu fallback segmentation.")

        if self.vesicle_method.get() in ("hough", "hough_transmitted"):
            from theatrics.vesicle.detection import OPENCV_AVAILABLE
            if OPENCV_AVAILABLE:
                self.log_message(f"Hough detection ({self.vesicle_method.get()}): using OpenCV")
            else:
                self.log_message(f"Hough detection ({self.vesicle_method.get()}): using skimage")

        if self.vesicle_method.get() == "hough":
            from theatrics.vesicle.detection import OPENCV_AVAILABLE
            if OPENCV_AVAILABLE:
                self.log_message("Hough detection: using OpenCV (fast)")
            else:
                self.log_message("Hough detection: using skimage (OpenCV not installed)")

        self.vesicle_listbox.delete(0, tk.END)
        for v in vesicles:
            if "radius_um" in v:
                info = (f"Vesicle {v['label']}: "
                        f"r={v.get('radius', '?')}px ({v.get('radius_um', 0):.1f}µm)  "
                        f"d≈{v.get('equivalent_diameter_um', 0):.1f}µm  "
                        f"area={v.get('area_um2', 0):.1f}µm²")
            else:
                info = (f"Vesicle {v['label']}: "
                        f"d≈{v.get('equivalent_diameter_um', v.get('equivalent_diameter', 0)):.1f}µm  "
                        f"area={v.get('area_um2', v.get('area', 0)):.1f}µm²")
            self.vesicle_listbox.insert(tk.END, info)

        self.vesicle_crop_btn.configure(state="normal")
        self.vesicle_crop_all_btn.configure(state="normal")
        self.vesicle_straighten_btn.configure(state="normal")
        self.vesicle_straighten_all_btn.configure(state="normal")
        self.vesicle_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._update_vesicle_display()

    def _handle_vesicle_crop_result(self, result):
        self.log_message(f"Cropped {result['n_vesicles']} vesicle(s), {result['n_frames']} frames each")
        self.log_message(f"Output directory: {result['output_dir']}")
        for p in result["output_paths"]:
            self.log_message(f"  → {p}")

    # ---- DISPLAY ----
    def _update_vesicle_display(self):
        if self._vesicle_detect_result is None:
            return

        result = self._vesicle_detect_result
        preview = result["preview_frame"]
        labels = result["labels_frame0"]
        vesicles = result["vesicles"]

        self.vesicle_fig.clear()
        gs = gridspec.GridSpec(1, 2, figure=self.vesicle_fig, width_ratios=[1, 1])

        # Left: raw image with overlay contours
        ax1 = self.vesicle_fig.add_subplot(gs[0, 0])
        ax1.imshow(preview, cmap="gray")
        ax1.set_title("Detected vesicles (click to select)")

        # draw contours
        from matplotlib.patches import Circle as MplCircle

        for v in vesicles:
            lbl = v["label"]
            color = "lime" if lbl in self._vesicle_selected_labels else "cyan"

            if "radius" in v:
                # Hough detection: draw a circle
                circ = MplCircle(
                    (v["centroid_x"], v["centroid_y"]),
                    v["radius"],
                    fill=False, edgecolor=color, linewidth=1.5, alpha=0.8
                )
                ax1.add_patch(circ)
            else:
                # Cellpose/Otsu: draw contour dots
                mask = labels == lbl
                contours_y, contours_x = np.where(
                    mask & ~scipy.ndimage.binary_erosion(mask)
                )
                ax1.scatter(contours_x, contours_y, s=0.3, c=color, alpha=0.7)

            ax1.text(
                v["centroid_x"], v["centroid_y"], str(lbl),
                color="white", fontsize=8, ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6)
            )
         # Right: label map
        ax2 = self.vesicle_fig.add_subplot(gs[0, 1])
        ax2.imshow(labels, cmap="nipy_spectral", interpolation="nearest")
        ax2.set_title("Segmentation labels")
        ax2.axis("off")
        self.vesicle_fig.tight_layout()
        self.vesicle_canvas.draw()

    # ---- CLICK ON IMAGE ----
    def _on_vesicle_click(self, event):
        if self._vesicle_detect_result is None:
            return
        if event.inaxes is None:
            return

        x, y = int(round(event.xdata)), int(round(event.ydata))
        labels = self._vesicle_detect_result["labels_frame0"]

        if y < 0 or y >= labels.shape[0] or x < 0 or x >= labels.shape[1]:
            return

        clicked_label = int(labels[y, x])
        if clicked_label == 0:
            return

        # toggle selection
        if clicked_label in self._vesicle_selected_labels:
            self._vesicle_selected_labels.discard(clicked_label)
            self.log_message(f"Deselected vesicle {clicked_label}")
        else:
            self._vesicle_selected_labels.add(clicked_label)
            self.log_message(f"Selected vesicle {clicked_label}")

        # sync listbox highlighting
        self._sync_listbox_selection()
        self._update_vesicle_display()

    def _on_listbox_select(self, event):
        self._vesicle_selected_labels = set()
        vesicles = self._vesicle_detect_result["vesicles"]
        for idx in self.vesicle_listbox.curselection():
            if idx < len(vesicles):
                self._vesicle_selected_labels.add(vesicles[idx]["label"])
        self._update_vesicle_display()

    def _sync_listbox_selection(self):
        self.vesicle_listbox.selection_clear(0, tk.END)
        vesicles = self._vesicle_detect_result["vesicles"]
        for i, v in enumerate(vesicles):
            if v["label"] in self._vesicle_selected_labels:
                self.vesicle_listbox.selection_set(i)

    def _run_vesicle_straighten(self):
        selected = list(self._vesicle_selected_labels)
        if not selected:
            messagebox.showwarning("Warning", "No vesicles selected.")
            return
        vesicles = [v for v in self._vesicle_detect_result["vesicles"] if v["label"] in selected]
        if not all("radius" in v for v in vesicles):
            messagebox.showwarning("Warning", "Straightening requires Hough-detected vesicles with known radius.")
            return
        self._start_vesicle_straighten(vesicles)

    def _run_vesicle_straighten_all(self):
        if self._vesicle_detect_result is None:
            return
        vesicles = [v for v in self._vesicle_detect_result["vesicles"] if "radius" in v]
        if not vesicles:
            messagebox.showwarning("Warning", "No vesicles with known radius. Use Hough detection.")
            return
        self._start_vesicle_straighten(vesicles)

    def _start_vesicle_straighten(self, vesicles):
        if self._is_worker_running("vesicle_proc"):
            messagebox.showwarning("Warning", "Vesicle processing is already running.")
            return

        self.log_message(f"Straightening {len(vesicles)} vesicle(s)...")
        self.status_var.set("Straightening membranes...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        pixel_size_um = self._vesicle_detect_result.get("pixel_size_um", 1.0)

        params = {
            "phase": "straighten",
            "czi_path": self._vesicle_detect_result["czi_path"],
            "vesicles": vesicles,
            "pixel_size_um": pixel_size_um,
            "thickness_um": self._safe_float(self.vesicle_thickness, "Thickness", 2.0),
            "straighten_channel": self._safe_int(self.vesicle_straighten_channel, "Channel", 0),
            "frame_start": self._safe_int(self.vesicle_frame_start, "Frame start", 0),
            "frame_end": int(self.vesicle_frame_end.get()) if self.vesicle_frame_end.get().strip() else None,
            "frame_step": self._safe_int(self.vesicle_frame_step, "Frame step", 1),
        }

        self.vesicle_queue = multiprocessing.Queue()
        self.vesicle_cancel_event = multiprocessing.Event()

        self.vesicle_proc = multiprocessing.Process(
            target=vesicle_process_main,
            args=(params, self.vesicle_queue, self.vesicle_cancel_event),
            daemon=False,
        )
        self.vesicle_proc.start()
        self._poll_vesicle_queue()

    def _handle_vesicle_straighten_result(self, result):
        self.log_message(f"Straightened {len(result['results'])} vesicle(s)")
        self.log_message(f"Output directory: {result['output_dir']}")

        for r in result["results"]:
            self.log_message(f"  Vesicle {r['vesicle_label']}:")
            self.log_message(f"    Strip TIFF: {r['tiff_path']}")
            self.log_message(f"    Profile CSV: {r['profile_csv']}")
            self.log_message(f"    Total CSV: {r['total_csv']}")

        self._update_straighten_display(result)

    def _update_straighten_display(self, result):
        results = result["results"]
        n_vesicles = len(results)
        pixel_size_um = result.get("pixel_size_um", 1.0)
        thickness_um = result.get("thickness_um", 2.0)

        self.vesicle_fig.clear()
        self.vesicle_fig.set_constrained_layout(True)
        if n_vesicles == 0:
            return

        # layout: for each vesicle, show 3 panels stacked vertically
        # row 0: straightened strip (first frame)
        # row 1: heatmap (intensity vs angle vs time)
        # row 2: total intensity vs time
        n_rows = 3
        gs = gridspec.GridSpec(n_rows, n_vesicles, figure=self.vesicle_fig,
                               height_ratios = [1,1,1])

        for vi, res in enumerate(results):
            strips = np.asarray(res["strips"], dtype=float)
            profile = np.asarray(res["intensity_profile"], dtype=float)
            total = np.asarray(res["total_intensity"], dtype=float)
            angles = np.asarray(res["angles_deg"], dtype=float)
            n_frames = res["n_frames"]
            lbl = res["vesicle_label"]
            radius_px = res["radius"]
            radius_um = radius_px * pixel_size_um

            # circumference in µm for x-axis
            circumference_um = 2.0 * np.pi * radius_um
            x_um = np.linspace(0, circumference_um, len(angles), endpoint=False)

            # y-axis for thickness in µm
            y_um = np.linspace(-thickness_um / 2, thickness_um / 2, strips.shape[1])

            # ── Panel 1: straightened strip (frame 0) ──
            ax1 = self.vesicle_fig.add_subplot(gs[0, vi])
            im1 = ax1.imshow(
                strips[0],
                aspect="auto",
                cmap="gray",
                extent=[0, circumference_um, -thickness_um / 2, thickness_um / 2],
                origin="lower",
            )
            ax1.set_xlabel("Position along membrane (µm)")
            ax1.set_ylabel("Radial (µm)")
            ax1.set_title(f"Vesicle {lbl} — frame 0\nr={radius_um:.1f}µm", fontsize=9)
            self.vesicle_fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # ── Panel 2: heatmap (time vs angle) ──
            ax2 = self.vesicle_fig.add_subplot(gs[1, vi])

            if n_frames > 1:
                im2 = ax2.imshow(
                    profile,
                    aspect="auto",
                    cmap="inferno",
                    extent=[0, circumference_um, n_frames - 1, 0],
                    interpolation="nearest",
                )
                ax2.set_xlabel("Position along membrane (µm)")
                ax2.set_ylabel("Frame")
                ax2.set_title(f"Vesicle {lbl} — intensity heatmap", fontsize=9)
                self.vesicle_fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            else:
                # single frame: show as a 1D intensity profile line plot
                ax2.plot(x_um, profile[0], "r-", linewidth=1.5)
                ax2.set_xlabel("Position along membrane (µm)")
                ax2.set_ylabel("Mean intensity")
                ax2.set_title(f"Vesicle {lbl} — membrane profile (single frame)", fontsize=9)
                ax2.grid(True, alpha=0.3)

            # ── Panel 3: total intensity vs time ──
            ax3 = self.vesicle_fig.add_subplot(gs[2, vi])

            if n_frames > 1:
                ax3.plot(range(n_frames), total, "b-", linewidth=1.5)
                ax3.set_xlabel("Frame")
                ax3.set_ylabel("Total intensity")
                ax3.set_title(f"Vesicle {lbl} — total membrane intensity", fontsize=9)
                ax3.grid(True, alpha=0.3)
            else:
                # single frame: show as a bar or text
                ax3.bar([0], [total[0]], color="steelblue", width=0.5)
                ax3.set_xlabel("Frame")
                ax3.set_ylabel("Total intensity")
                ax3.set_title(f"Vesicle {lbl} — total: {total[0]:.1f}", fontsize=9)
                ax3.set_xlim(-0.5, 0.5)

        self.vesicle_fig.tight_layout()
        self.vesicle_canvas.draw()

        # save SVG
        out_dir = result["output_dir"]
        svg_path = os.path.join(out_dir, "straighten_overview.svg")
        self.vesicle_fig.savefig(svg_path, dpi=300, bbox_inches="tight", facecolor="white")
        self.log_message(f"Saved overview SVG: {svg_path}")
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------`Results tab GUI-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- 



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

    def _safe_float(self, var, name, fallback=0.0):
        """Safely parse a tk.StringVar as float; log and return fallback on failure."""
        try:
            val = float(var.get())
            if not np.isfinite(val):
                raise ValueError("non-finite")
            return val
        except (ValueError, TypeError):
            self.log_message(f"WARNING: invalid value for '{name}', using {fallback}")
            return fallback

    def _safe_int(self, var, name, fallback=0):
        """Safely parse a tk.StringVar as int; log and return fallback on failure."""
        try:
            return int(var.get())
        except (ValueError, TypeError):
            self.log_message(f"WARNING: invalid value for '{name}', using {fallback}")
            return fallback

    def cancel_current_task(self):
        self._cleanup_mp()
        """
        Graceful cancel:
          1) signal cancel events (if present)
          2) wait briefly for processes to exit cleanly
          3) hard-terminate only if they don't stop
          4) close queues + reset UI
        """
        cancelled_any = False

        # 1) Signal cancellation (only works if your workers check cancel_event)
        for ev_attr in (
            "sfcs_cancel_event",
            "export_cancel_event",
            "fit_cancel_event",
            "sim_cancel_event",
            "diffmap_cancel_event",
            "fcsfit_cancel_event",
            "frap_cancel_event",
            "vesicle_cancel_event",

        ):
            ev = getattr(self, ev_attr, None)
            try:
                if ev is not None:
                    ev.set()
                    cancelled_any = True
            except Exception:
                pass

        # Helper to stop one process gracefully, then force if needed
        def _stop_proc(proc_attr, grace_s=2.0, kill_s=2.0):
            nonlocal cancelled_any
            p = getattr(self, proc_attr, None)
            if p is None:
                return
            try:
                if p.is_alive():
                    cancelled_any = True
                    # 2) Grace period
                    p.join(timeout=grace_s)
                    # 3) Force kill if still alive
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=kill_s)
            except Exception:
                pass
            finally:
                setattr(self, proc_attr, None)

        # 2–3) Stop any known worker processes
        for proc_attr in ("sfcs_proc", "export_proc", "fit_proc", "sim_proc", "diffmap_proc", "fcsfit_proc", "frap_proc","vesicle_proc",):
            _stop_proc(proc_attr)

        # 4) Close queues properly (prevents resource_tracker semaphore warnings)
        for qattr in ("sfcs_queue", "export_queue", "fit_queue", "sim_queue", "diffmap_queue", "fcsfit_queue", "frap_queue","vesicle_queue",):
            q = getattr(self, qattr, None)
            try:
                if q is not None:
                    q.close()
                    q.join_thread()
            except Exception:
                pass
            finally:
                setattr(self, qattr, None)

        # Clear cancel events (optional)
        for ev_attr in (
            "sfcs_cancel_event",
            "export_cancel_event",
            "fit_cancel_event",
            "sim_cancel_event",
            "diffmap_cancel_event",
            "fcsfit_cancel_event",
            "frap_cancel_event",
            "vesicle_cancel_event",
        ):
            setattr(self, ev_attr, None)

        # Reset UI
        self.progress_var.set(0.0)
        try:
            self.progress_bar.grid_remove()
        except Exception:
            pass

        self.set_ui_busy(False)

        if cancelled_any:
            self.log_message("Cancelled running task.")
            self.status_var.set("Cancelled")
        else:
            self.log_message("No running task to cancel.")
            self.status_var.set("Ready")

    def register_busy_widget(self, w):
        """Register a widget to be disabled while a task is running."""
        if not hasattr(self, "busy_widgets"):
            self.busy_widgets = []
        self.busy_widgets.append(w)

    def set_ui_busy(self, busy: bool):
        """Disable/enable registered widgets; keep Cancel usable while busy."""
        state = "disabled" if busy else "normal"

        for w in getattr(self, "busy_widgets", []):

            try:
                if w is not None:
                    w.configure(state=state)
                else:
                    pass
            except tk.TclError:
                pass  # some widgets may not support state

        # keep Cancel enabled while busy (and disabled when idle)
        if hasattr(self, "cancel_button"):
            try:
                self.cancel_button.configure(state=("normal" if busy else "disabled"))
            except tk.TclError:
                pass
    
    def _cleanup_mp(self):
        # terminate running processes
        for proc_attr in ("sfcs_proc", "export_proc", "fit_proc", "sim_proc", "diffmap_proc", "fcsfit_proc", "frap_proc","vesicle_proc"):
            p = getattr(self, proc_attr, None)
            try:
                if p is not None and p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
            except Exception:
                pass
            setattr(self, proc_attr, None)

        # close queues properly
        for q_attr in ("sfcs_queue", "export_queue", "fit_queue", "sim_queue", "diffmap_queue", "fcsfit_queue", "frap_queue","vesicle_queue"):
            q = getattr(self, q_attr, None)
            try:
                if q is not None:
                    q.close()
                    q.join_thread()   # IMPORTANT for cleaning feeder thread resources
            except Exception:
                pass
            setattr(self, q_attr, None)

    def shutdown(self):
        self._cleanup_mp()

    def on_close(self):
        """Gracefully shut down all workers before closing the window."""
        try:
            self.cancel_current_task()
            self._cleanup_mp()
        except Exception:
            pass
        finally:
            self.root.destroy()

    def _is_worker_running(self, proc_attr):
        """Return True if a worker process is currently alive."""
        p = getattr(self, proc_attr, None)

        return p is not None and p.is_alive()

    def restart_application(self):
        if not messagebox.askyesno(
            "Restart Application",
            "This will restart the software and clear all data.\n\nContinue?"
        ):
            return

        try:
            self.log_message("Restarting application...")
            self.cancel_current_task()
            self._cleanup_mp()
            self.root.destroy()
        finally:
            python = sys.executable
            os.execv(python, [python] + sys.argv)
        



    


