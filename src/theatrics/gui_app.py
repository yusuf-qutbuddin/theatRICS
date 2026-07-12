
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
# from pylibCZIrw import czi as pyczi
import multiprocessing
import queue
import tifffile
import scipy.ndimage
import json
import pandas as pd
import platform

# from pathlib import Path



# ── Heavy / workflow-specific imports are intentionally NOT done here. ──────
#
# This module is imported by theatrics.launcher at application startup,
# BEFORE the user has chosen which category (Correlation Methods / AFM /
# Imaging Methods) to open. To keep launcher startup fast and to avoid
# unconditionally importing heavy optional dependencies (tttrlib,
# pylibCZIrw, cellpose, AFMReader, scipy-heavy submodules, etc.) that a
# given session may never actually need, each worker/analysis module is
# imported LAZILY, inside the create_*_tab() method (and the run_*/
# _poll_*_queue methods that use it) for that specific tab.
#
# Python caches imports in sys.modules, so importing the same module
# lazily multiple times (e.g. every time a tab is (re)created) has no
# extra cost beyond the first import -- this is purely about deferring
# cost until the relevant tab is actually built, not about avoiding
# repeated work.
#
# theatrics.utils.file_utils and theatrics.utils.mp_utils are lightweight
# (no heavy third-party dependencies) and are used across almost every
# tab, so they remain eagerly imported here for convenience.
from theatrics.utils.file_utils import get_files_from_folder
from theatrics.utils.mp_utils import clamp_workers

# ── Tab registry for modular startup ────────────────────────────────────────
# Maps a short tab key to the name of the ModularRICSGUI method that
# creates that tab. Used both by the full/legacy application (which
# creates all tabs) and by theatrics.launcher.TheatricsLauncher, which
# creates a window containing only a chosen category's subset of tabs.
TAB_CREATORS = {
    "simulation":   "create_simulation_tab",
    "rics_export":  "create_rics_export_tab",
    "rics_fitting": "create_fitting_tab",
    "sfcs":         "create_SFCS_tab",
    "fcs_export":   "create_ptu_fcs_tab",
    "fcs_fitting":  "create_fcs_fit_tab",
    "frap":         "create_frap_tab",
    "ics":          "create_ics_tab",
    "afm":          "create_afm_tab",
    "vesicle":      "create_vesicle_tab",
    "results":      "create_results_tab",
}

# Which tab keys belong to each of the three modular-launcher categories.
#
# NOTE (flagged assumptions, not explicitly specified by the user):
#   - "simulation" (Image Simulation) is included under "Correlation
#     Methods" since it generates synthetic RICS data for testing the
#     same pipeline. Remove it from the list below if not wanted.
#   - "results" (Results & Logs) is included in EVERY category so each
#     window is self-contained with its own log/session tab.
CATEGORY_TABS = {
    "Correlation Methods": [
        "simulation", "rics_export", "rics_fitting", "sfcs",
        "fcs_export", "fcs_fitting", "results",
    ],
    "AFM": [
        "afm", "results",
    ],
    "Imaging Methods": [
        "ics", "vesicle", "frap", "results",
    ],
}

class ModularRICSGUI:
    def __init__(self, root, tabs=None, window_title="theatRICS"):
        self.root = root
        self.root.title(window_title)

        self.root.geometry("1400x900")
        self._dialog_parent = root
        # ── modular startup ──────────────────────────────────────────────────
        # `tabs` is an optional list of keys from the module-level
        # TAB_CREATORS registry, specifying which tabs to build. If None
        # (the default), ALL tabs are created -- this preserves the
        # original monolithic behaviour for any code that constructs
        # ModularRICSGUI directly, while theatrics.launcher.TheatricsLauncher
        # passes an explicit, category-specific subset instead.
        if tabs is None:
            tabs = list(TAB_CREATORS.keys())
        # "Results & Logs" is always included, even if the caller forgot
        # it, since several tabs log their output into it.
        if "results" not in tabs:
            tabs = list(tabs) + ["results"]
        self.enabled_tabs = tabs
        # self._set_dpi_scaling()   
        # self._set_default_fonts()
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
        

        # Create main interface
        self.setup_gui()
    
    def _ask_open_filename(self, **kwargs) -> str:
        """Wrapper for filedialog.askopenfilename with correct parent."""
        from tkinter import filedialog
        return filedialog.askopenfilename(parent=self._dialog_parent, **kwargs)

    def _ask_directory(self, **kwargs) -> str:
        """Wrapper for filedialog.askdirectory with correct parent."""
        from tkinter import filedialog
        return filedialog.askdirectory(parent=self._dialog_parent, **kwargs)

    def _ask_saveas_filename(self, **kwargs) -> str:
        """Wrapper for filedialog.asksaveasfilename with correct parent."""
        from tkinter import filedialog
        return filedialog.asksaveasfilename(parent=self._dialog_parent, **kwargs)
    def _showwarning(self, title, message):
        from tkinter import messagebox
        messagebox.showwarning(title, message, parent=self._dialog_parent)

    def _showerror(self, title, message):
        from tkinter import messagebox
        messagebox.showerror(title, message, parent=self._dialog_parent)

    def _askyesno(self, title, message):
        from tkinter import messagebox
        return messagebox.askyesno(title, message, parent=self._dialog_parent)

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

        # ── modular tab creation ─────────────────────────────────────────────
        # Only build the tabs requested via self.enabled_tabs. Tabs are
        # created in this fixed, sensible order regardless of the order
        # they happen to appear in self.enabled_tabs.
        _tab_order = [
            "simulation", "rics_export", "rics_fitting", "sfcs",
            "fcs_export", "fcs_fitting", "frap", "ics", "afm",
            "vesicle", "results",
        ]
        from theatrics.splash import SplashScreen
        splash = SplashScreen(self.root)
        for tab_key in _tab_order:
            if tab_key in self.enabled_tabs:
                creator_name = TAB_CREATORS[tab_key]
                getattr(self, creator_name)()
        self.root.after(2000, splash.dismiss)
        
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
        from theatrics.workers.sim_worker import simulate_rics_process_main
        self._simulate_rics_process_main = simulate_rics_process_main
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
        from theatrics.workers.export_worker import export_rics_process_main
        self._export_rics_process_main = export_rics_process_main
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
                                   values=[0,4], width=12)
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
        from theatrics.workers.sfcs_worker import sfcs_process_main_curvefit
        self._sfcs_process_main_curvefit = sfcs_process_main_curvefit
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
                                   values=["0", "1", "2", "3", "4"], width=12)  # String values for Combobox
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
        filename = self._ask_open_filename(
            title="Select metadata CZI file",
            filetypes=[("CZI files", "*.czi"), ("All files", "*.*")]
        )
        if filename:
            self.file_for_metadata.set(filename)
    

    # ═════════════════════════════════════════════════════════════════════════════
    # PTU FCS / PIE tab
    # ═════════════════════════════════════════════════════════════════════════════

    def create_ptu_fcs_tab(self):
        """FCS correlation export tab for PicoQuant PTU files (with PIE support)
        and Zeiss ConfoCor3/LSM980 .raw files."""
        from theatrics.workers.ptu_correlate_worker import ptu_correlate_worker_main
        self._ptu_correlate_worker_main = ptu_correlate_worker_main

        ptu_frame = ttk.Frame(self.notebook)
        self.notebook.add(ptu_frame, text="FCS Export")

        # ── scrollable left panel ────────────────────────────────────────────
        # We wrap in a Canvas so the long parameter list can be scrolled
        left_outer = ttk.Frame(ptu_frame)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        canvas_scroll = tk.Canvas(left_outer, width=460, highlightthickness=0)
        scrollbar     = ttk.Scrollbar(
            left_outer, orient="vertical", command=canvas_scroll.yview
        )
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        params_frame = ttk.Frame(canvas_scroll)
        canvas_scroll.create_window((0, 0), window=params_frame, anchor="nw")

        def _on_frame_configure(event):
            canvas_scroll.configure(
                scrollregion=canvas_scroll.bbox("all")
            )
        params_frame.bind("<Configure>", _on_frame_configure)

        # ── file selection ───────────────────────────────────────────────────
        row = 0
        ttk.Label(params_frame, text="Single file:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.ptu_fcs_file = tk.StringVar()
        e1 = ttk.Entry(params_frame, textvariable=self.ptu_fcs_file, width=32)
        e1.grid(row=row, column=1, sticky="ew")
        b1 = ttk.Button(
            params_frame, text="Browse", width=8,
            command=self._browse_ptu_fcs_file
        )
        b1.grid(row=row, column=2, padx=3)
        self.register_busy_widget(e1)
        self.register_busy_widget(b1)

        row += 1
        ttk.Label(params_frame, text="Batch folder:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.ptu_fcs_folder = tk.StringVar()
        e2 = ttk.Entry(params_frame, textvariable=self.ptu_fcs_folder, width=32)
        e2.grid(row=row, column=1, sticky="ew")
        b2 = ttk.Button(
            params_frame, text="Browse",
            command=self._browse_ptu_fcs_folder
        )
        b2.grid(row=row, column=2, padx=3)
        self.register_busy_widget(e2)
        self.register_busy_widget(b2)

        # ── mode toggle: standard vs PIE ────────────────────────────────────
        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1
        self.ptu_fcs_use_pie = tk.BooleanVar(value=False)
        self._ptu_pie_checkbox = ttk.Checkbutton(
            params_frame,
            text="PIE mode (Pulsed Interleaved Excitation)",
            variable=self.ptu_fcs_use_pie,
            command=self._on_ptu_pie_toggle,
        )
        self._ptu_pie_checkbox.grid(row=row, column=0, columnspan=3, sticky="w")

        # ── standard mode frame (PTU, no PIE) ────────────────────────────────
        row += 1
        self._ptu_std_frame = ttk.LabelFrame(
            params_frame, text="Standard mode", padding=6
        )
        self._ptu_std_frame.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4
        )

        sr = 0
        ttk.Label(self._ptu_std_frame, text="Routing channel:").grid(
            row=sr, column=0, sticky="w"
        )
        self.ptu_fcs_channel = tk.StringVar(value="0")
        ttk.Combobox(
            self._ptu_std_frame, textvariable=self.ptu_fcs_channel,
            values=["0", "1", "2", "3", "4"], width=5
        ).grid(row=sr, column=1, sticky="w")

        sr += 1
        ttk.Label(self._ptu_std_frame, text="Gate start (ns):").grid(
            row=sr, column=0, sticky="w"
        )
        self.ptu_fcs_gate_start = tk.StringVar(value="")
        ttk.Entry(
            self._ptu_std_frame,
            textvariable=self.ptu_fcs_gate_start, width=8
        ).grid(row=sr, column=1, sticky="w")

        sr += 1
        ttk.Label(self._ptu_std_frame, text="Gate stop (ns):").grid(
            row=sr, column=0, sticky="w"
        )
        self.ptu_fcs_gate_stop = tk.StringVar(value="")
        ttk.Entry(
            self._ptu_std_frame,
            textvariable=self.ptu_fcs_gate_stop, width=8
        ).grid(row=sr, column=1, sticky="w")

        # ── PIE mode frame ───────────────────────────────────────────────────
        row += 1
        self._ptu_pie_frame = ttk.LabelFrame(
            params_frame, text="PIE mode", padding=6
        )
        self._ptu_pie_frame.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4
        )

        pr = 0
        ttk.Label(self._ptu_pie_frame, text="Donor channel:").grid(
            row=pr, column=0, sticky="w"
        )
        self.ptu_pie_donor_ch = tk.StringVar(value="0")
        ttk.Combobox(
            self._ptu_pie_frame, textvariable=self.ptu_pie_donor_ch,
            values=["0", "1", "2", "3", "4"], width=5
        ).grid(row=pr, column=1, sticky="w")

        pr += 1
        ttk.Label(self._ptu_pie_frame, text="Acceptor channel:").grid(
            row=pr, column=0, sticky="w"
        )
        self.ptu_pie_acceptor_ch = tk.StringVar(value="1")
        ttk.Combobox(
            self._ptu_pie_frame, textvariable=self.ptu_pie_acceptor_ch,
            values=["0", "1", "2", "3", "4"], width=5
        ).grid(row=pr, column=1, sticky="w")

        pr += 1
        ttk.Label(self._ptu_pie_frame,
                  text="Prompt gate (0–1):").grid(
            row=pr, column=0, sticky="w"
        )
        pg_frame = ttk.Frame(self._ptu_pie_frame)
        pg_frame.grid(row=pr, column=1, sticky="w")
        self.ptu_pie_prompt_start = tk.StringVar(value="0.0")
        self.ptu_pie_prompt_stop  = tk.StringVar(value="0.5")
        ttk.Entry(pg_frame, textvariable=self.ptu_pie_prompt_start,
                  width=5).pack(side=tk.LEFT)
        ttk.Label(pg_frame, text=" – ").pack(side=tk.LEFT)
        ttk.Entry(pg_frame, textvariable=self.ptu_pie_prompt_stop,
                  width=5).pack(side=tk.LEFT)

        pr += 1
        ttk.Label(self._ptu_pie_frame,
                  text="Delay gate (0–1):").grid(
            row=pr, column=0, sticky="w"
        )
        dg_frame = ttk.Frame(self._ptu_pie_frame)
        dg_frame.grid(row=pr, column=1, sticky="w")
        self.ptu_pie_delay_start = tk.StringVar(value="0.5")
        self.ptu_pie_delay_stop  = tk.StringVar(value="1.0")
        ttk.Entry(dg_frame, textvariable=self.ptu_pie_delay_start,
                  width=5).pack(side=tk.LEFT)
        ttk.Label(dg_frame, text=" – ").pack(side=tk.LEFT)
        ttk.Entry(dg_frame, textvariable=self.ptu_pie_delay_stop,
                  width=5).pack(side=tk.LEFT)

        pr += 1
        self.ptu_pie_symmetric = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._ptu_pie_frame,
            text="Symmetric cross-corr (time reversal)",
            variable=self.ptu_pie_symmetric,
        ).grid(row=pr, column=0, columnspan=2, sticky="w")

        pr += 1
        ttk.Separator(
            self._ptu_pie_frame, orient="horizontal"
        ).grid(row=pr, column=0, columnspan=2, sticky="ew", pady=4)

        pr += 1
        ttk.Label(
            self._ptu_pie_frame,
            text="FRET corrections",
            font=("", 9, "bold"),
        ).grid(row=pr, column=0, columnspan=2, sticky="w")

        for lbl, attr, default in [
            ("γ factor:",            "ptu_pie_gamma",       "1.0"),
            ("Crosstalk α:",         "ptu_pie_crosstalk",   "0.0"),
            ("Direct excitation δ:", "ptu_pie_direct_exc",  "0.0"),
        ]:
            pr += 1
            ttk.Label(self._ptu_pie_frame, text=lbl).grid(
                row=pr, column=0, sticky="w"
            )
            sv = tk.StringVar(value=default)
            setattr(self, attr, sv)
            ttk.Entry(
                self._ptu_pie_frame, textvariable=sv, width=8
            ).grid(row=pr, column=1, sticky="w")

        # ── Zeiss .raw mode frame ────────────────────────────────────────────
        row += 1
        self._ptu_raw_frame = ttk.LabelFrame(
            params_frame, text="Zeiss .raw mode", padding=6
        )
        self._ptu_raw_frame.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4
        )

        _raw_ch_values = ["1 - ChS1", "2 - ChS2", "3 - Ch2", "4 - GaAsP1"]

        rr = 0
        ttk.Label(self._ptu_raw_frame, text="Channel 1:").grid(
            row=rr, column=0, sticky="w"
        )
        self.ptu_raw_ch1 = tk.StringVar(value=_raw_ch_values[3])   # default GaAsP1
        ttk.Combobox(
            self._ptu_raw_frame, textvariable=self.ptu_raw_ch1,
            values=_raw_ch_values, width=12, state="readonly"
        ).grid(row=rr, column=1, sticky="w")

        rr += 1
        ttk.Label(self._ptu_raw_frame, text="Channel 2:").grid(
            row=rr, column=0, sticky="w"
        )
        self.ptu_raw_ch2 = tk.StringVar(value=_raw_ch_values[3])   # default: autocorrelation
        ttk.Combobox(
            self._ptu_raw_frame, textvariable=self.ptu_raw_ch2,
            values=_raw_ch_values, width=12, state="readonly"
        ).grid(row=rr, column=1, sticky="w")
        ttk.Label(
            self._ptu_raw_frame,
            text="(same as Ch 1 = autocorrelation)",
            foreground="gray",
        ).grid(row=rr, column=2, sticky="w")

        rr += 1
        self.ptu_raw_symmetric = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._ptu_raw_frame,
            text="Symmetric cross-corr (time reversal)",
            variable=self.ptu_raw_symmetric,
        ).grid(row=rr, column=0, columnspan=2, sticky="w")

        rr += 1
        ttk.Label(self._ptu_raw_frame, text="Segments (Wohland SD):").grid(
            row=rr, column=0, sticky="w"
        )
        self.ptu_raw_n_segments = tk.StringVar(value="6")
        ttk.Entry(
            self._ptu_raw_frame, textvariable=self.ptu_raw_n_segments, width=8
        ).grid(row=rr, column=1, sticky="w")
        ttk.Label(
            self._ptu_raw_frame, text="(1 = no SD calc)", foreground="gray"
        ).grid(row=rr, column=2, sticky="w")

        rr += 1
        ttk.Label(self._ptu_raw_frame, text="Channel offset (s):").grid(
            row=rr, column=0, sticky="w"
        )
        self.ptu_raw_offset_s = tk.StringVar(value="0.0")
        ttk.Entry(
            self._ptu_raw_frame, textvariable=self.ptu_raw_offset_s, width=8
        ).grid(row=rr, column=1, sticky="w")

        # rr += 1
        # self.ptu_raw_correct_bleaching = tk.BooleanVar(value=False)
        # ttk.Checkbutton(
        #     self._ptu_raw_frame,
        #     text="Bleach correction (polynomial trend, per-photon weights)",
        #     variable=self.ptu_raw_correct_bleaching,
        # ).grid(row=rr, column=0, columnspan=2, sticky="w")

        rr += 1
        ttk.Label(
            self._ptu_raw_frame,
            text="Afterpulsing correction and calibration CSV are shared\n"
                 "with the sections below (same A1,tau1,A2,tau2 CSV format).",
            foreground="gray", justify="left",
        ).grid(row=rr, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # initial visibility: standard vs PIE (raw frame starts hidden;
        # it is shown only once a .raw file/folder is actually selected)
        self._on_ptu_pie_toggle()
        self._ptu_raw_frame.grid_remove()

        # ── correlation settings ─────────────────────────────────────────────
        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1
        ttk.Label(
            params_frame, text=" Correlation settings ",
            font=("", 9, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w")

        for lbl, attr, default in [
            ("Tau min (s):",       "ptu_fcs_tau_min",    "1e-6"),
            ("Tau max (s):",       "ptu_fcs_tau_max",    "1.0"),
            ("Correlator n_bins:", "ptu_fcs_nbins",      "9"),
        ]:
            row += 1
            ttk.Label(params_frame, text=lbl).grid(
                row=row, column=0, sticky="w", pady=2
            )
            sv = tk.StringVar(value=default)
            setattr(self, attr, sv)
            ttk.Entry(params_frame, textvariable=sv, width=10).grid(
                row=row, column=1, sticky="w"
            )
        row += 1
        ttk.Label(params_frame, text="Cores (batch):").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.ptu_fcs_n_cores = tk.StringVar(value="4")
        ttk.Entry(
            params_frame, textvariable=self.ptu_fcs_n_cores, width=10
        ).grid(row=row, column=1, sticky="w")
        ttk.Label(
            params_frame, text="(parallel files)", foreground="gray"
        ).grid(row=row, column=2, sticky="w")
        # ── correlation pair selection (DD / AA / DA) ────────────────────────
        row += 1
        self._ptu_ddaada_frame = ttk.LabelFrame(
            params_frame, text="Correlation pairs to compute", padding=6
        )
        self._ptu_ddaada_frame.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4
        )

        dr = 0
        ttk.Label(
            self._ptu_ddaada_frame,
            text="(only applies to two-channel measurements;\n"
                 "single-channel files always compute autocorrelation)",
            foreground="gray", justify="left",
        ).grid(row=dr, column=0, columnspan=3, sticky="w")

        dr += 1
        self.ptu_fcs_compute_dd = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._ptu_ddaada_frame,
            text="DD  (channel 1 / donor autocorrelation)",
            variable=self.ptu_fcs_compute_dd,
        ).grid(row=dr, column=0, columnspan=3, sticky="w")

        dr += 1
        self.ptu_fcs_compute_aa = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._ptu_ddaada_frame,
            text="AA  (channel 2 / acceptor autocorrelation)",
            variable=self.ptu_fcs_compute_aa,
        ).grid(row=dr, column=0, columnspan=3, sticky="w")

        dr += 1
        self.ptu_fcs_compute_da = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._ptu_ddaada_frame,
            text="DA  (channel 1 × channel 2 cross-correlation)",
            variable=self.ptu_fcs_compute_da,
        ).grid(row=dr, column=0, columnspan=3, sticky="w")
        # ── Wohland SD settings ──────────────────────────────────────────────
        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1
        ttk.Label(
            params_frame, text=" Uncertainty (Wohland SD) ",
            font=("", 9, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Window (s):").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.ptu_fcs_wohland_window = tk.StringVar(value="")
        ttk.Entry(
            params_frame,
            textvariable=self.ptu_fcs_wohland_window, width=10
        ).grid(row=row, column=1, sticky="w")
        ttk.Label(
            params_frame, text="(blank=auto)", foreground="gray"
        ).grid(row=row, column=2, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Bootstrap reps:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.ptu_fcs_n_bootstrap = tk.StringVar(value="20")
        ttk.Entry(
            params_frame,
            textvariable=self.ptu_fcs_n_bootstrap, width=6
        ).grid(row=row, column=1, sticky="w")

        # ── afterpulsing (shared between PTU and .raw modes) ────────────────
        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1
        ttk.Label(
            params_frame, text=" Afterpulsing correction ",
            font=("", 9, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w")

        row += 1
        self.ptu_fcs_use_ap = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            params_frame,
            text="Subtract afterpulsing",
            variable=self.ptu_fcs_use_ap,
            command=self._on_ptu_ap_toggle,
        ).grid(row=row, column=0, columnspan=2, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Calibration CSV:").grid(
            row=row, column=0, sticky="w"
        )
        self.ptu_fcs_ap_path  = tk.StringVar(value="")
        self.ptu_fcs_ap_entry = ttk.Entry(
            params_frame, textvariable=self.ptu_fcs_ap_path,
            width=28, state="disabled"
        )
        self.ptu_fcs_ap_entry.grid(row=row, column=1, sticky="ew")
        self.ptu_fcs_ap_btn = ttk.Button(
            params_frame, text="Browse",
            command=self._browse_ptu_ap_file, state="disabled"
        )
        self.ptu_fcs_ap_btn.grid(row=row, column=2, padx=3)
        # ── bleach / drift correction (shared: PTU polynomial undrifting
        # via FCS_Fixer, .raw polynomial bleach weights via
        # zeiss_raw_correlate.get_blcorr_weights) ───────────────────────────
        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1
        ttk.Label(
            params_frame, text=" Bleach / drift correction ",
            font=("", 9, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w")

        row += 1
        self.ptu_fcs_correct_bleaching = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            params_frame,
            text="Apply bleach/drift correction (auto polynomial trend)",
            variable=self.ptu_fcs_correct_bleaching,
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        # ── burst removal ────────────────────────────────────────────────────
        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1
        ttk.Label(
            params_frame, text=" Burst removal ",
            font=("", 9, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w")

        row += 1
        self.ptu_fcs_use_burst_removal = tk.BooleanVar(value=False)
        self._ptu_burst_removal_checkbox = ttk.Checkbutton(   # CHANGED: store reference
            params_frame,
            text="Remove bursts (auto-thresholded, per channel)",
            variable=self.ptu_fcs_use_burst_removal,
        )
        self._ptu_burst_removal_checkbox.grid(row=row, column=0, columnspan=2, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Threshold alpha:").grid(
            row=row, column=0, sticky="w"
        )
        self.ptu_fcs_burst_threshold_alpha = tk.StringVar(value="0.02")
        ttk.Entry(
            params_frame, textvariable=self.ptu_fcs_burst_threshold_alpha, width=10
        ).grid(row=row, column=1, sticky="w")
        ttk.Label(
            params_frame, text="(lower = fewer bursts flagged)", foreground="gray"
        ).grid(row=row, column=2, sticky="w")
        # ── FLCS (PTU only) ──────────────────────────────────────────────────
        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1
        ttk.Label(
            params_frame, text=" FLCS background correction ",
            font=("", 9, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w")

        row += 1
        self.ptu_fcs_use_flcs = tk.BooleanVar(value=False)
        self._ptu_flcs_checkbox = ttk.Checkbutton(
            params_frame,
            text="Apply FLCS background correction",
            variable=self.ptu_fcs_use_flcs,
        )
        self._ptu_flcs_checkbox.grid(row=row, column=0, columnspan=2, sticky="w")

        row += 1
        btn_frame = ttk.Frame(params_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)

        self.ptu_fcs_run_btn = ttk.Button(
            btn_frame, text="Run Correlation",
            command=self._run_ptu_fcs
        )
        self.ptu_fcs_run_btn.pack(side=tk.LEFT, padx=5)
        self.register_busy_widget(self.ptu_fcs_run_btn)

        # ── right panel: display ──────────────────────────────────────────────
        display_frame = ttk.LabelFrame(
            ptu_frame, text="Correlation Display", padding=10
        )
        display_frame.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        self.ptu_fcs_fig    = Figure(figsize=(9, 7), dpi=100, facecolor="white")
        self.ptu_fcs_canvas = FigureCanvasTkAgg(self.ptu_fcs_fig, display_frame)
        self.ptu_fcs_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.ptu_fcs_canvas, display_frame)
        toolbar.update()


    # ── PIE toggle ────────────────────────────────────────────────────────────────

    def _on_ptu_pie_toggle(self):
        """Show/hide the standard vs PIE parameter sections."""
        if self.ptu_fcs_use_pie.get():
            self._ptu_std_frame.grid_remove()
            self._ptu_pie_frame.grid()
        else:
            self._ptu_pie_frame.grid_remove()
            self._ptu_std_frame.grid()


    # ── browse helpers ────────────────────────────────────────────────────────────

    def _browse_ptu_fcs_file(self):
        fn = self._ask_open_filename(
            title="Select PTU or RAW file",
            filetypes=[
                ("Supported files", "*.ptu *.raw"),
                ("PicoQuant PTU",   "*.ptu"),
                ("Zeiss RAW",       "*.raw"),
                ("All files",       "*.*"),
            ],
        )
        if fn:
            self.ptu_fcs_file.set(fn)
            self._on_ptu_input_type_detected(is_raw=fn.lower().endswith(".raw"))


    def _browse_ptu_fcs_folder(self):
        folder = self._ask_directory(title="Select batch folder")
        if folder:
            self.ptu_fcs_folder.set(folder)
            self._detect_and_apply_ptu_input_type_folder(folder)


    def _detect_and_apply_ptu_input_type_folder(self, folder):
        """
        Peek into the selected batch folder to decide whether PIE/FLCS
        should remain available. If the folder contains ONLY .raw files,
        PIE/FLCS are disabled (not available for .raw). If it contains any
        .ptu files (with or without .raw mixed in), PIE/FLCS stay available
        -- individual .raw files within a mixed batch will simply fail with
        a clear "not yet implemented" message without affecting .ptu files.
        """
        import glob
        ptu_files = glob.glob(os.path.join(folder, "**", "*.ptu"), recursive=True)
        raw_files = glob.glob(os.path.join(folder, "**", "*.raw"), recursive=True)

        if raw_files and not ptu_files:
            self._on_ptu_input_type_detected(is_raw=True)
        else:
            self._on_ptu_input_type_detected(is_raw=False)
            if ptu_files and raw_files:
                self.log_message(
                    f"Batch folder contains both .ptu ({len(ptu_files)}) and "
                    f".raw ({len(raw_files)}) files. Both will be processed; "
                    f".raw files will report 'not yet implemented' until that "
                    f"pipeline is added."
                )


    def _on_ptu_input_type_detected(self, is_raw: bool):
        """
        Enable/disable PIE mode and FLCS background correction, and switch
        the channel-selection frame between PTU-style (routing channel +
        optional micro-time gate / PIE donor-acceptor) and Zeiss .raw-style
        (named LSM980 channel pair: ChS1/ChS2/Ch2/GaAsP1), based on whether
        the selected input is a Zeiss .raw file (no TCSPC micro-time
        information) or a PTU file (full TCSPC/PIE support).

        Also hides the DD/AA/DA correlation-pair-selection frame entirely
        in .raw mode, since .raw files always specify their channel pairing
        explicitly via the Channel 1 / Channel 2 comboboxes in
        _ptu_raw_frame -- DD/AA/DA would be redundant there.
        """
        if is_raw:
            self.ptu_fcs_use_pie.set(False)
            self.ptu_fcs_use_flcs.set(False)
            self.ptu_fcs_use_burst_removal.set(False)
            try:
                self._ptu_pie_checkbox.configure(state="disabled")
            except Exception:
                pass
            try:
                self._ptu_flcs_checkbox.configure(state="disabled")
            except Exception:
                pass
            try:                                         # NEW
                self._ptu_burst_removal_checkbox.configure(state="disabled")
            except Exception:
                pass

            # hide both PTU-style frames, show the .raw frame
            self._ptu_std_frame.grid_remove()
            self._ptu_pie_frame.grid_remove()
            self._ptu_raw_frame.grid()

            # NEW: DD/AA/DA is a PTU-only concept -- hide it in .raw mode
            self._ptu_ddaada_frame.grid_remove()

            self.log_message(
                "Zeiss .raw file selected -- PIE mode, FLCS background "
            "correction, and burst removal are not available for .raw "   
            "files (no TCSPC micro-time information for FLCS; burst "
            "removal for .raw files has not been implemented yet). "
            "Only ACF/CCF with Wohland SD, afterpulsing subtraction, "
            "and bleach/drift correction will be computed."
            )
        else:
            try:
                self._ptu_pie_checkbox.configure(state="normal")
            except Exception:
                pass
            try:
                self._ptu_flcs_checkbox.configure(state="normal")
            except Exception:
                pass
            try:                                         # NEW
                self._ptu_burst_removal_checkbox.configure(state="normal")
            except Exception:
                pass

            # hide the .raw frame, restore whichever PTU-style frame matches
            # the current PIE checkbox state
            self._ptu_raw_frame.grid_remove()
            self._on_ptu_pie_toggle()

            # NEW: restore DD/AA/DA selection frame for PTU mode
            self._ptu_ddaada_frame.grid()
    def _parse_raw_channel(self, combo_value: str) -> int:
        """Parse '1 - ChS1' style strings from the Zeiss .raw channel
        comboboxes into the plain integer channel index (1-4)."""
        try:
            return int(combo_value.split("-")[0].strip())
        except Exception:
            return 1

    def _browse_ptu_ap_file(self):
        fn = self._ask_open_filename(
            title="Select afterpulsing calibration CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if fn:
            self.ptu_fcs_ap_path.set(fn)


    def _on_ptu_ap_toggle(self):
        if self.ptu_fcs_use_ap.get():
            self.ptu_fcs_ap_entry.configure(state="normal")
            self.ptu_fcs_ap_btn.configure(state="normal")
        else:
            self.ptu_fcs_ap_entry.configure(state="disabled")
            self.ptu_fcs_ap_btn.configure(state="disabled")      
    def create_fcs_fit_tab(self):
        from theatrics.workers.fcsfit_worker import fcsfit_process_main
        self._fcsfit_process_main = fcsfit_process_main
        from theatrics.fcsfit import calculations as calculate
        self._calculate = calculate
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
        fn = self._ask_open_filename(
            title="Select FCS correlation CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if fn:
            self.fcsfit_csv.set(fn)

    def browse_fcsfit_folder(self):
        folder = self._ask_directory(title="Select folder containing correlation CSVs")
        if folder:
            self.fcsfit_folder.set(folder)

    def create_frap_tab(self):
        from theatrics.workers.frap_worker import frap_process_main
        self._frap_process_main = frap_process_main
        from theatrics.frap import analysis as frap_analysis
        self._frap_analysis = frap_analysis
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
        self.frap_ctrl_idx_entry = ttk.Entry(params_frame, textvariable=self.frap_ctrl_idx, width=10)
        self.frap_ctrl_idx_entry.grid(row=row, column=1, sticky="w")

        row += 1
        self.frap_no_control = tk.BooleanVar(value=False)
        frap_no_ctrl_cb = ttk.Checkbutton(
            params_frame,
            text="No control ROI (normalise to own pre-bleach mean)",
            variable=self.frap_no_control,
            command=self._on_frap_no_control_toggle,
        )
        frap_no_ctrl_cb.grid(row=row, column=0, columnspan=2, sticky="w")
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
        fn = self._ask_open_filename(
            title="Select FRAP CZI file",
            filetypes=[("CZI files", "*.czi"), ("All files", "*.*")]
        )
        if fn:
            self.frap_czi.set(fn)


    def browse_frap_folder(self):
        folder = self._ask_directory(title="Select FRAP batch folder")
        if folder:
            self.frap_folder.set(folder)

    def _on_frap_no_control_toggle(self):
        """Disable control ROI index field when no-control mode is selected."""
        if self.frap_no_control.get():
            self.frap_ctrl_idx_entry.configure(state="disabled")
            self.frap_ctrl_idx.set("")        # clear any stale value
        else:
            self.frap_ctrl_idx_entry.configure(state="normal")
    

    def create_fitting_tab(self):
        """Create the fitting tab using rics_fit module"""
        from theatrics.workers.fit_worker import fit_rics_process_main
        self._fit_rics_process_main = fit_rics_process_main
        from theatrics.workers.diffmap_worker import diffusion_map_process_main
        self._diffusion_map_process_main = diffusion_map_process_main
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
                                   values=[0,4], width=12)
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
                                   values=[0,4], width=12)
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
        filename = self._ask_saveas_filename(
            title="Select output file",
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)

    def browse_input_file(self):
        """Browse for input file"""
        filename = self._ask_open_filename(
            title="Select input image stack",
            filetypes=[
                ("All supported files", "*.czi *.tif *.tiff *.ptu"),
                ("CZI files",           "*.czi"),
                ("TIFF files",          "*.tif *.tiff"),
                ("PTU files",           "*.ptu"),
                ("All files",           "*.*"),
            ]
        )
        if filename:
            self.input_file.set(filename)

    def browse_sfcs_input_file(self):
        """Browse for SFCS input file"""
        filename = self._ask_open_filename(
            title="Select input line-scan file",
            filetypes=[
                ("All supported files", "*.czi *.ptu"),
                ("CZI files",           "*.czi"),
                ("PTU files",           "*.ptu"),
                ("All files",           "*.*"),
            ]
        )
        if filename:
            self.sfcs_input_file.set(filename)

            # for PTU files: log metadata so user can verify
            if filename.lower().endswith(".ptu"):
                self._log_ptu_linescan_metadata(filename)
    def browse_save_path(self):
        filename = self._ask_saveas_filename(
            title="Select output file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.saving_path.set(filename)
    def browse_batch_input_folder(self):
        filepath = self._ask_directory(
            title="Select directory for batch input",
        )
        if filepath:
            self.batch_input_folder.set(filepath)

    def browse_batch_fit_folder(self):
        filepath = self._ask_directory(
            title="Select directory for batch input",
        )
        if filepath:
            self.batch_fit_folder.set(filepath)

    def browse_input_file_diff_map(self):
        """Browse for input file"""
        filename = self._ask_open_filename(
            title="Select input image stack",
            filetypes=[
                ("All supported files", "*.czi *.tif *.tiff *.ptu"),
                ("CZI files",           "*.czi"),
                ("TIFF files",          "*.tif *.tiff"),
                ("PTU files",           "*.ptu"),
                ("All files",           "*.*"),
            ]
        )
        if filename:
            self.input_file_diff_map.set(filename)
    def browse_rics_file(self):
        """Browse for RICS map file"""
        filename = self._ask_open_filename(
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
            self._showwarning("Warning", "Simulation is already running.")
            return
        if not self.output_path.get():
            self._showwarning("Warning", "Please set an output path.")
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
            target=self._simulate_rics_process_main,
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
        filename = self._ask_open_filename(
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
    
    def _log_ptu_linescan_metadata(self, filepath: str):
        """
        Read and log PTU line-scan metadata so the user can verify
        timing parameters before running SFCS.
        """
        try:
            from theatrics.modules.SFCS_module import (
                read_ptu_linescan_metadata,
                TTTRLIB_AVAILABLE,
            )
            if not TTTRLIB_AVAILABLE:
                self.log_message(
                    "WARNING: tttrlib not installed — "
                    "PTU files cannot be read. "
                    "Install with:  pip install tttrlib"
                )
                return

            meta = read_ptu_linescan_metadata(filepath)

            self.log_message("PTU line-scan metadata:")
            self.log_message(
                f"  Pixel dwell time : {meta['pixel_dwell_time_us']:.4f} µs"
            )
            self.log_message(
                f"  Line time        : {meta['line_time_ms']:.4f} ms"
            )
            self.log_message(
                f"  Lines            : {meta['n_lines']}"
            )
            self.log_message(
                f"  Pixels per line  : {meta['n_pixels']}"
            )
            self.log_message(
                f"  Total duration   : {meta['total_duration_s']:.2f} s"
            )
            self.log_message(
                f"  Photon channels  : {meta['photon_channels']}"
            )
            self.log_message(
                f"  Line-start marker channel : {meta['line_start_channel']}"
            )

        except Exception as e:
            self.log_message(
                f"WARNING: could not read PTU line-scan metadata: {e}"
            )

    def run_SFCS(self):
        if self._is_worker_running("sfcs_proc"):
            self._showwarning("Warning", "SFCS is already running.")
            return

        if not self.sfcs_input_file.get():
            self._showwarning("Warning", "Please select an input file first")
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
            target=self._sfcs_process_main_curvefit,
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

    def _read_ptu_timing_to_gui(self, filepath: str):
        """
        Read pixel size, dwell time, and line time from a PTU file
        and populate the corresponding fitting parameter fields.
        Called automatically after a PTU file is selected for RICS export.
        """
        try:
            from theatrics.modules.export_rics import (
                read_ptu_metadata,
                TTTRLIB_AVAILABLE,
            )
            if not TTTRLIB_AVAILABLE:
                self.log_message(
                    "WARNING: tttrlib not installed — "
                    "PTU metadata cannot be read. "
                    "Install with:  pip install tttrlib"
                )
                return

            meta = read_ptu_metadata(filepath)

            if meta["pixel_size_nm"] is not None:
                self.fit_pixel_size.set(
                    f"{meta['pixel_size_nm']:.4f}"
                )
                self.log_message(
                    f"PTU pixel size: {meta['pixel_size_nm']:.4f} nm"
                )
            else:
                self.log_message(
                    "PTU metadata: pixel size not found in header"
                )

            if meta["pixel_dwell_time_us"] is not None:
                self.fit_pixel_dwell.set(
                    f"{meta['pixel_dwell_time_us']:.4f}"
                )
                self.log_message(
                    f"PTU pixel dwell time: "
                    f"{meta['pixel_dwell_time_us']:.4f} µs"
                )
            else:
                self.log_message(
                    "PTU metadata: pixel dwell time not found in header"
                )

            if meta["line_time_ms"] is not None:
                self.fit_line_time.set(
                    f"{meta['line_time_ms']:.4f}"
                )
                self.log_message(
                    f"PTU line time: {meta['line_time_ms']:.4f} ms"
                )
            else:
                self.log_message(
                    "PTU metadata: line time not found in header"
                )

            ch = meta.get("channels", [])
            self.log_message(
                f"PTU available channels: {ch}"
            )
            self.log_message(
                f"PTU frames: {meta['n_frames']}  "
                f"lines: {meta['n_lines']}  "
                f"pixels: {meta['n_pixels']}"
            )

        except Exception as e:
            self.log_message(
                f"WARNING: could not read PTU metadata: {e}"
            )

    def export_rics(self):
        if self._is_worker_running("export_proc"):
            self._showwarning("Warning", "RICS export is already running.")
            return
        """Export RICS map using export worker (process-based)."""
        if not self.input_file.get() and not self.batch_input_folder.get():
            self._showwarning("Warning", "Please select an input file or a folder for batch processing")
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
            target=self._export_rics_process_main,
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
        # collect both CZI and PTU files
        czi_files = get_files_from_folder(
            self.batch_input_folder.get(), ".czi", ""
        )
        ptu_files = get_files_from_folder(
            self.batch_input_folder.get(), ".ptu", ""
        )
        files = czi_files + ptu_files

        if not files:
            self._showwarning(
                "Warning",
                "No .czi or .ptu files found in the selected folder."
            )
            return

        self._batch_export_files = files
        self._batch_export_index = 0

        self.log_message(
            f"Starting batch RICS export for {len(files)} files "
            f"({len(czi_files)} CZI, {len(ptu_files)} PTU)..."
        )
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
            target=self._export_rics_process_main,
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
        filename = self._ask_open_filename(
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
            self._showwarning("Warning", "RICS fitting is already running.")
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
            target=self._fit_rics_process_main,
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
                    self._showerror("Fitting Error", "Fitting failed. See log.")
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
# -------------------------------------------------FCS Export GUI-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def _run_ptu_fcs(self):
        if self._is_worker_running("ptu_fcs_proc"):
            self._showwarning(
                "Warning", "PTU correlation is already running."
            )
            return

        single = self.ptu_fcs_file.get().strip()
        folder = self.ptu_fcs_folder.get().strip()

        if not single and not folder:
            self._showwarning(
                "Warning", "Select a single PTU/RAW file or a batch folder."
            )
            return

        is_raw_single = single.lower().endswith(".raw") if single else False

        # PIE / FLCS are forced off for .raw input regardless of checkbox
        # state (defence in depth -- the checkboxes are also disabled by
        # _on_ptu_input_type_detected(), but this guards against stale state)
        use_pie      = bool(self.ptu_fcs_use_pie.get())  and not is_raw_single
        use_flcs_bg  = bool(self.ptu_fcs_use_flcs.get()) and not is_raw_single
        use_burst_removal = bool(self.ptu_fcs_use_burst_removal.get()) and not is_raw_single

        ww_str = self.ptu_fcs_wohland_window.get().strip()
        wohland_window = float(ww_str) if ww_str else None

        params = dict(
        use_pie           = use_pie,
        tau_min_s         = self._safe_float(
                                self.ptu_fcs_tau_min,  "Tau min",    1e-6),
        tau_max_s         = self._safe_float(
                                self.ptu_fcs_tau_max,  "Tau max",    1.0),
        n_bins            = self._safe_int(
                                self.ptu_fcs_nbins,    "n_bins",     9),
        use_afterpulsing  = bool(self.ptu_fcs_use_ap.get()),
        ap_params_path    = self.ptu_fcs_ap_path.get().strip(),
        use_flcs_bg       = use_flcs_bg,
        wohland_window_s  = wohland_window,
        n_bootstrap       = self._safe_int(
                                self.ptu_fcs_n_bootstrap, "Bootstrap", 20),
        )

        # NEW: DD/AA/DA selection only meaningful for PTU files -- the .raw
        # pipeline (run_fcs_export_raw) always uses the explicit Channel 1 /
        # Channel 2 pairing from _ptu_raw_frame instead, so these keys are
        # simply omitted when the .raw frame is the active one.
        is_raw_mode = bool(self._ptu_raw_frame.winfo_ismapped())
        if not is_raw_mode:
            params["compute_dd"] = bool(self.ptu_fcs_compute_dd.get())
            params["compute_aa"] = bool(self.ptu_fcs_compute_aa.get())
            params["compute_da"] = bool(self.ptu_fcs_compute_da.get())

        if use_pie:
            params.update(
                donor_channel     = self._safe_int(
                                        self.ptu_pie_donor_ch,    "Donor ch",    0),
                acceptor_channel  = self._safe_int(
                                        self.ptu_pie_acceptor_ch, "Acceptor ch", 1),
                prompt_gate       = [
                    self._safe_float(self.ptu_pie_prompt_start, "Prompt start", 0.0),
                    self._safe_float(self.ptu_pie_prompt_stop,  "Prompt stop",  0.5),
                ],
                delay_gate        = [
                    self._safe_float(self.ptu_pie_delay_start, "Delay start", 0.5),
                    self._safe_float(self.ptu_pie_delay_stop,  "Delay stop",  1.0),
                ],
                symmetric_cc      = bool(self.ptu_pie_symmetric.get()),
                gamma             = self._safe_float(
                                        self.ptu_pie_gamma,       "Gamma",       1.0),
                crosstalk         = self._safe_float(
                                        self.ptu_pie_crosstalk,   "Crosstalk",   0.0),
                direct_excitation = self._safe_float(
                                        self.ptu_pie_direct_exc,  "Dir. exc.",   0.0),
            )
        else:
            ch = self._safe_int(self.ptu_fcs_channel, "Channel", 0)

            gs_str = self.ptu_fcs_gate_start.get().strip()
            ge_str = self.ptu_fcs_gate_stop.get().strip()

            if gs_str and ge_str:
                gate_start_ns = float(gs_str)
                gate_stop_ns  = float(ge_str)
                params["cs1"]           = None
                params["cs2"]           = None
                params["channel"]       = ch
                params["gate_start_ns"] = gate_start_ns
                params["gate_stop_ns"]  = gate_stop_ns
            else:
                params["cs1"]     = None
                params["cs2"]     = None
                params["channel"] = ch
        # ── NEW: always build Zeiss .raw-specific params too, regardless of
        # whether the current selection is actually a .raw file. These are
        # silently ignored by run_fcs_export() (the PTU pipeline) via its
        # **_ignored_kwargs catch-all, and are the ONLY params actually used
        # by run_fcs_export_raw(). This lets a single params dict safely
        # support single-file mode (of either type) and mixed-type batch
        # folders without needing to know the file type at build time.
        ch1_raw = self._parse_raw_channel(self.ptu_raw_ch1.get())
        ch2_raw = self._parse_raw_channel(self.ptu_raw_ch2.get())
        params["channels_pairs"]    = [(ch1_raw, ch2_raw)]
        params["n_segments"]       = self._safe_int(
                                        self.ptu_raw_n_segments, "Segments", 6)
        params["offset_s"]         = self._safe_float(
                                        self.ptu_raw_offset_s, "Offset (s)", 0.0)
        params["correct_bleaching"] = bool(self.ptu_fcs_correct_bleaching.get())
        params["use_burst_removal"] = bool(self.ptu_fcs_use_burst_removal.get())
        params["burst_threshold_alpha"] = self._safe_float(self.ptu_fcs_burst_threshold_alpha, "Burst threshold α", 0.02)
        if single:
            params["mode"]     = "single"
            params["filepath"] = single
        else:
            import glob
            ptu_paths = glob.glob(os.path.join(folder, "**", "*.ptu"), recursive=True)
            raw_paths = glob.glob(os.path.join(folder, "**", "*.raw"), recursive=True)
            filepaths = sorted(ptu_paths + raw_paths)

            if not filepaths:
                self._showwarning(
                    "Warning", "No PTU or RAW files found in the selected folder."
                )
                return

            params["mode"]      = "batch"
            params["filepaths"] = filepaths
            params["cpu_n"]     = clamp_workers(
                self.ptu_fcs_n_cores.get(), max_fraction=0.8, hard_cap=64
            )

        mode_label = "PIE" if use_pie else "standard"
        self.log_message(
            f"Starting FCS export ({mode_label}, "
            f"{'single' if single else 'batch'}) ..."
        )
        if not single:
            self.log_message(
                f"Batch: {len(params['filepaths'])} file(s), "
                f"using {params['cpu_n']} parallel worker(s)."
            )

        self.status_var.set("Running FCS export...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        self.ptu_fcs_queue        = multiprocessing.Queue()
        self.ptu_fcs_cancel_event = multiprocessing.Event()

        self.ptu_fcs_proc = multiprocessing.Process(
            target=self._ptu_correlate_worker_main,
            args=(params, self.ptu_fcs_queue, self.ptu_fcs_cancel_event),
            daemon=False,
        )
        self.ptu_fcs_proc.start()
        self._poll_ptu_fcs_queue()


# ── queue polling ─────────────────────────────────────────────────────────────

    def _poll_ptu_fcs_queue(self):
        try:
            while True:
                msg_type, payload = self.ptu_fcs_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    self.progress_var.set(float(payload))

                elif msg_type == "file_done":
                    self._update_ptu_fcs_display(payload)
                    n_ok = len(payload.get("results", []))
                    self.log_message(
                        f"Done: {n_ok} curve(s) from "
                        f"{os.path.basename(payload.get('results', [{}])[0].get('csv_path', ''))}"
                    )

                elif msg_type == "file_error":
                    self.log_message(payload)

                elif msg_type == "done":
                    # single-file result dict has key "results"
                    if isinstance(payload, dict) and "results" in payload:
                        self._update_ptu_fcs_display(payload)
                        self._log_ptu_result(payload)
                    else:
                        # batch summary
                        self.log_message(
                            f"PTU batch done — "
                            f"{payload.get('n_ok')}/{payload.get('n_total')} succeeded"
                        )
                        if payload.get("last_res"):
                            self._update_ptu_fcs_display(payload["last_res"])
                            self._log_ptu_result(payload["last_res"])

                    self.set_ui_busy(False)
                    self.status_var.set("Ready")
                    self.progress_bar.grid_remove()
                    return

                elif msg_type == "cancelled":
                    self.log_message("PTU FCS cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return

                elif msg_type == "error":
                    self.log_message(payload)
                    self.status_var.set("Error")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    self._showerror(
                        "PTU FCS Error", "Correlation failed. See log."
                    )
                    return

        except queue.Empty:
            pass

        if (
            self.ptu_fcs_proc is not None
            and not self.ptu_fcs_proc.is_alive()
        ):
            self.set_ui_busy(False)
            self.status_var.set("Error")
            self.progress_bar.grid_remove()
            self.log_message("PTU FCS worker terminated unexpectedly.")
            return

        self.root.after(50, self._poll_ptu_fcs_queue)


    def _log_ptu_result(self, res: dict):
        """Write a summary of a completed PTU/RAW export to the log."""
        out_dir = res.get("out_dir")
        if out_dir:
            self.log_message(f"All output written to: {out_dir}")
        for w in res.get("warnings", []):
            self.log_message(f"  WARNING: {w}")

        for r in res.get("results", []):
            # NEW: handle failed channel pairs (csv_path is None, "error" present)
            if r.get("csv_path") is None:
                self.log_message(
                    f"  [{r.get('label', '?')}]  FAILED — see traceback below:"
                )
                err = r.get("error")
                if err:
                    self.log_message(err)
                else:
                    self.log_message("  (no error traceback captured)")
                continue

            self.log_message(
                f"  [{r['label']}]  "
                f"ACR={r['acr1_Hz']:.0f} Hz  "
                f"FLCS={r['flcs_used']}  "
                f"AP={r['ap_used']}  "
                f"→ {os.path.basename(r['csv_path'])}"
            )

        fret = res.get("fret")
        if fret is not None:
            self.log_message(
                f"  FRET: E={fret.get('FRET_efficiency', float('nan')):.3f}  "
                f"S={fret.get('stoichiometry', float('nan')):.3f}  "
                f"PR={fret.get('proximity_ratio', float('nan')):.3f}"
            )
            self.log_message(
                f"  Photon counts:  "
                f"F_DD={fret.get('F_DD', 0):.0f}  "
                f"F_DA={fret.get('F_DA', 0):.0f}  "
                f"F_AA={fret.get('F_AA', 0):.0f}"
            )
        burst_info = res.get("burst_removal_info", {})
        for name, info in burst_info.items():
            if info.get("threshold_counts") is None:
                continue
            n_burst = info.get("n_burst_bins", 0)
            n_total = info.get("n_total_bins", 0)
            pct = 100.0 * n_burst / max(1, n_total)
            self.log_message(
                f"  Burst removal [{name}]: threshold={info['threshold_counts']} counts/bin, "
                f"{n_burst}/{n_total} bins flagged ({pct:.1f}%)"
            )


# ── display ───────────────────────────────────────────────────────────────────

    def _update_ptu_fcs_display(self, res: dict):
        """
        Draw a unified 2x2 overview:
          top-left     : all G(tau) curves overlaid
          top-right    : intensity traces of all involved channels overlaid
          bottom-left  : FCS_Fixer-style 2-row FLCS diagnostic
                         (top sub-panel: normalized patterns, log scale;
                          bottom sub-panel: filter functions, linear scale)
          bottom-right : FRET metrics summary (PIE mode) or placeholder
        """
        results          = res.get("results", [])
        fret             = res.get("fret", None)
        intensity_traces = res.get("intensity_traces", {})

        if not results:
            return

        self.ptu_fcs_fig.clear()
        outer_gs = gridspec.GridSpec(
            2, 2, figure=self.ptu_fcs_fig, hspace=0.5, wspace=0.32
        )
        ax_corr  = self.ptu_fcs_fig.add_subplot(outer_gs[0, 0])
        ax_trace = self.ptu_fcs_fig.add_subplot(outer_gs[0, 1])
        ax_fret  = self.ptu_fcs_fig.add_subplot(outer_gs[1, 1])

        corr_color_map = {
            "donor_ACF":    "steelblue",
            "acceptor_ACF": "tomato",
            "PIE_CCF":      "seagreen",
            "ACF":          "steelblue",
            "CCF":          "mediumpurple",
            "DD_ACF":       "steelblue",   # NEW
            "AA_ACF":       "tomato",      # NEW
            "DA_CCF":       "seagreen",    # NEW
        }
        trace_color_map = {"donor": "steelblue", "acceptor": "tomato"}
        trace_corrected_color_map = {   # NEW: distinct color for the
                                         # bleach-corrected curve, kept
                                         # in the same hue family as the
                                         # raw trace for the same channel
            "donor":    "navy",
            "acceptor": "darkred",
        }
        from itertools import cycle as _cycle
        fallback_colors = _cycle(
            ["steelblue", "tomato", "seagreen", "mediumpurple", "goldenrod", "slategray"]
        )

        # ── Panel 1 (top-left): all G(tau) curves overlaid ─────────────────
        any_corr_plotted = False   # NEW
        for r in results:
            if r.get("csv_path") is None:   # NEW: skip failed pairs entirely
                continue
            lbl   = r["label"]
            color = corr_color_map.get(lbl, next(fallback_colors))
            lag_s = np.asarray(r["lag_s"], dtype=float)
            G     = np.asarray(r["G"],     dtype=float)
            sg    = np.asarray(r["sigma_G"], dtype=float)
            if len(lag_s) == 0:
                continue
            ax_corr.semilogx(lag_s, G, "-", color=color, linewidth=2, label=lbl)
            ax_corr.fill_between(lag_s, G - sg, G + sg, alpha=0.15, color=color)
            any_corr_plotted = True   # NEW

        ax_corr.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax_corr.set_xlabel("τ (s)")
        ax_corr.set_ylabel("G(τ)")
        corr_flags = []
        if any(r.get("ap_used")   for r in results if r.get("csv_path")): corr_flags.append("AP corr.")   # NEW guard
        if any(r.get("flcs_used") for r in results if r.get("csv_path")): corr_flags.append("FLCS")        # NEW guard
        title = "Correlation curves"
        if corr_flags:
            title += "  (" + ", ".join(corr_flags) + ")"
        ax_corr.set_title(title, fontsize=10)
        if any_corr_plotted:                     # CHANGED: only call legend if something was plotted
            ax_corr.legend(fontsize=8)
        else:
            ax_corr.text(
                0.5, 0.5, "All correlations failed\n(see Results & Logs tab)",
                transform=ax_corr.transAxes, ha="center", va="center",
                fontsize=10, color="red",
            )
        ax_corr.grid(True, alpha=0.3)
        ax_corr.spines[["top", "right"]].set_visible(False)

        # ── Panel 2 (top-right): intensity traces overlaid ─────────────────
        any_bleach_corrected = False    #tracks whether to add to the title
        any_bursts_shaded = False
        for name, tr in intensity_traces.items():
            t_s = np.asarray(tr.get("t_s", []), dtype=float)
            cps = np.asarray(tr.get("cps", []), dtype=float)
            if len(t_s) == 0:
                continue
            color = trace_color_map.get(name, next(fallback_colors))
            ch    = tr.get("channel", "?")
            ax_trace.plot(t_s, cps / 1000.0, "-", color=color, linewidth=1.1,
                          alpha=0.6, label=f"{name} (ch{ch})")

            # NEW: overlay the bleach-corrected trace, if present, in a
            # distinct (but related) color
            cps_corr = tr.get("cps_corrected")
            if cps_corr is not None:
                t_s_corr = np.asarray(tr.get("t_s_corrected", []), dtype=float)
                cps_corr = np.asarray(cps_corr, dtype=float)
                if len(t_s_corr) > 0:
                    corr_color = trace_corrected_color_map.get(
                        name, next(fallback_colors)
                    )
                    ax_trace.plot(
                        t_s_corr, cps_corr / 1000.0, "-",
                        color=corr_color, linewidth=1.4, alpha=0.9,
                        label=f"{name} (ch{ch}) — bleach corr."
                    )
                    any_bleach_corrected = True
            burst_intervals = tr.get("burst_intervals_s", [])
            for i_b, (b_start, b_stop) in enumerate(burst_intervals):
                ax_trace.axvspan(
                    b_start, b_stop,
                    color="red", alpha=0.18, linewidth=0,
                    label=(f"{name} burst removed" if i_b == 0 and not any_bursts_shaded else None),
                )
                any_bursts_shaded = True

        ax_trace.set_xlabel("Time (s)")
        ax_trace.set_ylabel("Count rate (kHz)")
        title = "Intensity traces"
        if any_bleach_corrected:
            title += "  (raw + bleach-corrected)"
        if any_bursts_shaded:
            title += "  [red = burst removed]"
        ax_trace.set_title(title, fontsize=10)
        ax_trace.legend(fontsize=7)
        ax_trace.grid(True, alpha=0.3)
        ax_trace.spines[["top", "right"]].set_visible(False)
        

        # ── Panel 3 (bottom-left): FCS_Fixer-style 2-row FLCS diagnostic ────
        # Nested gridspec: top = normalized patterns (log), bottom = filter
        # functions (linear). NO twin axis -> avoids the tight_layout warning.
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[1, 0], height_ratios=[1, 1], hspace=0.12
        )
        ax_pat  = self.ptu_fcs_fig.add_subplot(inner_gs[0])
        ax_filt = self.ptu_fcs_fig.add_subplot(inner_gs[1], sharex=ax_pat)

        tcspc_color_map = {
            "donor_ACF":    "steelblue",
            "acceptor_ACF": "tomato",
            "PIE_CCF":      "seagreen",
            "ACF":          "steelblue",
            "CCF":          "mediumpurple",
            "DD_ACF":       "steelblue",   # NEW
            "AA_ACF":       "tomato",      # NEW
            "DA_CCF":       "seagreen",    # NEW
        }
        any_tcspc = False

        for r in results:
            if r.get("csv_path") is None:   # NEW: skip failed pairs
                continue
            for key, suffix, alpha_mul in [("tcspc_csv", "", 1.0), ("tcspc_csv_cs2", " (cs2)", 0.6)]:
                path = r.get(key)
                if not path or not os.path.isfile(path):
                    continue
                if "pattern_signal" not in pd.read_csv(path, nrows=0).columns:
                    continue  # older CSV without the new columns -- skip gracefully

                any_tcspc = True
                df    = pd.read_csv(path)
                color = tcspc_color_map.get(r["label"], next(fallback_colors))
                lbl   = f"{r['label']}{suffix}"

                # top: normalized patterns (log scale, scatter)
                sig_mask = df["pattern_signal"] > 0
                bg_mask  = df["pattern_background"] > 0
                ax_pat.semilogy(
                    df["time_ns"][sig_mask], df["pattern_signal"][sig_mask],
                    "o", color=color, markersize=2.5, alpha=0.9 * alpha_mul,
                    label=f"{lbl} signal"
                )
                ax_pat.semilogy(
                    df["time_ns"][bg_mask], df["pattern_background"][bg_mask],
                    "o", color="gray", markersize=2.5, alpha=0.6 * alpha_mul,
                    label=f"{lbl} bg" if suffix == "" else None
                )

                # bottom: filter functions (linear, line)
                ax_filt.plot(
                    df["time_ns"], df["filter_signal"],
                    "-", color=color, linewidth=1.1, alpha=0.9 * alpha_mul
                )
                ax_filt.plot(
                    df["time_ns"], df["filter_background"],
                    "--", color="gray", linewidth=0.9, alpha=0.6 * alpha_mul
                )

        ax_pat.set_title(
            "FLCS filter functions" if any_tcspc else "TCSPC (FLCS not used)",
            fontsize=10
        )
        ax_pat.tick_params(axis="x", labelbottom=False)
        if any_tcspc:
            ax_pat.legend(fontsize=6, loc="upper right", ncol=1)
        ax_pat.grid(True, alpha=0.3)

        ax_filt.axhline(0, color="lightgray", linewidth=0.6, zorder=0)
        # ax_filt.set_title("FLCS filter functions", fontsize=10)
        ax_filt.set_xlabel("Time (ns)")
        ax_filt.set_ylabel("Filter weight")
        ax_filt.grid(True, alpha=0.3)

        # ── Panel 4 (bottom-right): FRET metrics or placeholder ────────────
        ax_fret.axis("off")
        if fret is not None:
            lines = [
                "PIE-FRET summary",
                "─" * 26,
                f"E (FRET eff.)     = {fret.get('FRET_efficiency', float('nan')):.3f}",
                f"S (stoichiometry) = {fret.get('stoichiometry', float('nan')):.3f}",
                f"PR (proximity)    = {fret.get('proximity_ratio', float('nan')):.3f}",
                "",
                f"F_DD = {fret.get('F_DD', 0):.0f} photons",
                f"F_DA = {fret.get('F_DA', 0):.0f} photons",
                f"F_AA = {fret.get('F_AA', 0):.0f} photons",
                f"F_DA_corr = {fret.get('F_DA_corrected', 0):.0f}",
                "",
                f"Donor ACR    = {fret.get('donor_acr_Hz', 0):.0f} Hz",
                f"Acceptor ACR = {fret.get('acceptor_acr_Hz', 0):.0f} Hz",
            ]
            ax_fret.text(
                0.05, 0.95, "\n".join(lines),
                transform=ax_fret.transAxes, va="top", ha="left",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="goldenrod", alpha=0.9),
            )
            ax_fret.set_title("FRET metrics", fontsize=10)
        else:
            ax_fret.text(
                0.5, 0.5, "No PIE / FRET data\n(enable PIE mode to compute)",
                transform=ax_fret.transAxes, ha="center", va="center",
                fontsize=10, color="gray",
            )
            ax_fret.set_title("FRET metrics", fontsize=10)

        # CHANGED: replaced tight_layout() with explicit subplots_adjust().
        # tight_layout() previously triggered a UserWarning because of the
        # twinx() axis that used to live in the TCSPC panel. That axis has
        # been removed entirely (replaced by the 2-row nested gridspec
        # above), so the warning's root cause is gone -- but subplots_adjust
        # is kept anyway as a robust, warning-free alternative that also
        # gives predictable spacing for the nested gridspec.
        self.ptu_fcs_fig.subplots_adjust(
            left=0.08, right=0.96, top=0.93, bottom=0.08,
            hspace=0.5, wspace=0.32
        )
        self.ptu_fcs_canvas.draw()

        # ── save overview SVG next to first output CSV ──────────────────────
        valid_results = [r for r in results if r.get("csv_path")]   # NEW
        if valid_results:                                            # CHANGED
            first_csv = valid_results[0]["csv_path"]
            out_dir  = os.path.dirname(first_csv)
            svg_path = os.path.join(out_dir, "overview.svg")
            self.ptu_fcs_fig.savefig(
                svg_path, dpi=300, bbox_inches="tight", facecolor="white"
            )
            self.log_message(f"Output folder: {out_dir}")
            self.log_message(f"Saved overview SVG: {svg_path}")
        else:
            self.log_message("No successful correlations to save an overview for.")


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

            "tau_D limits":                  [-7, -1],
            "number of diffusion components": 200,
            "number of iterations":           20000,   # increased from 10000
            "chi2 target":                    1.0,
            "stop criterion":                 5e-6,
            "stop window":                    100,
            "check every":                    200,
            "viscosity_mPas":                 0.835,
            
            "prior width decades": 0.5,
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
            "g3diffMEMFCS": [
                "tau_D limits",
                "number of diffusion components",
                "number of iterations",
                "chi2 target",
                "stop criterion",
                "stop window",
                "check every",
                "prior width decades",
                "viscosity_mPas",
                # "temperature_K"  ← removed, taken from Experiment T field
            ],
        }
    def _safe_float_from_str(self, value, name: str, fallback: float) -> float:
        """Parse a raw value (str, int, or float) as float with fallback."""
        try:
            v = float(value)
            if not np.isfinite(v):
                raise ValueError
            return v
        except (ValueError, TypeError):
            self.log_message(f"WARNING: invalid value for '{name}', using {fallback}")
            return fallback
    def _log_memfcs_interpretation(self, res: dict):
        """
        Log the MEMFCS interpretation guide to the Results & Logs tab.
        """
        chi2_sc  = res.get("memfcs_chi2_sc",    float("nan"))
        chi2_sh  = float(res.get("return_dict",
                                  {}).get("Chi squared", float("nan")))
        chi2_jy  = res.get("memfcs_chi2_jy",    float("nan"))
        pk_D_sh  = res.get("memfcs_max_freq_D")
        pk_D_jy  = res.get("memfcs_max_freq_D_jy")
        mn_D_sh  = res.get("return_dict", {}).get("D")
        mn_D_jy  = res.get("return_dict", {}).get("mean_D_jy")
        pk_Rh_sh = res.get("memfcs_max_freq_R_h")
        pk_Rh_jy = res.get("memfcs_max_freq_R_h_jy")
        conv_sh  = res.get("return_dict", {}).get("converged",    False)
        conv_jy  = res.get("return_dict", {}).get("converged_jy", False)

        self.log_message("─" * 60)
        self.log_message("MEMFCS RESULTS SUMMARY")
        self.log_message("─" * 60)
        self.log_message(
            f"Single-component fit chi2 = {chi2_sc:.3f}"
        )
        self.log_message(
            f"Shannon MEMFCS:  chi2={chi2_sh:.4f}  "
            f"peak D={pk_D_sh:.1f} µm²/s  "
            f"mean D={mn_D_sh:.1f} µm²/s  "
            f"peak R_h={pk_Rh_sh:.2f} nm  "
            f"converged={conv_sh}"
        )
        self.log_message(
            f"Jaynes MEMFCS:   chi2={chi2_jy:.4f}  "
            f"peak D={pk_D_jy:.1f} µm²/s  "
            f"mean D={mn_D_jy:.1f} µm²/s  "
            f"peak R_h={pk_Rh_jy:.2f} nm  "
            f"converged={conv_jy}"
        )
        self.log_message("")
        self.log_message("INTERPRETATION GUIDE")
        self.log_message("─" * 60)

        if not np.isnan(chi2_sc):
            if chi2_sc < 2.0:
                self.log_message(
                    f"✓ Single-component chi2={chi2_sc:.2f} < 2  →  "
                    f"data is CONSISTENT with a single species."
                )
                self.log_message(
                    "  Both Shannon and Jaynes should give a single peak "
                    "at the same D."
                )
                self.log_message(
                    "  Jaynes will give a SHARPER peak (prior reinforces "
                    "single-species interpretation)."
                )
                self.log_message(
                    "  Agreement between the two methods is strong "
                    "evidence for sample homogeneity."
                )
            elif chi2_sc < 10.0:
                self.log_message(
                    f"⚠ Single-component chi2={chi2_sc:.2f} (2–10)  →  "
                    f"MILD heterogeneity suggested."
                )
                self.log_message(
                    "  Shannon MEMFCS may reveal a secondary population."
                )
                self.log_message(
                    "  Jaynes MEMFCS will only deviate from the "
                    "single-species prior as much as the data demands."
                )
                self.log_message(
                    "  Compare peak positions — if they agree, the "
                    "secondary peak in Shannon may be a noise artefact."
                )
            else:
                self.log_message(
                    f"✗ Single-component chi2={chi2_sc:.2f} >> 2  →  "
                    f"data REQUIRES more than one species."
                )
                self.log_message(
                    "  Shannon MEMFCS will reveal the heterogeneous "
                    "distribution without bias."
                )
                self.log_message(
                    "  Jaynes MEMFCS uses the fastest species as its "
                    "prior — secondary peaks are suppressed unless "
                    "strongly required."
                )
                self.log_message(
                    "  Prefer Shannon for unbiased characterisation of "
                    "heterogeneous samples."
                )

        self.log_message("")
        if (pk_D_sh is not None and pk_D_jy is not None
                and not np.isnan(pk_D_sh) and not np.isnan(pk_D_jy)):
            ratio = max(pk_D_sh, pk_D_jy) / max(
                min(pk_D_sh, pk_D_jy), 1e-10
            )
            if ratio < 1.5:
                self.log_message(
                    f"✓ Shannon peak D = {pk_D_sh:.1f} µm²/s  "
                    f"Jaynes peak D = {pk_D_jy:.1f} µm²/s  "
                    f"(ratio = {ratio:.2f} < 1.5)"
                )
                self.log_message(
                    "  Both methods agree on peak D  →  "
                    "ROBUST result independent of prior choice."
                )
            else:
                self.log_message(
                    f"⚠ Shannon peak D = {pk_D_sh:.1f} µm²/s  "
                    f"Jaynes peak D = {pk_D_jy:.1f} µm²/s  "
                    f"(ratio = {ratio:.2f} > 1.5)"
                )
                self.log_message(
                    "  Methods disagree  →  result depends on prior "
                    "choice. Consider varying 'prior width decades'."
                )

        self.log_message("")
        self.log_message(
            "Prior width guide:  "
            "0.25d = very conservative (tight single-species bias)  |  "
            "0.5d = moderate  |  "
            "1.0d ≈ flat Shannon prior"
        )
        self.log_message("─" * 60)
    def run_fcsfit(self):
        if self._is_worker_running("fcsfit_proc"):
            self._showwarning("Warning", "FCS fitting is already running.")
            return
        csv_path = self.fcsfit_csv.get().strip()
        folder = self.fcsfit_folder.get().strip()

        if not csv_path and not folder:
            self._showwarning("Warning", "Select a single CSV or a batch folder.")
            return

        mode = "single" if csv_path else "batch"

        tau_min = self._safe_float(self.fcsfit_tau_min, "Tau min", 1e-6)
        tau_max = self._safe_float(self.fcsfit_tau_max, "Tau max", 1.0)
        # experiment_T is in °C; MEMFCS needs Kelvin and viscosity in Pa·s
        T_C = self._safe_float(self.fcsfit_expt_T, "Experiment T", 28.0)

        initial_params = dict(self.fcs_default_initial_params())
        # update temperature_K in params so the editor shows the correct value
        initial_params['temperature_K'] = T_C + 273.15

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
        # convert viscosity from mPa·s (GUI) → Pa·s (SI, needed by calculations)
        eta_mPas = self._safe_float_from_str(
            initial_params.get('viscosity_mPas', 0.8324),
            'viscosity_mPas', 0.8324
        )
        initial_params['viscosity_Pa_s'] = eta_mPas * 1e-3
        model = self.fcsfit_model.get()

        kwargs = dict(
            fitting_model=model,
            tau_domain=(tau_min, tau_max),
            user_tau_domain=True,
            psf_radius_um=self._safe_float(self.fcsfit_psf_radius, "PSF radius", 0.25),
            psf_aspect_ratio=self._safe_float(self.fcsfit_psf_ar, "PSF aspect ratio", 5.0),
            experiment_T=self._safe_float(self.fcsfit_expt_T, "Experiment T", 28.0),
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
            target=self._fcsfit_process_main,
            args=(params, self.fcsfit_queue, self.fcsfit_cancel_event),
            daemon=False
        )
        self.fcsfit_proc.start()
        self._poll_fcsfit_queue()
    def update_fcs_fit_display(self, res: dict, write_summary: bool = True):
        self.fcsfit_fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fcsfit_fig)

        fitting_model = res["fitting_model"]
        base_path     = res["base_path"]

        tau   = np.asarray(res["tau"],          dtype=float)
        G     = np.asarray(res["G"],            dtype=float)
        sigma = np.asarray(res["sigma_G"],      dtype=float)
        pred  = np.asarray(res["ccPrediction"], dtype=float)
        wr    = np.asarray(res["weighted_r"],   dtype=float)

        is_memfcs = (fitting_model == "g3diffMEMFCS")

        # ── top-left: correlation curve ───────────────────────────
        ax00 = self.fcsfit_fig.add_subplot(gs[0, 0])
        ax00.semilogx(tau, G,    "r",  label="G observed")
        ax00.semilogx(tau, pred, "g",
                      label="Shannon fit" if is_memfcs else "G fit")
        ax00.fill_between(tau, G - sigma, G + sigma,
                          color="b", alpha=0.2, label="±σ")

        if is_memfcs:
            G_fit_jy = res.get("memfcs_G_fit_jy")
            if G_fit_jy is not None:
                ax00.semilogx(tau,
                              np.asarray(G_fit_jy, dtype=float),
                              "b--", linewidth=1.8, label="Jaynes fit")
            rd = res.get("return_dict", {})
            G_pred_sc = rd.get("G_pred_sc")
            if G_pred_sc is not None:
                ax00.semilogx(tau,
                              np.asarray(G_pred_sc, dtype=float),
                              "k:", linewidth=1.5,
                              label=(f"1-comp "
                                     f"chi2={rd.get('chi2_sc', 0):.2f}"))

        ax00.set_xlabel("τ (s)")
        ax00.set_ylabel("G(τ)")
        ax00.set_title("Correlation curve")
        ax00.legend(fontsize=8)
        ax00.grid(True, alpha=0.3)

        # ── top-right: weighted residuals ─────────────────────────
        ax01 = self.fcsfit_fig.add_subplot(gs[0, 1])
        ax01.semilogx(tau, wr, "g", linewidth=1,
                      label="Shannon" if is_memfcs else None)
        ax01.axhline( 0, color="k", lw=1,   alpha=0.5)
        ax01.axhline( 3, color="r", lw=0.8, alpha=0.5, linestyle="--")
        ax01.axhline(-3, color="r", lw=0.8, alpha=0.5, linestyle="--")

        if is_memfcs:
            rd    = res.get("return_dict", {})
            wr_jy = rd.get("weighted_r_jy")
            if wr_jy is not None:
                ax01.semilogx(tau,
                              np.asarray(wr_jy, dtype=float),
                              "b", linewidth=1, alpha=0.7, label="Jaynes")
            ax01.legend(fontsize=8)

        ax01.set_xlabel("τ (s)")
        ax01.set_ylabel("Weighted residual")
        ax01.set_title("Weighted residuals")
        ax01.grid(True, alpha=0.3)

        # ── bottom-left: D distribution (MEMFCS) / iMSD (others) ─
        ax10 = self.fcsfit_fig.add_subplot(gs[1, 0])

        if is_memfcs:
            memfcs_D       = res.get("memfcs_D")
            memfcs_amps    = res.get("memfcs_amplitudes")
            memfcs_D_jy    = res.get("memfcs_D_jy")
            memfcs_amps_jy = res.get("memfcs_amplitudes_jy")

            if memfcs_D is not None and memfcs_amps is not None:
                memfcs_D    = np.asarray(memfcs_D,    dtype=float)
                memfcs_amps = np.asarray(memfcs_amps, dtype=float)
                ax10.semilogx(memfcs_D, memfcs_amps,
                              color="seagreen", linewidth=2, label="Shannon")
                max_D = res.get("memfcs_max_freq_D")
                if max_D is not None:
                    ax10.axvline(float(max_D), color="tomato",
                                 linestyle="--", linewidth=1.5,
                                 label=f"Sh peak {float(max_D):.1f} µm²/s")

            if memfcs_D_jy is not None and memfcs_amps_jy is not None:
                memfcs_D_jy    = np.asarray(memfcs_D_jy,    dtype=float)
                memfcs_amps_jy = np.asarray(memfcs_amps_jy, dtype=float)
                ax10.semilogx(memfcs_D_jy, memfcs_amps_jy,
                              color="steelblue", linewidth=2,
                              linestyle="--", label="Jaynes")
                max_D_jy = res.get("memfcs_max_freq_D_jy")
                if max_D_jy is not None:
                    ax10.axvline(float(max_D_jy), color="navy",
                                 linestyle=":", linewidth=1.5,
                                 label=f"Jy peak {float(max_D_jy):.1f} µm²/s")

            D_fit_sc = res.get("memfcs_D_fit_sc")
            if D_fit_sc is not None and not np.isnan(float(D_fit_sc)):
                ax10.axvline(float(D_fit_sc), color="k",
                             linestyle=":", linewidth=1.5,
                             label=f"1-comp {float(D_fit_sc):.1f} µm²/s")

            ax10.set_xlabel("D (µm²/s)")
            ax10.set_ylabel("Amplitude")
            ax10.set_title("MEMFCS — D distribution\n"
                           "green=Shannon  blue=Jaynes  black=1-comp")
            ax10.legend(fontsize=7)
            ax10.grid(True, alpha=0.3)

        else:
            reIMSD = None
            if fitting_model not in ["siFCS", "siFCSTwoComponents",
                                     "g3diffMEMFCS"]:
                aR     = res.get("PSF_aspect_ratio")
                N      = res.get("N")
                offset = res.get("offset", 0.0)
                if aR is not None and N is not None:
                    
                    reIMSD = self._calculate.iMSD_calc(
                        tau, float(aR), float(N), pred, float(offset)
                    )
                    ax10.loglog(tau, reIMSD)
                    ax10.set_ylabel("iMSD")
                    ax10.set_title("iMSD")
                else:
                    ax10.semilogx(tau, G,    "r", label="G observed")
                    ax10.semilogx(tau, pred, "g", label="G fit")
                    ax10.set_ylabel("G(τ)")
                    ax10.set_title("Correlation (log-log)")
                    ax10.legend(fontsize=8)
            else:
                ax10.semilogx(tau, G,    "r", label="G observed")
                ax10.semilogx(tau, pred, "g", label="G fit")
                ax10.set_ylabel("G(τ)")
                ax10.set_title("Correlation")
                ax10.legend(fontsize=8)

            ax10.set_xlabel("τ (s)")
            ax10.grid(True, alpha=0.3)

        # ── bottom-right: R_h (MEMFCS) / residual histogram (others)
        ax11 = self.fcsfit_fig.add_subplot(gs[1, 1])

        if is_memfcs:
            memfcs_R_h     = res.get("memfcs_R_h_nm")
            memfcs_amps    = res.get("memfcs_amplitudes")
            memfcs_R_h_jy  = res.get("memfcs_R_h_nm_jy")
            memfcs_amps_jy = res.get("memfcs_amplitudes_jy")

            if memfcs_R_h is not None and memfcs_amps is not None:
                memfcs_R_h  = np.asarray(memfcs_R_h,  dtype=float)
                memfcs_amps = np.asarray(memfcs_amps, dtype=float)
                ax11.semilogx(memfcs_R_h, memfcs_amps,
                              color="mediumpurple", linewidth=2,
                              label="Shannon")
                max_R_h  = res.get("memfcs_max_freq_R_h")
                mean_R_h = res.get("memfcs_R_h_mean_nm")
                if max_R_h is not None:
                    ax11.axvline(float(max_R_h), color="tomato",
                                 linestyle="--", linewidth=1.5,
                                 label=f"Sh peak {float(max_R_h):.2f} nm")
                if mean_R_h is not None:
                    ax11.axvline(float(mean_R_h), color="orange",
                                 linestyle=":", linewidth=1.5,
                                 label=f"Sh mean {float(mean_R_h):.2f} nm")

            if memfcs_R_h_jy is not None and memfcs_amps_jy is not None:
                memfcs_R_h_jy  = np.asarray(memfcs_R_h_jy,  dtype=float)
                memfcs_amps_jy = np.asarray(memfcs_amps_jy, dtype=float)
                ax11.semilogx(memfcs_R_h_jy, memfcs_amps_jy,
                              color="navy", linewidth=2,
                              linestyle="--", label="Jaynes")
                max_R_h_jy  = res.get("memfcs_max_freq_R_h_jy")
                mean_R_h_jy = res.get("memfcs_R_h_mean_nm_jy")
                if max_R_h_jy is not None:
                    ax11.axvline(float(max_R_h_jy), color="navy",
                                 linestyle="--", linewidth=1.5,
                                 label=f"Jy peak {float(max_R_h_jy):.2f} nm")
                if mean_R_h_jy is not None:
                    ax11.axvline(float(mean_R_h_jy), color="steelblue",
                                 linestyle=":", linewidth=1.5,
                                 label=f"Jy mean {float(mean_R_h_jy):.2f} nm")

            ax11.set_xlabel("Hydrodynamic radius R_h (nm)")
            ax11.set_ylabel("Amplitude")
            ax11.set_title("MEMFCS — R_h distribution\n"
                           "purple=Shannon  navy=Jaynes")
            ax11.legend(fontsize=7)
            ax11.grid(True, alpha=0.3)

        else:
            finite = np.isfinite(wr)
            ax11.hist(wr[finite], bins=40, density=True)
            ax11.set_xlabel("Weighted residual")
            ax11.set_ylabel("Density")
            ax11.set_title("Residual distribution")

        self.fcsfit_fig.tight_layout()
        self.fcsfit_canvas.draw()

        # ── per-file outputs ──────────────────────────────────────
        edit_path = self.fcs_make_edit_path(base_path, fitting_model)

        if write_summary:
            self.fcsfit_fig.savefig(
                edit_path + ".svg", dpi=300,
                bbox_inches="tight", facecolor="white"
            )
            cc_fits_df = pd.DataFrame({
                "tau": tau, "G": G, "sigma G": sigma, "cc Fit": pred
            })
            cc_fits_df.to_csv(edit_path + ".csv", header=True, index=False)

        if not is_memfcs:
            reIMSD_local = None
            aR     = res.get("PSF_aspect_ratio")
            N      = res.get("N")
            offset = res.get("offset", 0.0)
            if (fitting_model not in ["siFCS", "siFCSTwoComponents"]
                    and aR is not None and N is not None):
                try:
                    
                    reIMSD_local = self._calculate.iMSD_calc(
                        tau, float(aR), float(N), pred, float(offset)
                    )
                except Exception:
                    pass
            if reIMSD_local is not None:
                iMSD_df = pd.DataFrame({"tau": tau, "iMSD": reIMSD_local})
                iMSD_df.to_csv(edit_path + "_iMSD.csv",
                               header=True, index=False)

        if write_summary:
            summary_csv = os.path.join(
                os.path.dirname(edit_path),
                f"{fitting_model}_fit_summary.csv"
            )
            estimate = res.get("estimate_data", {})
            row = {}
            for k, v in estimate.items():
                if v == [None]:
                    continue
                row[k] = v[0] if (isinstance(v, list) and len(v) == 1) else v
            row["Filename"] = base_path
            df = pd.DataFrame([row])
            if not os.path.exists(summary_csv):
                df.to_csv(summary_csv, header=True, index=False)
            else:
                df.to_csv(summary_csv, mode="a", header=False, index=False)

        if is_memfcs:
            self._log_memfcs_interpretation(res)

        self.log_message(
            f"Saved FCS outputs to: {os.path.dirname(edit_path)}"
        )

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
                elif msg_type == "file_error":
                    self.log_message(payload)
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
                    self._showerror("FCS Fit Error", "FCS fitting failed. See log.")
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
            self._showwarning("Warning", "FRAP analysis is already running.")
            return
        czi_path = self.frap_czi.get().strip()
        folder = self.frap_folder.get().strip()

        if not czi_path and not folder:
            self._showwarning("Warning", "Select a single CZI or a batch folder.")
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
            "no_control": bool(self.frap_no_control.get()),
            "n_rois": int(n_rois_text) if n_rois_text else None,
            "ctrl_idx": int(ctrl_idx_text) if ctrl_idx_text and not self.frap_no_control.get() else None,
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
            target=self._frap_process_main,
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

        #  ctrl_idx may now be None 
        ctrl_idx = res.get("ctrl_idx")          # None in no-control mode
        no_control = res.get("no_control", False)

        frap_idxs = list(res["frap_idxs"])
        fit_results = res["fit_results"]
        colors = res["roi_colors"]
        imaging_bleach = bool(res["imaging_bleach"])

        tb_s = bleach_frame * dt
        T = t_all[-1] - t_all[0]

        model_label = ' + imaging bleach' if imaging_bleach else 'no imaging bleach'
        ctrl_label = ' | no control ROI' if no_control else ''
        self.frap_fig.suptitle(
            f"{stem}   [{model_label}{ctrl_label}]",
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

        #  Panel A: raw traces 
        _style(axA, 'Raw intensity (pre-normalisation)', 'Time [s]', 'Mean intensity [counts]')

        if ctrl_idx is not None:
            # draw control trace in grey
            axA.plot(t_all, raw_traces[ctrl_idx], color='#AAAAAA', lw=1.5, ls='--',
                     label=f'ROI {ctrl_idx+1} — control')

        for k, fi in enumerate(frap_idxs):
            axA.plot(t_all, raw_traces[fi], color=colors[k], lw=2, label=f'ROI {fi+1} — FRAP')

        axA.autoscale(axis='y')
        _vline(axA)
        axA.legend(fontsize=8, frameon=False)

        #  Panels B, C, D: unchanged — ctrl_idx not needed 
        _style(axB, 'Normalised fit', 'Time [s]', 'Normalised intensity [counts]')
        xs = np.linspace(0, len(t_all) - 1, 1000)
        ts = xs * dt
        for k, fi in enumerate(frap_idxs):
            if fit_results[k] is None:
                continue
            popt = fit_results[k][0]
            axB.plot(t_all, norm_traces[fi], 'o', color=colors[k], ms=2.5, alpha=0.35)
            axB.plot(ts, self._frap_analysis.evaluate_model(xs, popt, imaging_bleach), '-',
                     color=colors[k], lw=2.2,
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

            axC.plot(t_post, np.clip(corr, -0.3, f_mob_f * 1.4),
                     'o', color=colors[k], ms=2.5, alpha=0.4)
            S_fit = self._frap_analysis._soumpasis(x_abs, x_0f, R_f, D_f)
            axC.plot(t_rel, f_mob_f * S_fit, '-', color=colors[k], lw=2.2,
                     label=(f'ROI {fi+1}: Mobile fraction={f_mob_f:.2f}, '
                            f'f_bl={f_bl_f:.2f}, t½={fit_results[k][2]:.2f} s'))
            axC.axhline(f_mob_f, color=colors[k], lw=0.8, ls='--', alpha=0.45)
            axC.axvline(fit_results[k][2], color=colors[k], lw=0.8, ls=':', alpha=0.65)

        axC.set_xlim(left=0)
        max_fm = max(
            (fit_results[k][0][4] for k in range(len(frap_idxs)) if fit_results[k] is not None),
            default=1.0
        )
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
            resid = (norm_traces[fi][bleach_frame:].astype(float)
                     - self._frap_analysis.evaluate_model(post_i, popt, imaging_bleach)) / pre_mu
            all_r.extend(resid.tolist())
            win = max(3, len(resid) // 15)
            if len(resid) > win * 2:
                rm = np.convolve(resid, np.ones(win) / win, mode='valid')
                axD.plot(t_post[win // 2: win // 2 + len(rm)], rm,
                         '-', color=colors[k], lw=1.8, label=f'ROI {fi+1}')

        axD.set_xlim(left=0)
        ylim = max(0.12, np.percentile(np.abs(all_r), 99) * 1.3) if all_r else 0.12
        axD.set_ylim(-ylim, ylim)
        axD.legend(fontsize=8, frameon=False)

        self.frap_canvas.draw()

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
                    self._showerror("FRAP Error", "FRAP analysis failed. See log.")
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
            self._showwarning("Warning", "Diffusion map is already running.")
            return
        if not self.input_file_diff_map.get():
            self._showwarning("Warning", "Please select an input file first")
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
            target=self._diffusion_map_process_main,
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
                    self._showerror("Diffusion Map Error", "Diffusion map failed. See log.")
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
# -------------------------------------------------ICS GUI-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def create_ics_tab(self):
        from theatrics.workers.ics_worker import ics_process_main
        self._ics_process_main = ics_process_main
        ics_frame = ttk.Frame(self.notebook)
        self.notebook.add(ics_frame, text="ICS")

        params_frame = ttk.LabelFrame(ics_frame, text="ICS Parameters", padding=10)
        params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        row = 0
        #  single file 
        ttk.Label(params_frame, text="Single TIFF / PTU:").grid(
            row=row, column=0, sticky="w")
        self.ics_tiff = tk.StringVar()
        e1 = ttk.Entry(params_frame, textvariable=self.ics_tiff, width=28)
        e1.grid(row=row, column=1, sticky="ew")
        b1 = ttk.Button(params_frame, text="Browse",
                         command=self._browse_ics_tiff)
        b1.grid(row=row, column=2, padx=5)
        self.register_busy_widget(e1)
        self.register_busy_widget(b1)

        row += 1
        #  batch folder 
        ttk.Label(params_frame, text="Batch folder (TIFF or PTU):").grid(
            row=row, column=0, sticky="w")
        self.ics_folder = tk.StringVar()
        e2 = ttk.Entry(params_frame, textvariable=self.ics_folder, width=28)
        e2.grid(row=row, column=1, sticky="ew")
        b2 = ttk.Button(params_frame, text="Browse",
                         command=self._browse_ics_folder)
        b2.grid(row=row, column=2, padx=5)
        self.register_busy_widget(e2)
        self.register_busy_widget(b2)

        row += 1
        ttk.Label(params_frame, text="File pattern:").grid(row=row, column=0, sticky="w")
        self.ics_pattern = tk.StringVar(value="*.tiff")
        pattern_combo = ttk.Combobox(
            params_frame,
            textvariable=self.ics_pattern,
            values=["*.tiff", "*.tif", "*.ptu"],
            width=18,
        )
        pattern_combo.grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(params_frame, text="Channel:").grid(
            row=row, column=0, sticky="w")
        self.ics_channel = tk.StringVar(value="0")
        ttk.Combobox(params_frame, textvariable=self.ics_channel,
                     values=["0", "1", "2", "3", "4"], width=5).grid(
            row=row, column=1, sticky="w")
        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6)

        row += 1
        ttk.Label(params_frame, text="Block length (frames):").grid(
            row=row, column=0, sticky="w")
        self.ics_block_length = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=self.ics_block_length,
                  width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Frame skip (lag τ):").grid(
            row=row, column=0, sticky="w")
        self.ics_frame_skip = tk.StringVar(value="1")
        ttk.Entry(params_frame, textvariable=self.ics_frame_skip,
                  width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Bin frames:").grid(
            row=row, column=0, sticky="w")
        self.ics_bin_frames = tk.StringVar(value="1")
        ttk.Entry(params_frame, textvariable=self.ics_bin_frames,
                  width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Threshold multiplier:").grid(
            row=row, column=0, sticky="w")
        self.ics_threshold_mult = tk.StringVar(value="0.1")
        ttk.Entry(params_frame, textvariable=self.ics_threshold_mult,
                  width=10).grid(row=row, column=1, sticky="w")

        row += 1
        self.ics_save_block_images = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame,
                         text="Save block images (MIP / mean / mask / G map)",
                         variable=self.ics_save_block_images).grid(
            row=row, column=0, columnspan=2, sticky="w")

        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6)

        row += 1
        btn_frame = ttk.Frame(params_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)

        self.ics_run_btn = ttk.Button(btn_frame, text="Run ICS",
                                       command=self._run_ics)
        self.ics_run_btn.pack(side=tk.LEFT, padx=5)
        self.register_busy_widget(self.ics_run_btn)

        #  right-side display 
        display_frame = ttk.LabelFrame(ics_frame, text="ICS Display", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,
                           padx=5, pady=5)

        self.ics_fig = Figure(figsize=(9, 7), dpi=100, facecolor="white")
        self.ics_canvas = FigureCanvasTkAgg(self.ics_fig, display_frame)
        self.ics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.ics_canvas, display_frame)
        toolbar.update()

    #  browse helpers 
    def _browse_ics_tiff(self):
        fn = self._ask_open_filename(
            title="Select TIFF or PTU file",
            filetypes=[
                ("All supported files", "*.tif *.tiff *.ptu"),
                ("TIFF files",          "*.tif *.tiff"),
                ("PTU files",           "*.ptu"),
                ("All files",           "*.*"),
            ]
        )
        if fn:
            self.ics_tiff.set(fn)

    def _browse_ics_folder(self):
        folder = self._ask_directory(title="Select ICS batch folder")
        if folder:
            self.ics_folder.set(folder)

    #  run 
    def _run_ics(self):
        if self._is_worker_running("ics_proc"):
            self._showwarning("Warning", "ICS is already running.")
            return

        tiff_path = self.ics_tiff.get().strip()
        folder    = self.ics_folder.get().strip()

        if not tiff_path and not folder:
            self._showwarning("Warning",
                                   "Select a single TIFF or a batch folder.")
            return

        mode = "single" if tiff_path else "batch"

        config = {
            "block_length":             self._safe_int(
                                            self.ics_block_length,
                                            "Block length", 10),
            "frame_skip":               self._safe_int(
                                            self.ics_frame_skip,
                                            "Frame skip", 1),
            "bin_frames":               self._safe_int(
                                            self.ics_bin_frames,
                                            "Bin frames", 1),
            "threshold_multiplication": self._safe_float(
                                            self.ics_threshold_mult,
                                            "Threshold multiplier", 0.1),
            "save_block_images":        bool(self.ics_save_block_images.get()),
            "pattern":                  self.ics_pattern.get().strip(),
            "channel":                  self._safe_int(
                                            self.ics_channel,
                                            "Channel", 0),
        }

        self.log_message(f"Starting ICS ({mode})...")
        self.status_var.set("Running ICS...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        self.ics_queue        = multiprocessing.Queue()
        self.ics_cancel_event = multiprocessing.Event()

        params = {"mode": mode, "config": config}
        if mode == "single":
            params["tiff_path"] = tiff_path
        else:
            params["folder"] = folder

        self.ics_proc = multiprocessing.Process(
            target=self._ics_process_main,
            args=(params, self.ics_queue, self.ics_cancel_event),
            daemon=False,
        )
        self.ics_proc.start()
        self._poll_ics_queue()

    #  display 
    def _update_ics_display(self, res: dict):
        """
        Draw the 2×2 overview for the most recently completed TIFF.
        Works for both single and batch (shows last processed file).
        """
        self.ics_fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.ics_fig,
                               hspace=0.35, wspace=0.3)

        df          = res.get("blocks_df")
        mean_stack  = res.get("mean_stack")
        G_stack     = res.get("G_stack")
        stem        = res.get("stem", "")

        self.ics_fig.suptitle(f"ICS — {stem}", fontsize=11,
                              fontweight="bold")

        #  top-left: mean G with SD error bars 
        ax00 = self.ics_fig.add_subplot(gs[0, 0])
        if df is not None and "mean_G" in df.columns:
            ax00.errorbar(df.index, df["mean_G"],
                          yerr=df["sd_G"],
                          marker="o", capsize=4,
                          color="steelblue", linewidth=1.5)
        ax00.set_xlabel("Block number")
        ax00.set_ylabel("Mean G")
        ax00.set_title("ICS correlation vs time")
        ax00.spines[["top", "right"]].set_visible(False)
        ax00.grid(True, alpha=0.3)

        #  top-right: normalised G 
        ax01 = self.ics_fig.add_subplot(gs[0, 1])
        if df is not None and "Normalized" in df.columns:
            ax01.plot(df.index, df["Normalized"],
                      marker="o", color="seagreen", linewidth=1.5)
            ax01.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax01.set_xlabel("Block number")
        ax01.set_ylabel("Normalised G")
        ax01.set_title("Normalised correlation")
        ax01.spines[["top", "right"]].set_visible(False)
        ax01.grid(True, alpha=0.3)

        #  bottom-left: mean intensity of last block 
        ax10 = self.ics_fig.add_subplot(gs[1, 0])
        if mean_stack is not None and len(mean_stack) > 0:
            im = ax10.imshow(mean_stack[-1], cmap="gray")
            self.ics_fig.colorbar(im, ax=ax10, fraction=0.046, pad=0.04)
        ax10.set_title("Mean intensity (last block)")
        ax10.axis("off")

        #  bottom-right: G map of last block 
        ax11 = self.ics_fig.add_subplot(gs[1, 1])
        if G_stack is not None and len(G_stack) > 0:
            vmax = float(np.nanpercentile(G_stack[-1], 99))
            im = ax11.imshow(G_stack[-1], cmap="hot",
                             vmin=0, vmax=max(vmax, 1e-12))
            self.ics_fig.colorbar(im, ax=ax11, fraction=0.046, pad=0.04)
        ax11.set_title("G map (last block)")
        ax11.axis("off")

        self.ics_canvas.draw()

    #  queue polling 
    def _poll_ics_queue(self):
        try:
            while True:
                msg_type, payload = self.ics_queue.get_nowait()

                if msg_type == "progress":
                    self.set_ui_busy(True)
                    self.progress_var.set(float(payload))

                elif msg_type == "block_done":
                    # lightweight log — no redraw for every block
                    info = payload
                    self.log_message(
                        f"  {info['stem']} — block "
                        f"{info['block']+1}/{info['n_blocks']}  "
                        f"mean G = {info['stats']['mean_G']:.4f}"
                        if np.isfinite(info['stats']['mean_G'])
                        else f"  {info['stem']} — block "
                             f"{info['block']+1}/{info['n_blocks']}  "
                             f"mean G = NaN (empty mask)"
                    )

                elif msg_type == "file_done":
                    # update the plot for the most recently finished file
                    self._update_ics_display(payload)
                    self.log_message(
                        f"Finished: {payload.get('stem')}  "
                        f"({payload.get('n_blocks')} blocks)  "
                        f"CSV: {payload.get('csv_path')}"
                    )

                elif msg_type == "error_file":
                    self.log_message(f"FAILED: {payload}")

                elif msg_type == "done":
                    # single-file: payload is the result dict
                    # batch: payload is the summary dict
                    if "blocks_df" in payload:
                        # single file result
                        self._update_ics_display(payload)
                        self.log_message(
                            f"ICS complete — {payload.get('n_blocks')} blocks  "
                            f"SVG: {payload.get('svg_path')}"
                        )
                    else:
                        # batch summary
                        n_ok     = payload.get("n_ok", 0)
                        n_failed = payload.get("n_failed", 0)
                        self.log_message(
                            f"ICS batch complete — "
                            f"{n_ok} succeeded, {n_failed} failed"
                        )
                        if payload.get("combined_csv"):
                            self.log_message(
                                f"Global CSV: {payload['combined_csv']}"
                            )
                        if payload.get("combined_fig"):
                            self.log_message(
                                f"Global plot: {payload['combined_fig']}"
                            )

                    self.set_ui_busy(False)
                    self.status_var.set("Ready")
                    self.progress_bar.grid_remove()
                    self.log_message("ICS done.")
                    return

                elif msg_type == "cancelled":
                    self.log_message("ICS cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return

                elif msg_type == "error":
                    self.log_message(payload)
                    self.status_var.set("Error")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    self._showerror("ICS Error",
                                         "ICS failed. See log.")
                    return

        except queue.Empty:
            pass

        if self.ics_proc is not None and not self.ics_proc.is_alive():
            self.set_ui_busy(False)
            self.status_var.set("Error")
            self.progress_bar.grid_remove()
            self.log_message("ICS worker terminated unexpectedly.")
            return

        self.root.after(50, self._poll_ics_queue)
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- Vesicle Finder GUI ------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------

    def create_vesicle_tab(self):
        from theatrics.workers.vesicle_worker import vesicle_process_main
        self._vesicle_process_main = vesicle_process_main
        from theatrics.vesicle import detection as vesicle_detection
        self._vesicle_detection = vesicle_detection
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
                     values=["0", "1", "2", "3", "4"], width=5).grid(row=row, column=1, sticky="w")
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
                     values=["hough", "cellpose","weighted_intensity","hough_transmitted", "otsu"], width=10)
        method_cmb.grid(row=row, column=1, sticky="w")
        # Weighted intensity-specific params (shown/hidden based on method)
        row += 1
        self.vesicle_weight_frame = ttk.LabelFrame(params_frame, text="Weighted Intensity Circle Parameters", padding=5)
        self.vesicle_weight_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)

        wr = 0
        ttk.Label(self.vesicle_weight_frame, text="Min radius (µm):").grid(row=wr, column=0, sticky="w")
        self.vesicle_min_radius = tk.StringVar(value="5.0")
        ttk.Entry(self.vesicle_weight_frame, textvariable=self.vesicle_min_radius, width=8).grid(row=wr, column=1, sticky="w")

        wr += 1
        ttk.Label(self.vesicle_weight_frame, text="Max radius (µm):").grid(row=wr, column=0, sticky="w")
        self.vesicle_max_radius = tk.StringVar(value="25.0")
        ttk.Entry(self.vesicle_weight_frame, textvariable=self.vesicle_max_radius, width=8).grid(row=wr, column=1, sticky="w")

        wr += 1
        ttk.Label(self.vesicle_weight_frame, text="Search range (um):").grid(row=wr, column=0, sticky="w")
        self.search_range = tk.StringVar(value="2.0")
        ttk.Entry(self.vesicle_weight_frame, textvariable=self.search_range, width=8).grid(row=wr, column=1, sticky="w")
        wr += 1
        ttk.Label(self.vesicle_weight_frame, text="Threshold method:").grid(row=wr, column=0, sticky="w")
        self.vesicle_threshold_method = tk.StringVar(value="huang")
        ttk.Combobox(self.vesicle_weight_frame, textvariable=self.vesicle_threshold_method,
                     values=["huang", "otsu", "yen", "triangle", "mean", "li"],
                     width=10).grid(row=wr, column=1, sticky="w")
        
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
            self.vesicle_cellpose_frame.grid_remove()
            self.vesicle_hough_frame.grid_remove()
            self.vesicle_weight_frame.grid_remove()
            m = self.vesicle_method.get()
            if m in ("hough", "hough_transmitted"):
                self.vesicle_hough_frame.grid()
            elif m == "weighted_intensity":
                self.vesicle_weight_frame.grid()
            elif m == "cellpose":
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
        row += 1
        self.vesicle_debug = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Save debug images",
                        variable=self.vesicle_debug).grid(row=row, column=0, columnspan=3, sticky="w")
        self.vesicle_crop_btn = ttk.Button(btn_frame, text="Crop Selected", command=self._run_vesicle_crop,
                                           state="disabled")
        self.vesicle_crop_btn.pack(side=tk.LEFT, padx=5)

        self.vesicle_crop_all_btn = ttk.Button(btn_frame, text="Export All", command=self._run_vesicle_crop_all,
                                               state="disabled")
        self.vesicle_crop_all_btn.pack(side=tk.LEFT, padx=5)

        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)

        row += 1
        ttk.Label(params_frame, text=" Membrane Straightening ", font=("", 10, "bold")).grid(
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
                     values=["0", "1", "2", "3", "4"], width=5).grid(row=row, column=1, sticky="w")

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
        fn = self._ask_open_filename(
            title="Select CZI or PTU file",
            filetypes=[
                ("All supported files", "*.czi *.ptu"),
                ("CZI files",           "*.czi"),
                ("PTU files",           "*.ptu"),
                ("All files",           "*.*"),
            ]
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
            weight_search_range = self._safe_float(self.search_range, "Search Range", 2.0),
            threshold_method=self.vesicle_threshold_method.get(),
            fallback_pixel_size_um = fallback_pixel_size_um,
            debug=bool(self.vesicle_debug.get()),
        )
    # ---- DETECT ----
    def _run_vesicle_detect(self):
        if self._is_worker_running("vesicle_proc"):
            self._showwarning("Warning", "Vesicle detection is already running.")
            return

        if not self.vesicle_czi.get().strip():
            self._showwarning("Warning", "Please select a CZI file.")
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
            target=self._vesicle_process_main,
            args=(params, self.vesicle_queue, self.vesicle_cancel_event),
            daemon=False,
        )
        self.vesicle_proc.start()
        self._poll_vesicle_queue()

    # ---- CROP SELECTED ----
    def _run_vesicle_crop(self):
        selected = list(self._vesicle_selected_labels)
        if not selected:
            self._showwarning("Warning", "No vesicles selected. Click on the image or listbox.")
            return
        self._start_vesicle_crop(selected)

    # ---- CROP ALL ----
    def _run_vesicle_crop_all(self):
        if self._vesicle_detect_result is None:
            return
        all_labels = [v["label"] for v in self._vesicle_detect_result["vesicles"]]
        if not all_labels:
            self._showwarning("Warning", "No vesicles detected.")
            return
        self._start_vesicle_crop(all_labels)

    def _start_vesicle_crop(self, labels):
        if self._is_worker_running("vesicle_proc"):
            self._showwarning("Warning", "Vesicle processing is already running.")
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
            target=self._vesicle_process_main,
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
                    self._showerror("Vesicle Error", "Vesicle detection failed. See log.")
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

        if not self._vesicle_detection.CELLPOSE_AVAILABLE:
            self.log_message("NOTE: Cellpose not installed; used Otsu fallback segmentation.")

        if self.vesicle_method.get() in ("hough", "hough_transmitted"):
            from theatrics.vesicle.detection import OPENCV_AVAILABLE
            if OPENCV_AVAILABLE:
                self.log_message(f"Hough detection ({self.vesicle_method.get()}): using OpenCV")
            else:
                self.log_message(f"Hough detection ({self.vesicle_method.get()}): using skimage")
        if self.vesicle_method.get() == "weighted_intensity":
            self.log_message("Detection: weighted peripheral intensity method (improved and modified from Kohyama et al. 2022)")
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
            self._showwarning("Warning", "No vesicles selected.")
            return
        vesicles = [v for v in self._vesicle_detect_result["vesicles"] if v["label"] in selected]
        if not all("radius" in v for v in vesicles):
            self._showwarning("Warning", "Straightening requires Hough-detected vesicles with known radius.")
            return
        self._start_vesicle_straighten(vesicles)

    def _run_vesicle_straighten_all(self):
        if self._vesicle_detect_result is None:
            return
        vesicles = [v for v in self._vesicle_detect_result["vesicles"] if "radius" in v]
        if not vesicles:
            self._showwarning("Warning", "No vesicles with known radius. Use Hough detection.")
            return
        self._start_vesicle_straighten(vesicles)

    def _start_vesicle_straighten(self, vesicles):
        if self._is_worker_running("vesicle_proc"):
            self._showwarning("Warning", "Vesicle processing is already running.")
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
            target=self._vesicle_process_main,
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
        # self.vesicle_fig.set_constrained_layout(True)
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

            #  Panel 1: straightened strip (frame 0) 
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
            # ax1.set_rasterized(True)   # ← fix for SVG blank panel

            #  Panel 2: heatmap (time vs angle) 
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
                # ax2.set_rasterized(True)   # ← fix for SVG blank panel
            else:
                # single frame: show as a 1D intensity profile line plot
                ax2.plot(x_um, profile[0], "r-", linewidth=1.5)
                ax2.set_xlabel("Position along membrane (µm)")
                ax2.set_ylabel("Mean intensity")
                ax2.set_title(f"Vesicle {lbl} — membrane profile (single frame)", fontsize=9)
                ax2.grid(True, alpha=0.3)

            #  Panel 3: total intensity vs time 
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
    # ------------------------------------------------- Vesicle Finder GUI ------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    def create_afm_tab(self):
        from theatrics.workers.afm_worker import afm_worker_main
        self._afm_worker_main = afm_worker_main
        afm_frame = ttk.Frame(self.notebook)
        self.notebook.add(afm_frame, text="AFM")

        #  left panel: parameters 
        params_frame = ttk.LabelFrame(
            afm_frame, text="AFM Parameters", padding=10
        )
        params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        row = 0
        ttk.Label(params_frame, text="JPK file:").grid(
            row=row, column=0, sticky="w"
        )
        self.afm_filepath = tk.StringVar()
        e1 = ttk.Entry(params_frame, textvariable=self.afm_filepath, width=28)
        e1.grid(row=row, column=1, sticky="ew")
        b1 = ttk.Button(params_frame, text="Browse",
                        command=self._browse_afm_file)
        b1.grid(row=row, column=2, padx=5)
        self.register_busy_widget(e1)
        self.register_busy_widget(b1)

        row += 1
        ttk.Label(params_frame, text="Channel:").grid(
            row=row, column=0, sticky="w"
        )
        self.afm_channel = tk.StringVar(value="height_trace")
        ttk.Combobox(
            params_frame, textvariable=self.afm_channel,
            values=["height_trace", "height_retrace",
                    "adhesion_force_trace", "stiffness_trace"],
            width=20
        ).grid(row=row, column=1, sticky="w")

        row += 1
        load_btn = ttk.Button(
            params_frame, text="Load File",
            command=self._afm_load_file
        )
        load_btn.grid(row=row, column=0, columnspan=2,
                      pady=6, sticky="ew")
        self.register_busy_widget(load_btn)

        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4
        )

        #  profile parameters 
        row += 1
        ttk.Label(params_frame,
                  text=" Profile parameters ",
                  font=("", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )

        row += 1
        ttk.Label(params_frame, text="Fit points:").grid(
            row=row, column=0, sticky="w"
        )
        self.afm_n_fit_points = tk.IntVar(value=15)
        fit_scale = ttk.Scale(
            params_frame,
            from_=3, to=60,
            variable=self.afm_n_fit_points,
            orient="horizontal",
            command=self._afm_on_fit_points_changed,
        )
        fit_scale.grid(row=row, column=1, sticky="ew")
        self.afm_fit_points_label = ttk.Label(params_frame, text="15")
        self.afm_fit_points_label.grid(row=row, column=2, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Smooth window:").grid(
            row=row, column=0, sticky="w"
        )
        self.afm_smooth_window = tk.StringVar(value="15")
        ttk.Entry(params_frame, textvariable=self.afm_smooth_window,
                  width=8).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Smooth poly order:").grid(
            row=row, column=0, sticky="w"
        )
        self.afm_smooth_poly = tk.StringVar(value="3")
        ttk.Entry(params_frame, textvariable=self.afm_smooth_poly,
                  width=8).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame, text="Profile points:").grid(
            row=row, column=0, sticky="w"
        )
        self.afm_n_points = tk.StringVar(value="300")
        ttk.Entry(params_frame, textvariable=self.afm_n_points,
                  width=8).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(params_frame,
                  text="Threshold fraction (0-1):").grid(
            row=row, column=0, sticky="w"
        )
        self.afm_threshold_frac = tk.StringVar(value="0.05")
        ttk.Entry(params_frame, textvariable=self.afm_threshold_frac,
                  width=8).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4
        )

        #  instructions 
        row += 1
        instructions = (
            "How to draw a profile:\n"
            "1. Click START on bare membrane\n"
            "2. Click END   on bare membrane\n"
            "   (line must cross the condensate)\n\n"
            "The two clicked heights define the\n"
            "linear baseline (membrane plane)."
        )
        ttk.Label(params_frame, text=instructions,
                  justify="left",
                  foreground="gray").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=4
        )

        row += 1
        ttk.Separator(params_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4
        )

        #  action buttons 
        row += 1
        btn_row = ttk.Frame(params_frame)
        btn_row.grid(row=row, column=0, columnspan=3, pady=4)

        self.afm_reset_btn = ttk.Button(
            btn_row, text="Reset last",
            command=self._afm_reset_last,
            state="disabled"
        )
        self.afm_reset_btn.pack(side=tk.LEFT, padx=3)

        self.afm_clear_btn = ttk.Button(
            btn_row, text="Clear all",
            command=self._afm_clear_all,
            state="disabled"
        )
        self.afm_clear_btn.pack(side=tk.LEFT, padx=3)

        self.afm_save_btn = ttk.Button(
            btn_row, text="Save results",
            command=self._afm_save_results,
            state="disabled"
        )
        self.afm_save_btn.pack(side=tk.LEFT, padx=3)

        #  results table 
        row += 1
        ttk.Label(params_frame,
                  text="Profiles (click to highlight):",
                  font=("", 9, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w"
        )

        row += 1
        self.afm_profile_listbox = tk.Listbox(
            params_frame, height=8, width=38,
            selectmode=tk.SINGLE
        )
        self.afm_profile_listbox.grid(
            row=row, column=0, columnspan=3, sticky="ew"
        )
        self.afm_profile_listbox.bind(
            "<<ListboxSelect>>", self._afm_on_listbox_select
        )

        #  right panel: figure 
        display_frame = ttk.LabelFrame(
            afm_frame, text="AFM Display", padding=5
        )
        display_frame.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        self.afm_fig = Figure(figsize=(11, 7), dpi=100,
                              facecolor="white")
        self.afm_canvas = FigureCanvasTkAgg(self.afm_fig, display_frame)
        self.afm_canvas.get_tk_widget().pack(
            fill=tk.BOTH, expand=True
        )
        toolbar = NavigationToolbar2Tk(self.afm_canvas, display_frame)
        toolbar.update()

        # bind click on the AFM image axes
        self.afm_canvas.mpl_connect(
            "button_press_event", self._afm_on_canvas_click
        )

        #  internal state 
        self._afm_height_nm    = None    # loaded ndarray
        self._afm_pixel_size   = None    # nm/px
        self._afm_scan_size_um = None
        self._afm_click_points = []      # accumulates (x_px, y_px, h_nm)
        self._afm_profiles     = []      # list of result dicts
        self._afm_ax_image     = None    # axes reference
        self._afm_ax_profile   = None
        self._afm_colorbar     = None

        self._afm_draw_empty()

    # 
    # Browse / load
    # 

    def _browse_afm_file(self):
        fn = self._ask_open_filename(
            title="Select JPK file",
            filetypes=[
                ("JPK QI image", "*.jpk-qi-image"),
                ("JPK image",    "*.jpk"),
                ("All files",    "*.*"),
            ]
        )
        if fn:
            self.afm_filepath.set(fn)

    def _afm_load_file(self):
        fp = self.afm_filepath.get().strip()
        if not fp:
            self._showwarning("Warning", "Please select a JPK file.")
            return

        if self._is_worker_running("afm_proc"):
            self._showwarning("Warning", "AFM worker is already running.")
            return

        self.log_message(f"Loading AFM file: {fp}")
        self.status_var.set("Loading AFM file...")
        self.progress_var.set(0.0)
        self.progress_bar.grid()

        self.afm_queue        = multiprocessing.Queue()
        self.afm_cancel_event = multiprocessing.Event()

        self.afm_proc = multiprocessing.Process(
            target=self._afm_worker_main,
            args=(
                {"task": "load",
                 "filepath": fp,
                 "channel":  self.afm_channel.get()},
                self.afm_queue,
                self.afm_cancel_event,
            ),
            daemon=False,
        )
        self.afm_proc.start()
        self._poll_afm_queue()

    # 
    # Canvas interaction
    # 

    def _afm_on_canvas_click(self, event):
        """Handle mouse click on the AFM image axes."""
        if self._afm_height_nm is None:
            return
        if event.inaxes is not self._afm_ax_image:
            return
        if event.button != 1:
            return

        scan = self._afm_scan_size_um
        ny, nx = self._afm_height_nm.shape

        x_px = float(np.clip(
            event.xdata / scan * nx, 0, nx - 1
        ))
        y_px = float(np.clip(
            event.ydata / scan * ny, 0, ny - 1
        ))

        # sample height at click
        from scipy import ndimage as _ndi
        h_nm = float(_ndi.map_coordinates(
            self._afm_height_nm, [[y_px], [x_px]], order=1
        )[0])

        self._afm_click_points.append((x_px, y_px, h_nm))

        # draw marker on image
        self._afm_ax_image.plot(
            event.xdata, event.ydata,
            "+", color="cyan", markersize=14, markeredgewidth=2,
            zorder=10
        )
        self.afm_canvas.draw_idle()

        self.log_message(
            f"  Click {len(self._afm_click_points)}: "
            f"x={x_px:.1f}px  y={y_px:.1f}px  h={h_nm:.2f} nm"
        )

        if len(self._afm_click_points) == 2:
            self._afm_compute_profile()
            self._afm_click_points = []

    # 
    # Profile computation
    # 

    def _afm_compute_profile(self):
        (x0, y0, h0), (x1, y1, h1) = self._afm_click_points

        if self._is_worker_running("afm_proc"):
            self._showwarning("Warning", "AFM worker is busy.")
            return

        params = {
            "task":         "profile",
            "height_nm":    self._afm_height_nm,
            "pixel_size_nm": self._afm_pixel_size,
            "start_px":     (x0, y0),
            "end_px":       (x1, y1),
            "h_start_nm":   h0,
            "h_end_nm":     h1,
            "n_fit_points": int(self.afm_n_fit_points.get()),
            "smooth_window": self._safe_int(
                self.afm_smooth_window, "Smooth window", 15
            ),
            "smooth_poly":  self._safe_int(
                self.afm_smooth_poly, "Smooth poly", 3
            ),
            "n_points":     self._safe_int(
                self.afm_n_points, "Profile points", 300
            ),
            "threshold_fraction": self._safe_float(
                self.afm_threshold_frac, "Threshold fraction", 0.05
            ),
        }

        self.status_var.set("Computing profile...")
        self.afm_queue        = multiprocessing.Queue()
        self.afm_cancel_event = multiprocessing.Event()

        self.afm_proc = multiprocessing.Process(
            target=self._afm_worker_main,
            args=(params, self.afm_queue, self.afm_cancel_event),
            daemon=False,
        )
        self.afm_proc.start()
        self._poll_afm_queue()

    # 
    # Fit-points slider callback
    # 

    def _afm_on_fit_points_changed(self, _val):
        nfp = int(self.afm_n_fit_points.get())
        self.afm_fit_points_label.configure(text=str(nfp))

        if not self._afm_profiles:
            return

        idx = len(self._afm_profiles) - 1
        sel = self.afm_profile_listbox.curselection()
        if sel:
            idx = int(sel[0])

        last = self._afm_profiles[idx]

        if self._is_worker_running("afm_proc"):
            return

        params = {
            "task":          "refit_angles",
            "distances_nm":  last["distances_nm"],
            "h_adj":         last["h_adj"],
            "n_fit_points":  nfp,
            "threshold_fraction": self._safe_float(
                self.afm_threshold_frac, "Threshold fraction", 0.05
            ),
            "profile_idx":   idx,
        }

        self.afm_queue        = multiprocessing.Queue()
        self.afm_cancel_event = multiprocessing.Event()

        self.afm_proc = multiprocessing.Process(
            target=self._afm_worker_main,
            args=(params, self.afm_queue, self.afm_cancel_event),
            daemon=False,
        )
        self.afm_proc.start()
        self._poll_afm_queue()

    # 
    # Drawing helpers
    # 

    def _afm_draw_empty(self):
        self.afm_fig.clear()
        gs = gridspec.GridSpec(
            1, 2, figure=self.afm_fig,
            width_ratios=[1, 1], wspace=0.3
        )
        self._afm_ax_image   = self.afm_fig.add_subplot(gs[0, 0])
        self._afm_ax_profile = self.afm_fig.add_subplot(gs[0, 1])

        self._afm_ax_image.set_title("AFM height image")
        self._afm_ax_image.text(
            0.5, 0.5, "Load a JPK file to begin",
            ha="center", va="center",
            transform=self._afm_ax_image.transAxes,
            color="gray", fontsize=11
        )

        self._afm_ax_profile.set_title("Line profile")
        self._afm_ax_profile.text(
            0.5, 0.5,
            "Click two points on the\nAFM image to draw a profile",
            ha="center", va="center",
            transform=self._afm_ax_profile.transAxes,
            color="gray", fontsize=11
        )
        self.afm_canvas.draw_idle()

    def _afm_draw_image(self):
        """Redraw only the AFM image panel — removes old colorbar first."""
        ax = self._afm_ax_image

        # ── remove any existing colorbar axes ──────────────────
        # matplotlib attaches the colorbar to a stored attribute
        # if we set it ourselves; remove it before clearing
        if hasattr(self, "_afm_colorbar") and self._afm_colorbar is not None:
            try:
                self._afm_colorbar.remove()
            except Exception:
                pass
            self._afm_colorbar = None

        ax.cla()

        im = ax.imshow(
            self._afm_height_nm,
            cmap="afmhot", aspect="equal",
            extent=[
                0, self._afm_scan_size_um,
                self._afm_scan_size_um, 0
            ],
            interpolation="bilinear",
        )

        # store the colorbar so we can remove it next time
        self._afm_colorbar = self.afm_fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04, label="Height (nm)"
        )

        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title("AFM height image — click to draw profile")

        # draw all existing profile lines
        s  = self._afm_scan_size_um
        ny, nx = self._afm_height_nm.shape

        for i, prof in enumerate(self._afm_profiles):
            x0, y0 = prof["start_px"]
            x1, y1 = prof["end_px"]
            ax.plot(
                [x0 * s / nx, x1 * s / nx],
                [y0 * s / ny, y1 * s / ny],
                "c-", linewidth=1.8, zorder=5
            )
            mx = (x0 + x1) / 2 * s / nx
            my = (y0 + y1) / 2 * s / ny
            ax.text(
                mx, my, str(i + 1),
                color="cyan", fontsize=8,
                ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="black", alpha=0.5)
            )

        self.afm_canvas.draw_idle()

    def _afm_draw_profile(self, prof: dict):
        """Draw the profile panel for one result dict."""
        ax = self._afm_ax_profile
        ax.cla()

        dist     = np.asarray(prof["distances_nm"])
        raw      = np.asarray(prof["heights_raw"])
        baseline = np.asarray(prof["baseline"])
        h_adj    = np.asarray(prof["h_adj"])
        angles   = prof.get("angles")
        nfp      = prof.get("n_fit_points", 15)
        n        = len(dist)

        # raw
        ax.plot(dist, raw,
                color="steelblue", linewidth=1.5, alpha=0.4,
                label="Raw")

        # baseline
        ax.plot(dist, baseline,
                color="saddlebrown", linewidth=2, linestyle="--",
                label=(f"Baseline  "
                       f"({prof['h_start_nm']:.1f}→"
                       f"{prof['h_end_nm']:.1f} nm)"))

        # corrected
        ax.plot(dist, h_adj,
                color="royalblue", linewidth=2,
                label="Corrected (membrane = 0)")
        ax.fill_between(dist, 0, h_adj,
                        where=(h_adj > 0),
                        alpha=0.15, color="royalblue")

        # membrane reference
        ax.axhline(0, color="saddlebrown", linewidth=1,
                   linestyle="-", alpha=0.4)

        # contact points and tangent lines
        if angles is not None:
            for side, ck, sk, tk, color, direction in [
                ("left",  "left_contact",  "left_slope",
                 "theta_left_deg",  "tomato",         +1),
                ("right", "right_contact", "right_slope",
                 "theta_right_deg", "mediumseagreen", -1),
            ]:
                if ck not in angles or sk not in angles:
                    continue

                ci    = angles[ck]
                x_c   = float(dist[ci])
                slope = float(angles[sk])
                theta = angles.get(tk, float("nan"))

                # contact marker at membrane level
                ax.plot(x_c, 0, "o",
                        color=color, markersize=10, zorder=6,
                        label=f"{side.capitalize()} θ = {theta:.1f}°")

                # points used for slope
                fit_idx = np.arange(
                    ci, ci + direction * nfp, direction
                )
                fit_idx = fit_idx[
                    (fit_idx >= 0) & (fit_idx < n)
                ]
                ax.plot(dist[fit_idx], h_adj[fit_idx],
                        "o", color=color, markersize=5,
                        alpha=0.9, zorder=5)

                # tangent line
                span  = float(dist[-1]) * 0.25
                x_ext = np.array([x_c - span, x_c + span])
                y_ext = slope * (x_ext - x_c)
                ax.plot(x_ext, y_ext, "--",
                        color=color, linewidth=2)

            tl = angles.get("theta_left_deg",  float("nan"))
            tr = angles.get("theta_right_deg", float("nan"))
            tm = angles.get("theta_mean_deg",  float("nan"))
            title = (f"θ_left={tl:.1f}°  |  "
                     f"θ_right={tr:.1f}°  |  "
                     f"θ_mean={tm:.1f}°")
        else:
            title = (
                f"Contact points not found — "
                f"ensure line crosses condensate fully  "
                f"(peak = {h_adj.max():.2f} nm)"
            )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Distance (nm)")
        ax.set_ylabel("Height (nm)")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        self.afm_canvas.draw_idle()

    # 
    # Listbox
    # 

    def _afm_rebuild_listbox(self):
        self.afm_profile_listbox.delete(0, tk.END)
        for i, prof in enumerate(self._afm_profiles):
            m = prof.get("measurements", {})
            angles = prof.get("angles") or {}
            th_m = angles.get("theta_mean_deg", float("nan"))
            ph   = m.get("peak_height_nm", float("nan"))
            fw   = m.get("fwhm_nm",        float("nan"))
            self.afm_profile_listbox.insert(
                tk.END,
                f"#{i+1}  h={ph:.1f}nm  "
                f"FWHM={fw:.0f}nm  "
                f"θ={th_m:.1f}°"
            )

    def _afm_on_listbox_select(self, _event):
        sel = self.afm_profile_listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < len(self._afm_profiles):
            self._afm_draw_profile(self._afm_profiles[idx])

    # 
    # Button callbacks
    # 

    def _afm_reset_last(self):
        if self._afm_profiles:
            self._afm_profiles.pop()
        self._afm_click_points = []
        self._afm_rebuild_listbox()
        self._afm_draw_image()
        if self._afm_profiles:
            self._afm_draw_profile(self._afm_profiles[-1])
        else:
            self._afm_ax_profile.cla()
            self.afm_canvas.draw_idle()
        if not self._afm_profiles:
            self.afm_reset_btn.configure(state="disabled")
            self.afm_clear_btn.configure(state="disabled")
            self.afm_save_btn.configure(state="disabled")

    def _afm_clear_all(self):
        self._afm_profiles.clear()
        self._afm_click_points = []
        self._afm_rebuild_listbox()
        self._afm_draw_image()
        self._afm_ax_profile.cla()
        self.afm_canvas.draw_idle()
        self.afm_reset_btn.configure(state="disabled")
        self.afm_clear_btn.configure(state="disabled")
        self.afm_save_btn.configure(state="disabled")

    def _afm_save_results(self):
        if not self._afm_profiles:
            self._showwarning("Warning", "No profiles to save.")
            return

        fp = self.afm_filepath.get().strip()
        default_dir  = str(Path(fp).parent) if fp else "."
        default_stem = Path(fp).stem        if fp else "afm"

        save_dir = self._ask_directory(
            title="Select folder to save AFM results",
            initialdir=default_dir,
        )
        if not save_dir:
            return

        import pandas as pd
        rows = []
        for i, prof in enumerate(self._afm_profiles):
            m      = prof.get("measurements", {})
            angles = prof.get("angles") or {}
            row    = {"Profile": i + 1}
            row.update(m)
            row["start_px_x"] = prof["start_px"][0]
            row["start_px_y"] = prof["start_px"][1]
            row["end_px_x"]   = prof["end_px"][0]
            row["end_px_y"]   = prof["end_px"][1]
            rows.append(row)

            # per-profile CSV with full arrays
            prof_csv = os.path.join(
                save_dir, f"{default_stem}_profile{i+1}.csv"
            )
            pd.DataFrame({
                "distance_nm": prof["distances_nm"],
                "height_raw_nm": prof["heights_raw"],
                "height_smoothed_nm": prof["h_smooth"],
                "baseline_nm": prof["baseline"],
                "height_corrected_nm": prof["h_adj"],
            }).to_csv(prof_csv, index=False)

        # summary CSV
        summary_csv = os.path.join(
            save_dir, f"{default_stem}_AFM_summary.csv"
        )
        pd.DataFrame(rows).to_csv(summary_csv, index=False)

        # save current figure as SVG
        svg_path = os.path.join(
            save_dir, f"{default_stem}_AFM_overview.svg"
        )
        self.afm_fig.savefig(
            svg_path, dpi=300,
            bbox_inches="tight", facecolor="white"
        )

        self.log_message(
            f"AFM results saved to {save_dir}  "
            f"({len(self._afm_profiles)} profiles)"
        )
        self.log_message(f"  Summary: {summary_csv}")
        self.log_message(f"  Figure:  {svg_path}")

    # 
    # Queue polling
    # 

    def _poll_afm_queue(self):
        try:
            while True:
                msg_type, payload = self.afm_queue.get_nowait()

                if msg_type == "loaded":
                    self._afm_height_nm    = payload["height_nm"]
                    self._afm_pixel_size   = payload["pixel_size_nm"]
                    info = payload["info"]
                    self._afm_scan_size_um = info["scan_size_um"]

                    self.log_message(
                        f"AFM loaded: {payload['filepath']}"
                    )
                    self.log_message(
                        f"  Shape={info['shape']}  "
                        f"px={info['pixel_size_nm']:.3f} nm  "
                        f"scan={info['scan_size_um']:.3f} µm"
                    )
                    self.log_message(
                        f"  Heights: "
                        f"{info['height_min_nm']:.2f} – "
                        f"{info['height_max_nm']:.2f} nm"
                    )

                    # rebuild the two-panel figure
                    self.afm_fig.clear()
                    gs = gridspec.GridSpec(
                        1, 2, figure=self.afm_fig,
                        width_ratios=[1, 1], wspace=0.3
                    )
                    self._afm_ax_image   = self.afm_fig.add_subplot(
                        gs[0, 0]
                    )
                    self._afm_ax_profile = self.afm_fig.add_subplot(
                        gs[0, 1]
                    )
                    self._afm_draw_image()

                    self.status_var.set("Ready — click image to draw profile")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return

                elif msg_type == "profile":
                    self._afm_profiles.append(payload)
                    self._afm_rebuild_listbox()
                    self._afm_draw_image()
                    self._afm_draw_profile(payload)

                    m = payload.get("measurements", {})
                    self.log_message(
                        f"Profile #{len(self._afm_profiles)}  "
                        f"h={m.get('peak_height_nm', float('nan')):.2f} nm  "
                        f"FWHM={m.get('fwhm_nm', float('nan')):.1f} nm  "
                        f"θ_mean={m.get('theta_mean_deg', float('nan')):.1f}°"
                    )

                    self.afm_reset_btn.configure(state="normal")
                    self.afm_clear_btn.configure(state="normal")
                    self.afm_save_btn.configure(state="normal")

                    self.status_var.set("Ready")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return

                elif msg_type == "refit_done":
                    idx  = payload.get("profile_idx", -1)
                    if 0 <= idx < len(self._afm_profiles):
                        self._afm_profiles[idx]["angles"]       = (
                            payload["angles"]
                        )
                        self._afm_profiles[idx]["measurements"] = (
                            payload["measurements"]
                        )
                        self._afm_profiles[idx]["n_fit_points"] = (
                            payload["n_fit_points"]
                        )
                        self._afm_rebuild_listbox()
                        self._afm_draw_profile(self._afm_profiles[idx])

                    self.status_var.set("Ready")
                    self.set_ui_busy(False)
                    return

                elif msg_type == "cancelled":
                    self.log_message("AFM cancelled.")
                    self.status_var.set("Cancelled")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    return

                elif msg_type == "error":
                    self.log_message(payload)
                    self.status_var.set("Error")
                    self.progress_bar.grid_remove()
                    self.set_ui_busy(False)
                    self._showerror(
                        "AFM Error", "AFM processing failed. See log."
                    )
                    return

        except queue.Empty:
            pass

        if (self.afm_proc is not None
                and not self.afm_proc.is_alive()):
            self.set_ui_busy(False)
            self.status_var.set("Error")
            self.progress_bar.grid_remove()
            self.log_message("AFM worker terminated unexpectedly.")
            return

        self.root.after(50, self._poll_afm_queue)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------`Results tab GUI-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- 



    def save_results(self):
        """Save analysis results"""
        filename = self._ask_saveas_filename(
            title="Save results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.results_text.get(1.0, tk.END))
            self.log_message(f"Results saved to {filename}")

    def save_session(self):
        """Save current session parameters (only for tabs currently enabled
        in this window -- see ModularRICSGUI.enabled_tabs)."""
        filename = self._ask_saveas_filename(
            title="Save session",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                session_data = {}

                if hasattr(self, "img_width"):
                    session_data['simulation_params'] = {
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
                    }

                if hasattr(self, "input_file"):
                    session_data['export_params'] = {
                        'input_file': self.input_file.get(),
                        'channel': self.channel.get(),
                        'crop_factor': self.crop_factor.get(),
                        'window_size': self.window_size.get(),
                        'correct_drift': self.correct_drift.get()
                    }

                if hasattr(self, "rics_file"):
                    session_data['fitting_params'] = {
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

                with open(filename, 'w') as f:
                    json.dump(session_data, f, indent=2)

                self.log_message(f"Session saved to {filename}")

            except Exception as e:
                self._showerror("Error", f"Could not save session: {str(e)}")

    def load_session(self):
        """Load session parameters"""
        filename = self._ask_open_filename(
            title="Load session",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    session_data = json.load(f)

                # Load simulation parameters
                if 'simulation_params' in session_data and hasattr(self, "img_width"):
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
                if 'export_params' in session_data and hasattr(self, "input_file"):
                    export_params = session_data['export_params']
                    self.input_file.set(export_params.get('input_file', ''))
                    self.channel.set(export_params.get('channel', '0'))
                    self.crop_factor.set(export_params.get('crop_factor', '0.5'))
                    self.window_size.set(export_params.get('window_size', '3'))
                    self.correct_drift.set(export_params.get('correct_drift', False))

                # Load fitting parameters
                if 'fitting_params' in session_data and hasattr(self, "rics_file"):
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
                self._showerror("Error", f"Could not load session: {str(e)}")

    def export_plots(self):
        """Export all plots"""
        directory = self._ask_directory(title="Select directory for plot export")
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
                self._showerror("Error", f"Could not export plots: {str(e)}")

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
            "ics_cancel_event",
            "afm_cancel_event",
            "ptu_fcs_cancel_event",

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
        for proc_attr in ("sfcs_proc", "export_proc", "fit_proc", "sim_proc", "diffmap_proc", "fcsfit_proc", "frap_proc","vesicle_proc","ics_proc","afm_proc","ptu_fcs_proc",):
            _stop_proc(proc_attr)

        # 4) Close queues properly (prevents resource_tracker semaphore warnings)
        for qattr in ("sfcs_queue", "export_queue", "fit_queue", "sim_queue", "diffmap_queue", "fcsfit_queue", "frap_queue","vesicle_queue","ics_queue","afm_queue","ptu_fcs_queue",):
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
            "afm_cancel_event",
            "ptu_fcs_cancel_event",
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
        for proc_attr in ("sfcs_proc", "export_proc", "fit_proc", "sim_proc", "diffmap_proc", "fcsfit_proc", "frap_proc","vesicle_proc","ics_proc","afm_proc","ptu_fcs_proc",):
            p = getattr(self, proc_attr, None)
            try:
                if p is not None and p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
            except Exception:
                pass
            setattr(self, proc_attr, None)

        # close queues properly
        for q_attr in ("sfcs_queue", "export_queue", "fit_queue", "sim_queue", "diffmap_queue", "fcsfit_queue", "frap_queue","vesicle_queue","ics_queue", "afm_queue","ptu_fcs_queue",):
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
        if not self._askyesno(
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
        



    


