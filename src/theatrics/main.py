import multiprocessing as mp
import tkinter as tk
# from theatrics.launcher import TheatricsLauncher
from theatrics.utils.mp_utils import set_single_threaded_blas, setup_multiprocessing



def main():
    mp.freeze_support()  # MUST be first thing for Windows
    
    set_single_threaded_blas()
    setup_multiprocessing("spawn", force=True)
    root = tk.Tk()
    # ── splash screen ─────────────────────────────────────────────────────
    from theatrics.splash import SplashScreen
    splash = SplashScreen(root)
    root.update()
    
    splash.set_status("Loading launcher...")
    from theatrics.launcher import TheatricsLauncher
    import theatrics.launcher
    splash.set_status("Loading fonts...")
    theatrics.launcher._load_fonts()
    splash.set_status("Loading Python modules...")

    import os
    import sys
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
    import multiprocessing
    import queue
    import tifffile
    import scipy.ndimage
    import json
    import pandas as pd
    import platform
    splash.set_status("Loading modules...")

    try:
        import tttrlib
    except ImportError:
        pass

    try:
        from AFMReader.jpk import load_jpk
    except ImportError:
        pass
    from scipy import ndimage
    from scipy.signal import savgol_filter
    splash.set_status("Loading theatrics...")

    from theatrics.utils.file_utils import get_files_from_folder
    from theatrics.utils.mp_utils import clamp_workers
    
    splash.set_status("Building interface...")

    # ── build launcher, then dismiss splash ───────────────────────────────
    launcher = TheatricsLauncher(root)
    root.protocol("WM_DELETE_WINDOW", launcher.on_launcher_close)
    splash.set_status("Ready")
    root.after(350, splash.dismiss)

    root.mainloop()
    
    


if __name__ == "__main__":
    main()