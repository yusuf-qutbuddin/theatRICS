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
    
    splash.set_status("Loading theatRICS…")
    from theatrics.utils.file_utils import get_files_from_folder
    from theatrics.utils.mp_utils import clamp_workers
    from theatrics.launcher import TheatricsLauncher

    # ── build launcher, then dismiss splash ───────────────────────────────
    launcher = TheatricsLauncher(root)
    root.protocol("WM_DELETE_WINDOW", launcher.on_launcher_close)
    sv_ttk.set_theme("light")

    splash.set_status("Ready.")
    root.after(350, splash.dismiss)

    root.mainloop()
    
    


if __name__ == "__main__":
    main()