"""
theatrics/splash.py

Minimal splash screen shown at startup.
Displays the theatRICS logo PNG centred on screen.
Requires Pillow (pip install Pillow).
"""

from __future__ import annotations

import os
import tkinter as tk
from PIL import Image, ImageTk


# Path to logo relative to this file
_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")


class SplashScreen:

    def __init__(self, root: tk.Tk, logo_width_px: int = 420):
        self._root      = root
        self._dismissed = False
        self._photo     = None   # prevent GC

        self._root.withdraw()

        self._win = tk.Toplevel(root)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        self._win.configure(background="white")

        # load and resize logo
        img    = Image.open(_LOGO_PATH)
        w, h   = img.size
        new_h  = max(1, int(h * logo_width_px / w))
        img    = img.resize((logo_width_px, new_h), Image.LANCZOS)

        # # handle transparency
        # if img.mode in ("RGBA", "LA", "P"):
        #     bg = Image.new("RGBA", img.size, "white")
        #     if img.mode == "P":
        #         img = img.convert("RGBA")
        #     bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        #     img = bg.convert("RGB")

        self._photo = ImageTk.PhotoImage(img)
        tk.Label(self._win, image=self._photo, bd=0).pack()

        self._win.update_idletasks()

        # centre on screen
        sw = self._win.winfo_screenwidth()
        sh = self._win.winfo_screenheight()
        ww = self._win.winfo_width()
        wh = self._win.winfo_height()
        self._win.geometry(f"+{(sw - ww) // 2}+{(sh - wh) // 2}")

    def dismiss(self):
        if self._dismissed:
            return
        self._dismissed = True
        try:
            self._win.destroy()
        except Exception:
            pass
        try:
            self._root.deiconify()
            self._root.lift()
            self._root.focus_force()
        except Exception:
            pass