"""
theatrics/splash.py

Minimal splash screen shown at startup.
Displays the theatRICS logo PNG stretched to fill the entire screen.
Requires Pillow (pip install Pillow).
"""

from __future__ import annotations

import os
import tkinter as tk
from PIL import Image, ImageTk


_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "theatrics_splash.png")


class SplashScreen:

    def __init__(self, root: tk.Tk, logo_width_px: int = 420):
        # logo_width_px kept for backwards-compatible signature but no
        # longer used -- the image now fills the whole screen instead of
        # being shown at a fixed width.
        self._root      = root
        self._dismissed = False
        self._photo     = None   # prevent GC

        self._root.withdraw()

        self._win = tk.Toplevel(root)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)

        sw = self._win.winfo_screenwidth()
        sh = self._win.winfo_screenheight()
        self._win.geometry(f"{sw}x{sh}+0+0")

        # BILINEAR instead of LANCZOS: much faster, plenty good enough
        # for a splash screen shown for well under a second.
        img = Image.open(_LOGO_PATH).convert("RGB")
        img = img.resize((sw, sh), Image.BILINEAR)

        self._photo = ImageTk.PhotoImage(img)
        label = tk.Label(self._win, image=self._photo, bd=0)
        label.place(x=0, y=0, width=sw, height=sh)

        # ── status text, pinned to bottom-center ────────────────────────
        self._status_var = tk.StringVar(value="")
        self._status_label = tk.Label(
            self._win,
            textvariable=self._status_var,
            font=("", 11),
            fg="white",
            bg="black",
            padx=12,
            pady=4,
        )
        # place near the bottom, horizontally centred
        self._status_label.place(relx=0.5, rely=0.96, anchor="s")
        # Force an immediate repaint so the splash is visible right away,
        # even if the caller doesn't call root.update() themselves.
        self._win.update_idletasks()
        self._win.update()

    def set_status(self, text: str):
        """
        Update the small status line at the bottom of the splash screen.
        Safe to call repeatedly from main() while heavy imports / setup
        happen, since it forces an immediate repaint (the main event loop
        isn't running yet at that point).
        """
        if self._dismissed:
            return
        try:
            self._status_var.set(text)
            self._win.update_idletasks()
            self._win.update()
        except Exception:
            pass
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