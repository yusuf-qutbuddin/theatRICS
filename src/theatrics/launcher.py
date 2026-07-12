"""
theatrics/launcher.py

Small modular launcher window shown on startup. Presents one large
button per analysis category (see gui_app.CATEGORY_TABS for exactly
which tabs belong to each category). Clicking a button opens a
Toplevel window containing only that category's tabs. Only one window
per category may be open at a time -- clicking the button again while
that category's window is already open simply brings it to the front
instead of opening a duplicate.

The launcher window itself stays open regardless of how many category
windows are opened, and the whole application shuts down (including
any running background worker processes in every open category window)
only when the launcher window itself is closed.
"""

from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk
import pyglet
from pathlib import Path
import platform

from pyglet.font import fontconfig
from theatrics.gui_app import ModularRICSGUI, CATEGORY_TABS

# Path to logo — same file used by the splash screen
_LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logo.png"
)

# ── Font loading ─────────────────────────────────────────────────────────────
def _load_fonts():
    """Load bundled fonts before the Tk window is created."""

    # Patch pyglet's weight table to handle unusual font weights
    for w in [47.5, 47, 48, 55.5, 55, 56, 72.5, 87.5]:
        if w not in fontconfig.weight_to_name:
            fontconfig.weight_to_name[w] = "normal"

    fonts_dir = Path(__file__).parent / "fonts"  # theatrics/fonts/

    for ext in ["**/*.ttf", "**/*.TTF", "**/*.otf", "**/*.OTF"]:
        for font_file in fonts_dir.glob(ext):
            try:
                pyglet.font.add_file(str(font_file))
            except KeyError as e:
                # catch any remaining unknown weights and retry
                fontconfig.weight_to_name[float(str(e).strip("'"))] = "normal"
                try:
                    pyglet.font.add_file(str(font_file))
                except Exception:
                    pass
            except Exception:
                pass


class TheatricsLauncher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("theatRICS")
        self.root.geometry("1400x900")
        self.root.resizable(False, False)
        self._set_dpi_scaling()
        self._set_theme()    
        self._set_default_fonts()
        # category_name -> currently open ModularRICSGUI instance
        self._open_apps: dict[str, ModularRICSGUI] = {}
        # category_name -> Toplevel window hosting that instance
        self._open_windows: dict[str, tk.Toplevel] = {}

        self._build_ui()
    def _set_theme(self):
        """Set ttk theme based on OS."""
        style = ttk.Style()
        
        if platform.system() == "Linux":
            style.theme_use("clam")
    def _set_dpi_scaling(self):
        """Auto-detect screen DPI and scale accordingly."""
        system = platform.system()

        # Windows: enable DPI awareness first
        if system == "Windows":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass

        # get actual screen DPI from tkinter
        screen_width_px  = self.root.winfo_screenwidth()
        screen_width_mm  = self.root.winfo_screenmmwidth()
        screen_dpi       = screen_width_px / (screen_width_mm / 25.4)

        # calculate scaling factor relative to 96 DPI baseline
        scaling = screen_dpi / 96.0

        # clamp between sensible values so nothing goes crazy
        scaling = max(1.0, min(scaling, 2.5))

        self.root.tk.call('tk', 'scaling', scaling)
        # print(f"🖥️  OS: {system} | DPI: {screen_dpi:.1f} | Scaling: {scaling:.2f}")
    def _set_default_fonts(self):
        """Override tkinter's default fonts with Poppins globally."""
        defaults = {
            "TkDefaultFont":   ("Poppins SemiBold", 14),
            "TkTextFont":      ("Poppins", 14),
            "TkFixedFont":     ("Poppins SemiBold", 14),        # monospace elements
            "TkMenuFont":      ("Poppins SemiBold", 14),
            "TkHeadingFont":   ("Poppins Black", 16),
            "TkCaptionFont":   ("Poppins SemiBold", 16),
            "TkSmallCaptionFont": ("Poppins SemiBold", 14),
            "TkIconFont":      ("Poppins SemiBold", 16),
            "TkTooltipFont":   ("Poppins SemiBold", 14),
        }

        for font_name, spec in defaults.items():
            f = tk.font.nametofont(font_name)
            f.configure(family=spec[0], size=spec[1])
            if len(spec) > 2:
                f.configure(weight=spec[2])

        # # Also set it as the default for ttk widgets
        # style = ttk.Style()
        # style.configure(".", font=("Poppins", 20))
    # ── UI construction ──────────────────────────────────────────────────
    def _build_ui(self):


        # header = ttk.Label(
        #     self.root, text="theatRICS", 
        # )
        # header.pack(pady=(24, 4))
        self._add_logo()
        subheader = ttk.Label(
            self.root,
            text="Select an analysis category to begin",
            
            foreground="gray",
        )
        
        subheader.pack(pady=(0, 20))

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(expand=True, fill="both", padx=30, pady=10)

        categories = [
            (
                "Correlation Methods",
                "Image Simulation, RICS Export, RICS Fitting, SFCS, FCS Export, FCS Fitting",
            ),
            (
                "AFM",
                "AFM height profile analysis",
            ),
            (
                "Imaging Methods",
                "ICS, Vesicle Finder, FRAP",
            ),
        ]

        for name, description in categories:
            self._make_category_button(btn_frame, name, description)

    def _add_logo(self):
        if not os.path.isfile(_LOGO_PATH):
            return
        try:
            from PIL import Image, ImageTk

            target_width = 260

            img   = Image.open(_LOGO_PATH).convert("RGBA")
            w, h  = img.size
            new_h = max(1, int(h * target_width / w))
            img   = img.resize((target_width, new_h), Image.LANCZOS)

            # composite onto white — the launcher background is white in
            # both light and dark sv_ttk (the launcher is small and plain).
            # Using a hardcoded white is more reliable than reading
            # self.root.cget("background"), which can return internal Tk
            # colour names that PIL cannot parse.
            bg = Image.new("RGBA", img.size, (28, 28, 28, 255))
            bg.paste(img, mask=img.split()[3])   # alpha channel as mask
            img = bg.convert("RGB")

            self._photo = ImageTk.PhotoImage(img)

            tk.Label(
                self.root,
                image=self._photo,
                bd=0,
            ).pack(pady=(20, 4))

        except Exception as exc:
            # print so silent failures are at least visible in the terminal
            print(f"[theatRICS launcher] logo load failed: {exc}")



    def _make_category_button(self, parent, category_name, description):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=10)

        btn = tk.Button(
            frame,
            text=category_name,
            
            height=2,
            command=lambda: self._open_category(category_name),
        )
        btn.pack(fill="x")

        label = ttk.Label(
            frame, text=description, foreground="gray",
            
        )
        
        label.pack(fill="x", pady=(3, 0))

    # ── category window management ───────────────────────────────────────
    def _open_category(self, category_name: str):
        # if already open, just bring it to the front instead of
        # creating a duplicate window
        existing_win = self._open_windows.get(category_name)
        if existing_win is not None:
            try:
                if existing_win.winfo_exists():
                    existing_win.deiconify()
                    existing_win.lift()
                    existing_win.focus_force()
                    return
            except tk.TclError:
                pass
            # stale reference (window destroyed without going through our
            # own close handler) -- clear it and fall through to recreate
            self._open_windows.pop(category_name, None)
            self._open_apps.pop(category_name, None)

        win = tk.Toplevel(self.root)
        win.geometry("1400x900")

        app = ModularRICSGUI(
            win,
            tabs=CATEGORY_TABS[category_name],
            window_title=f"theatRICS — {category_name} ",
        )

        self._open_windows[category_name] = win
        self._open_apps[category_name] = app

        def _on_category_close():
            # NOTE: this intentionally replaces the WM_DELETE_WINDOW
            # handler that ModularRICSGUI.setup_gui() already registered
            # on this same window (self.root.protocol(...) -> app.on_close).
            # We perform equivalent cleanup here ourselves (cancel running
            # tasks, tear down worker processes) so we can ALSO remove the
            # window from the launcher's open-window tracking dicts before
            # destroying it.
            try:
                app.cancel_current_task()
                app._cleanup_mp()
            except Exception:
                pass
            finally:
                self._open_windows.pop(category_name, None)
                self._open_apps.pop(category_name, None)
                win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_category_close)

    def on_launcher_close(self):
        """
        Close every open category window (cancelling their running tasks
        and cleaning up their worker processes) before shutting down the
        whole application.
        """
        for category_name, win in list(self._open_windows.items()):
            app = self._open_apps.get(category_name)
            try:
                if app is not None:
                    app.cancel_current_task()
                    app._cleanup_mp()
            except Exception:
                pass
            try:
                win.destroy()
            except Exception:
                pass

        self._open_windows.clear()
        self._open_apps.clear()
        self.root.destroy()