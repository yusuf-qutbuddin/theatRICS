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
import sv_ttk
from pathlib import Path
import platform

from pyglet.font import fontconfig
from theatrics.gui_app import ModularRICSGUI, CATEGORY_TABS

# ── Logo paths — one per theme ──────────────────────────────────────────────
# "light" -> shown while the app is in light mode
# "dark"  -> shown while the app is in dark mode
_LOGO_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGO_PATHS = {
    "light": os.path.join(_LOGO_DIR, "icons","logo_light.png"),
    "dark":  os.path.join(_LOGO_DIR, "icons","logo_dark.png"),
}
# ── ttk widget classes whose font we forcibly override to Poppins ──────────
# sv_ttk's theme sets its own font on each of these classes directly (more
# specific than the "." default style), so simply configuring "." or the
# named Tk fonts (TkDefaultFont etc.) is not enough to make ttk widgets
# actually use Poppins -- these per-class entries must be reconfigured too.
_FONT_STYLE_TARGETS_BODY = [
    "TLabel", "TButton", "TCheckbutton", "TRadiobutton", "TEntry",
    "TCombobox", "TMenubutton", "TNotebook", "TScale", "TProgressbar",
    "Treeview", "TScrollbar", "TSpinbox", "TPanedwindow",
]
_FONT_STYLE_TARGETS_HEADING = [
    "TNotebook.Tab", "Treeview.Heading", "TLabelframe.Label",
]

# ── Font loading ─────────────────────────────────────────────────────────────
def _load_fonts():
    """Load bundled fonts before the Tk window is created, and register
    them with both pyglet/Tk and matplotlib so every part of the
    application (widgets and plots alike) uses the same font files."""

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

    _load_matplotlib_fonts()

def _load_matplotlib_fonts():
    """
    Register every bundled font file (theatrics/fonts/*.ttf, *.otf) with
    Matplotlib's font manager, and set the bundled "Poppins" family as
    the default font for every matplotlib Figure/Axes created anywhere
    in the application -- including Figures embedded in the GUI via
    FigureCanvasTkAgg, and any figures saved directly to SVG/PNG.

    This mirrors _load_fonts() (which registers the same files with
    pyglet/tkinter) but targets matplotlib.font_manager instead. Safe to
    call multiple times -- addfont() on an already-registered file is a
    harmless no-op.
    """
    import matplotlib
    import matplotlib.font_manager as fm

    fonts_dir = Path(__file__).parent / "fonts"  # theatrics/fonts/
    registered_families = set()

    for ext in ["**/*.ttf", "**/*.TTF", "**/*.otf", "**/*.OTF"]:
        for font_file in fonts_dir.glob(ext):
            try:
                fm.fontManager.addfont(str(font_file))
                family_name = fm.FontProperties(fname=str(font_file)).get_name()
                registered_families.add(family_name)
            except Exception:
                pass

    if not registered_families:
        return

    # Prefer plain "Poppins" as the default body font if present; otherwise
    # fall back to any bundled family whose name contains "Poppins", or
    # failing that, whatever was actually registered.
    if "Poppins" in registered_families:
        default_family = "Poppins"
    else:
        default_family = next(
            (f for f in registered_families if "Poppins" in f),
            next(iter(registered_families)),
        )

    # Set as the literal default family (mirrors the user's original
    # snippet: plt.rcParams['font.family'] = font_name)...
    matplotlib.rcParams["font.family"] = default_family

    # ...and ALSO register it as the preferred sans-serif fallback, so any
    # plot code that explicitly requests family="sans-serif" (rather than
    # inheriting the rcParams default) still resolves to Poppins first.
    matplotlib.rcParams["font.sans-serif"] = [default_family] + list(
        matplotlib.rcParams.get("font.sans-serif", [])
    )
class TheatricsLauncher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("theatRICS")
        # self.root.geometry("600x600")
        self.root.resizable(False, False)

        self._set_dpi_scaling()
        # self._set_theme()
        self._set_default_fonts()
        self._apply_custom_fonts_to_style()

        # category_name -> currently open ModularRICSGUI instance
        self._open_apps: dict[str, ModularRICSGUI] = {}
        # category_name -> Toplevel window hosting that instance
        self._open_windows: dict[str, tk.Toplevel] = {}

        # logo state (populated by _add_logo / _update_logo)
        self._logo_photo = None
        self._logo_label = None
        self._last_theme = self._current_theme()

        self._build_ui()

        # keep the logo + fonts in sync even if the theme is toggled from
        # a category window's own "Toggle Dark Mode" button (sv_ttk's
        # theme is global to the whole Tk interpreter, so we just poll
        # for a change rather than needing an explicit callback wired
        # up from every place that can call sv_ttk.toggle_theme()).
        self._poll_theme_change()

    def _set_theme(self):
        """Set ttk theme based on OS."""
        style = ttk.Style()

        if platform.system() == "Linux":
            style.theme_use("clam")

    # ── DPI handling ──────────────────────────────────────────────────────
    def _compute_scaling_from_mm(self) -> float:
        """
        Compute a Tk scaling factor from the reported screen width in
        pixels vs. millimetres, relative to a 96 DPI baseline.

        Guards against bogus/zero millimetre reports (fairly common on
        some Linux setups -- virtual displays, VNC, certain multi-monitor
        configurations -- where winfo_screenmmwidth() can return 0 or an
        implausible value) by falling back to 1.0 (no extra scaling)
        rather than raising a ZeroDivisionError or applying a nonsense
        scale factor.
        """
        try:
            screen_width_px = self.root.winfo_screenwidth()
            screen_width_mm = self.root.winfo_screenmmwidth()
            if screen_width_mm <= 0:
                return 1.0
            screen_dpi = screen_width_px / (screen_width_mm / 25.4)
            return screen_dpi / 96.0
        except Exception:
            return 1.0

    def _set_dpi_scaling(self):
        """
        Auto-detect screen DPI and scale the Tk UI accordingly, across
        Windows, Linux, and macOS.
        """
        system = platform.system()

        if system == "Windows":
            
            # # Enable per-monitor DPI awareness first so Windows doesn't
            # # apply its own bitmap-stretching scaling on top of ours
            # # (which would make everything blurry).
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass
            scaling = self._compute_scaling_from_mm()

        elif system == "Darwin":
            # macOS: Tk built against the Cocoa framework is already
            # HiDPI/Retina-aware out of the box and presents geometry in
            # device-independent points rather than raw physical pixels.
            # Forcibly overriding `tk scaling` here would double-apply
            # the OS's own Retina scaling (making everything oversized),
            # so we deliberately leave Tk's own default scaling alone.
            return

        else:
            # Linux / other X11 / Wayland desktops. The X server's
            # reported physical monitor size (used by the mm-based
            # calculation) is frequently wrong or missing, so prefer an
            # explicit scale-factor environment variable set by the
            # desktop environment when one is present.
            scaling = None
            for env_var in ("GDK_SCALE", "QT_SCALE_FACTOR", "GDK_DPI_SCALE"):
                val = os.environ.get(env_var)
                if val:
                    try:
                        scaling = float(val)
                        break
                    except ValueError:
                        pass

            if scaling is None:
                scaling = self._compute_scaling_from_mm()

        # clamp between sensible values so nothing goes crazy
        scaling = max(1.0, min(scaling, 2.5))

        self.root.tk.call('tk', 'scaling', scaling)
        # print(f"🖥️  OS: {system} | Scaling: {scaling:.2f}")

    def _set_default_fonts(self):
        """Override tkinter's default fonts with Poppins globally."""
        defaults = {
            "TkDefaultFont":   ("Poppins SemiBold", 18),
            "TkTextFont":      ("Poppins", 18),
            "TkFixedFont":     ("Poppins SemiBold", 18),        # monospace elements
            "TkMenuFont":      ("Poppins SemiBold", 18),
            "TkHeadingFont":   ("Poppins Black", 20),
            "TkCaptionFont":   ("Poppins SemiBold", 20),
            "TkSmallCaptionFont": ("Poppins SemiBold", 18),
            "TkIconFont":      ("Poppins SemiBold", 20),
            "TkTooltipFont":   ("Poppins SemiBold", 18),
        }

        for font_name, spec in defaults.items():
            f = tk.font.nametofont(font_name)
            f.configure(family=spec[0], size=spec[1])
            if len(spec) > 2:
                f.configure(weight=spec[2])

    def _apply_custom_fonts_to_style(self):
        """
        Force every common ttk widget class to use our bundled Poppins
        fonts instead of whatever font sv_ttk's theme bakes in.

        sv_ttk.set_theme()/toggle_theme() fully (re)applies its own ttk
        style, including its own default font per widget class. Simply
        setting the named Tk fonts (TkDefaultFont etc.) or configuring
        the "." (default) style is NOT enough to override this, since
        sv_ttk's per-widget-class style entries (e.g. "TButton",
        "TLabel", ...) are more specific and take precedence over ".".

        This must therefore be called again any time the sv_ttk theme is
        (re)applied -- see _on_theme_changed() -- not just once at
        startup.
        """
        style = ttk.Style(self.root)

        body_font    = ("Poppins", 18)
        heading_font = ("Poppins SemiBold", 18)

        try:
            style.configure(".", font=body_font)
        except tk.TclError:
            pass

        for widget_class in _FONT_STYLE_TARGETS_BODY:
            try:
                style.configure(widget_class, font=body_font)
            except tk.TclError:
                pass

        for widget_class in _FONT_STYLE_TARGETS_HEADING:
            try:
                style.configure(widget_class, font=heading_font)
            except tk.TclError:
                pass

    # ── theme helpers ─────────────────────────────────────────────────────
    def _current_theme(self) -> str:
        """Return 'light' or 'dark' based on sv_ttk's current theme."""
        try:
            theme = sv_ttk.get_theme()
            if theme in _LOGO_PATHS:
                return theme
        except Exception:
            pass
        return "light"

    def _theme_background_rgb(self) -> tuple[int, int, int, int]:
        """
        Return the CURRENT theme's background colour as an 8-bit RGBA
        tuple, read from the ttk Style (not from root.cget("background")).

        root.cget("background") can lag behind sv_ttk.toggle_theme() --
        sv_ttk updates the ttk Style synchronously, but some of its own
        window-level fixups (e.g. Windows dark title bar) are scheduled
        via root.after(...) and the root widget's own -background option
        isn't guaranteed to reflect the new theme the instant
        toggle_theme() returns. Reading straight from the Style avoids
        that race entirely.
        """
        try:
            style = ttk.Style(self.root)
            bg_name = style.lookup("TFrame", "background")
            if bg_name:
                r16, g16, b16 = self.root.winfo_rgb(bg_name)
                return (r16 // 257, g16 // 257, b16 // 257, 255)
        except Exception:
            pass

        # fallback: raw root background (may lag by one theme toggle)
        r16, g16, b16 = self.root.winfo_rgb(self.root.cget("background"))
        return (r16 // 257, g16 // 257, b16 // 257, 255)

    def _on_theme_changed(self):
        """
        Run everything that needs to be refreshed whenever the global
        sv_ttk theme changes: the logo (light/dark asset + recomposited
        background) and the ttk style fonts (sv_ttk re-applies its own
        fonts every time its theme is (re)applied, clobbering our
        Poppins override, so it must be reapplied here too).
        """
        self._apply_custom_fonts_to_style()
        self._update_logo()

    def _toggle_theme(self):
        """Flip the global sv_ttk theme and refresh the logo/fonts to match."""
        try:
            sv_ttk.toggle_theme()
        except Exception as exc:
            print(f"[theatRICS launcher] theme toggle failed: {exc}")

        self._last_theme = self._current_theme()

        # Defer the refresh by one idle cycle so any of sv_ttk's own
        # internal after()-scheduled fixups run first, and so the ttk
        # Style has fully settled before we read its background colour
        # / reapply our font overrides.
        self.root.after_idle(self._on_theme_changed)

    def _poll_theme_change(self):
        """
        Periodically check whether the global theme has changed (e.g.
        toggled from inside a category window) and refresh the logo/fonts
        if so.
        """
        theme = self._current_theme()
        if theme != self._last_theme:
            self._last_theme = theme
            self.root.after_idle(self._on_theme_changed)
        self.root.after(400, self._poll_theme_change)

    # ── UI construction ──────────────────────────────────────────────────
    def _build_ui(self):

        # top_bar = ttk.Frame(self.root)
        # top_bar.pack(fill="x", padx=10, pady=(6, 0))

        # toggle_btn = ttk.Button(
        #     top_bar, text="Toggle Dark Mode", command=self._toggle_theme
        # )
        # toggle_btn.pack(side=tk.RIGHT)

        # ── main two-column layout: logo on the left, categories on the
        # right, so the window feels visually balanced ─────────────────
        container = ttk.Frame(self.root)
        container.pack(expand=True, fill="both", padx=30, pady=20)

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)

        left_frame = ttk.Frame(container)
        left_frame.grid(row=0, column=0, sticky="nsew")

        right_frame = ttk.Frame(container)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(25, 0))

        # thin vertical separator between the two halves
        # sep_frame = ttk.Frame(container, width=1)
        # sep_frame.grid(row=0, column=0, sticky="nse")
        # ttk.Separator(sep_frame, orient="vertical").pack(fill="y", expand=True)

        self._add_logo(left_frame)

        subheader = ttk.Label(
            right_frame,
            text="Select an analysis category to begin",
            foreground="gray",
        )
        subheader.pack(pady=(0, 20))

        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(expand=True, fill="both")

        categories = [
            (
                "Correlation Methods",
                "Image Simulation, RICS Export, RICS Fitting, \n SFCS, FCS Export, FCS Fitting",
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

    def _add_logo(self, parent):
        """Create the logo label and populate it with the theme-correct image."""
        self._logo_label = tk.Label(parent, bd=0)
        # expand=True with no fill centers the label both horizontally
        # and vertically within the available space in `parent`, which
        # keeps it visually balanced against the button column on the
        # right regardless of exact window height.
        self._logo_label.pack(expand=True)
        self._update_logo()

    def _update_logo(self):
        """(Re)build the logo image to match the current theme and redisplay it."""
        theme = self._current_theme()
        logo_path = _LOGO_PATHS.get(theme, _LOGO_PATHS["light"])

        if self._logo_label is None or not os.path.isfile(logo_path):
            return

        try:
            from PIL import Image, ImageTk

            target_width = 100

            img   = Image.open(logo_path).convert("RGBA")
            w, h  = img.size
            new_h = max(1, int(h * target_width / w))
            img   = img.resize((target_width, new_h), Image.LANCZOS)

            # Read the CURRENT theme's background straight from the ttk
            # Style rather than root.cget("background") -- see
            # _theme_background_rgb() for why.
            bg_rgb = self._theme_background_rgb()

            # composite the RGBA logo onto the exact window background
            bg = Image.new("RGBA", img.size, bg_rgb)
            bg.paste(img, mask=img.split()[3])
            img = bg.convert("RGB")

            self._logo_photo = ImageTk.PhotoImage(img)
            self._logo_label.configure(image=self._logo_photo)

        except Exception as exc:
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