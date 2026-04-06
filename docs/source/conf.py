# Configuration file for the Sphinx documentation builder.

from __future__ import annotations
import os

project = "theatRICS"
author = "Yusuf Qutbuddin"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"

# Headless-safe matplotlib backend (RTD is headless)
os.environ["MPLBACKEND"] = "Agg"

# If autodoc imports modules that require GUI/CZI libs, mock them here.
# Keep this minimal; add more only if RTD build complains.
autodoc_mock_imports = [
    "tkinter",
    "sv_ttk",
    "pylibCZIrw",
]

# MyST settings (optional)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]