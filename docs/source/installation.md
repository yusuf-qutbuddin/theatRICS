# Installation

## Requirements

theatRICS requires:

- Python **3.10 or newer**
- a working **Tkinter** installation for the GUI
- standard scientific Python packages
- support for reading **CZI** files

Depending on your operating system, some dependencies may require system libraries.

---

## Install from PyPI

If the package is available on PyPI, install it with:

```bash
pip install theatrics
```

Then start the GUI with:

```bash
theatrics
```

---

## Install from source

Clone the repository and install locally:

```bash
git clone https://github.com/yusuf-qutbuddin/theatRICS.git
cd theatRICS
pip install -e .
```

Then run:

```bash
theatrics
```

---

## Core Python dependencies

The project uses the following scientific and GUI-related Python packages.

### Core analysis packages
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `tifffile`
- `scikit-image`
- `lmfit`

### GUI / usability
- `sv-ttk`

### Correlation and statistics
- `multipletau`
- `statsmodels`

### File I/O
- `pylibCZIrw`
- `openpyxl`

These are needed across the different workflows:
- RICS
- SFCS
- FCS fitting
- FRAP

---

## Optional dependencies by workflow

| Package | Workflow | Purpose |
|---------|----------|---------|
| `opencv-python` | Vesicle Finder | Faster Hough circle detection |
| `cellpose` | Vesicle Finder | Deep learning segmentation |
| `picasso` | ICS | Large multi-file TIFF stack loading |
| `AFMReader` | AFM | Loading JPK QI image files |
| `tttrlib` | RICS, Diffusion Map | Loading PicoQuant PTU CLSM image files |
---

## Example dependency installation

If you need to install dependencies manually in an existing environment:

```bash
pip install numpy scipy matplotlib pandas tifffile scikit-image lmfit multipletau statsmodels openpyxl sv-ttk pylibCZIrw
```

---

## Virtual environment (recommended)

It is recommended to use a dedicated virtual environment.

### Using `venv`
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

### Using conda
```bash
conda create -n theatrics python=3.11
conda activate theatrics
pip install -e .
```

---

## Tkinter notes

The GUI uses Tkinter. On many systems this is already included with Python, but on some Linux installations it may need to be installed separately.

For example, on Debian/Ubuntu systems:

```bash
sudo apt install python3-tk
```

If Tkinter is missing, the GUI may fail to start even if the package installation succeeds.

---

## CZI support

theatRICS uses:

- `pylibCZIrw`

to read Zeiss CZI files.

If CZI support is not available:
- RICS workflows on TIFF files may still work
- CZI-based workflows such as FRAP and some export pipelines may fail

If you frequently work with CZI files, verify that `pylibCZIrw` installs and imports correctly in your environment.

---

## Optional notes by workflow

### RICS and SFCS
Require the standard numerical/scientific stack:
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `tifffile`

### FCS fitting
Also requires:
- `lmfit`
- `statsmodels`

### FRAP
Also requires:
- `scikit-image`
- `openpyxl`
- `pylibCZIrw`

### Vesicle Finder
Requires the standard stack plus optionally:
- `opencv-python` for faster Hough circle detection
- `cellpose` for deep learning segmentation

### ICS
Requires the standard numerical stack plus optionally:
- `picasso` for loading large multi-file TIFF stacks (`TiffMultiMap`)

Install picasso:

```bash
pip install picasso
```

If picasso is not installed, the software falls back to `tifffile` for
standard TIFF files.

### AFM
Requires:
- `AFMReader` for loading JPK QI image files

Install AFMReader:

```bash
pip install AFMReader
```

Without AFMReader, JPK files cannot be loaded and the AFM tab will
raise an error on file load.

### PTU files (PicoQuant Luminosa)
PTU file support requires:
- `tttrlib`

Install:

```bash
pip install tttrlib
```

Without tttrlib, PTU files cannot be loaded. CZI and TIFF workflows
are unaffected.

### Install optional vesicle dependencies:

```bash
pip install opencv-python cellpose 
```
---

## Install documentation dependencies

If you want to build the documentation locally:

```bash
pip install -r docs/requirements.txt
```

or, if using the optional docs extra:

```bash
pip install -e ".[docs]"
```

Then build the docs with:

```bash
sphinx-build -b html docs/source docs/build/html
```

---

## Verify the installation

After installation, you can test the command-line entry point:

```bash
theatrics
```

If successful, the GUI should open.

---

## Common installation issues

### `theatrics` command not found
This usually means:
- the package was not installed in the active environment
- the wrong Python environment is active
- the console script location is not on your `PATH`

Try:

```bash
python -m pip install -e .
```

and make sure you run the command in the same environment.

### Tkinter import errors
Install system Tk support if necessary.

### `pylibCZIrw` installation problems
This may depend on the platform and available system libraries.
If CZI-based workflows fail, verify that this package is importable.

### Missing package during runtime
Install the missing dependency explicitly with `pip install <package>` and retry.

---

## Recommended workflow after installation

A typical first test is:

1. start the GUI with `theatrics`
2. open the **Image Simulation** tab and run a short simulation
3. try **RICS Export** or **FCS Fitting** on known test data
4. verify that figures and result files are written successfully