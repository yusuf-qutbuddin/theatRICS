The software is being actively developed and currently has basic functionalities available, in case of any issues please contact yusufqq@biochem.mpg.de.

# Raster Image Correlation Spectroscopy simulation and analysis, and perpendicular scanning FCS analysis
A modular, extensible graphical user interface for performing and analyzing Raster Image Correlation Spectroscopy (RICS) experiments. This toolkit supports simulation, data import/export, and advanced analysis with a user-friendly workflow designed for membrane biophysics, imaging, and fluorescence correlation studies.
Currently the software is limited to Zeiss (.czi) files and TIFF files for the input image format for the raster scanned image. There will be a future update to involve other commonly used file types from other commercial microscope companies. 
## Features
**Flexible RICS simulations**: Isotropic, anisotropic, and rotated diffusion models.

**Analysis**: Fit and analyze real or simulated image stacks. (Non-GUI batch analysis)

**Progress monitoring**: Responsive GUI with real-time progress and status bars.

**Modular design**: Easily extend with new simulation, import, or analysis modules.

**Visualization**: Integrated with Matplotlib for RICS map display and fitting results.

## Installation and Use

Go to https://theatrics.readthedocs.io/en/latest/index.html for full documentation.

 ## Contributions and Authors

 The majority of the code and functionality is developed by Yusuf Qutbuddin (yusufqq@biochem.mpg.de) and the code and the method is inspired and follows similar algorithms as the [PAM](https://gitlab.com/PAM-PIE/PAM.git) software. Some functionalities have been derived from an earlier script by Jan-Hagen Krohn. Perplexity.ai has been used for debugging, annotation and file parsing algorithms and for searching and implementing tkinter. 
