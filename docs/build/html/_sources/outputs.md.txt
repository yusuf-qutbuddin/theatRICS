# Outputs

The application writes analysis results to disk so that large arrays are not passed
between processes.

Common output types:
- **TIFF**: RICS correlation maps and uncertainty maps
- **CSV**: per-file fit summaries
- **NPZ**: model arrays, residuals, and intermediate fit products
- **SVG/PNG**: exported plots

Exact filenames depend on the worker functions and user-selected output paths.
If you want, paste example output filenames from a typical run and I will document
the naming conventions precisely.