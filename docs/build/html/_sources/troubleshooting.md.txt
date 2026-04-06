# Troubleshooting

## Read the Docs build fails on `pylibCZIrw`
This is common because CZI readers may require system libraries not present on RTD.

Solution (recommended):
- Add a `docs` extra in `pyproject.toml` and configure RTD to install only docs
  dependencies, while mocking `pylibCZIrw` in `conf.py`.

If you paste the RTD build log error, I can give you the exact fix.

## Matplotlib / display errors
RTD is headless. The docs configuration forces:
- `MPLBACKEND=Agg`

## Multiprocessing notes (Windows)
When running from source on Windows, ensure GUI startup is protected by:
```python
if __name__ == "__main__":
    main()

```

## Tkinter missing (Linux)
Install system Tk support (package name varies by distribution, often `python3-tk`).