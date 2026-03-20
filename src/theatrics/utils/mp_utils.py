# utils/mp_utils.py
import os
import multiprocessing as mp

def set_single_threaded_blas():
    """
    Prevents OpenBLAS/MKL/NumExpr/OMP from using many threads inside each process.
    Call this before importing numpy/scipy-heavy modules if possible.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

def setup_multiprocessing(start_method="spawn", force=True):
    """
    Call once in main entry point (main.py), inside if __name__ == '__main__'.
    On Windows/macOS 'spawn' is the safe default.
    """
    mp.set_start_method(start_method, force=force)

def clamp_workers(requested, max_fraction=0.8, min_workers=1, hard_cap=None):
    """
    Choose a safe number of workers based on machine capacity and user request.

    Parameters
    ----------
    requested : int
        User requested workers (e.g. from GUI input).
    max_fraction : float
        Fraction of logical CPUs to allow (0..1).
    min_workers : int
        Minimum workers.
    hard_cap : int or None
        Absolute maximum, useful for e.g. Windows overhead control.

    Returns
    -------
    int
        Worker count to use.
    """
    ncpu = mp.cpu_count()
    if requested is None:
        requested = ncpu

    try:
        requested = int(requested)
    except Exception:
        requested = ncpu

    allowed = max(min_workers, int(max_fraction * ncpu))
    n = min(requested, allowed)

    if hard_cap is not None:
        n = min(n, int(hard_cap))

    return max(min_workers, n)