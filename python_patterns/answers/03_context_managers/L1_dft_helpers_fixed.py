"""Reference fix for PR #231 (python_patterns/exercises/03_context_managers/L1_dft_helpers.py).
See the answer key for the reasoning."""
import logging
import os
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)


class Timer:
    """Log how long a block took; the duration is also kept in `elapsed` (seconds)."""

    def __init__(self, label: str):
        self.label = label
        self.elapsed: float | None = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        status = " (failed)" if exc_type is not None else ""
        log.info("%s took %.2f s%s", self.label, self.elapsed, status)
        # No return value: any exception from the block propagates.


@contextmanager
def in_dir(path):
    """Run a block inside `path` (on Python 3.11+ this is contextlib.chdir)."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def read_energy(outcar_path: str) -> float:
    """Final total energy in eV.

    VASP prints a line like "  free  energy   TOTEN  =      -1234.56789012 eV"
    after every ionic step; we want the last one.
    """
    energy = None
    with open(outcar_path) as f:
        for line in f:
            if "free  energy   TOTEN" in line:
                energy = float(line.split()[-2])
    if energy is None:
        raise ValueError(f"no TOTEN line in {outcar_path}: the job never finished an ionic step")
    return energy
