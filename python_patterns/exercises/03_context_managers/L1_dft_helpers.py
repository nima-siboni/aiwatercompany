"""
PR #231: Timing and working-directory helpers for DFT post-processing

Our VASP post-processing scripts all hand-roll the same boilerplate: time a step, log
how long it took, and parse files inside each job's directory. This adds a `Timer`
context manager, an `in_dir` helper that runs a block inside another directory and
comes back afterwards, and `read_energy`, which pulls the final total energy (in eV)
out of a VASP OUTCAR file. Typical use in the collection script:

    for job_dir in sorted(glob("runs/*")):
        with Timer(f"parse {job_dir}"), in_dir(job_dir):
            energy = read_energy("OUTCAR")
        energies[job_dir] = energy
"""
import logging
import os
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)


class Timer:
    """Log how long a block took; the duration is also kept in `elapsed` (seconds)."""

    def __init__(self, label: str):
        self.label = label
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self.start
        log.info("%s took %.2f s", self.label, self.elapsed)
        return True


@contextmanager
def in_dir(path):
    prev = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(prev)


def read_energy(outcar_path: str) -> float:
    """Final total energy in eV.

    VASP prints a line like "  free  energy   TOTEN  =      -1234.56789012 eV"
    after every ionic step; we want the last one.
    """
    energy = None
    f = open(outcar_path)
    for line in f:
        if "free  energy   TOTEN" in line:
            energy = float(line.split()[-2])
    return energy
