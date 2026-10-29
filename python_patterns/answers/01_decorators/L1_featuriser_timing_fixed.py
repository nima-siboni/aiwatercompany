"""
Reference fix for PR #211 (python_patterns/exercises/01_decorators/L1_featuriser_timing.py).
See the answer key for the reasoning.
"""
import functools
import logging
import time
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

METALS = {"Zn", "Cu", "Co", "Ni", "Mg", "Zr", "Al", "Fe", "Cr", "Mn"}

FEATURISERS = {}
TIMINGS = defaultdict(float)  # featuriser name -> total seconds


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            TIMINGS[func.__name__] += elapsed
            logger.debug("%s took %.2f ms", func.__name__, 1000 * elapsed)

    return wrapper


def register(func):
    if func.__name__ in FEATURISERS:
        raise ValueError(f"a featuriser called {func.__name__!r} is already registered")
    FEATURISERS[func.__name__] = func
    return func


@register
@timed
def volume_per_atom(structure, include_hydrogen: bool = True) -> float:
    """Cell volume per atom in cubic angstrom. Experimental CIFs often lack H atoms."""
    n_atoms = sum(1 for s in structure.species if include_hydrogen or s != "H")
    return abs(float(np.linalg.det(structure.lattice))) / n_atoms


@register
@timed
def metal_fraction(structure) -> float:
    return sum(s in METALS for s in structure.species) / len(structure.species)


@register
@timed
def n_elements(structure) -> float:
    return float(len(set(structure.species)))


def featurise(structures) -> tuple[list[str], np.ndarray]:
    """Feature names, and an (n_structures, n_features) matrix in the same order."""
    names = sorted(FEATURISERS)
    X = np.array([[FEATURISERS[name](s) for name in names] for s in structures], dtype=float)
    return names, X
