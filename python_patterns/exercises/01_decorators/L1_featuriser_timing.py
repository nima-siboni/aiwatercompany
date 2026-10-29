"""
PR #211: Time each featuriser and register them in one place

Featurising 40k MOFs for the surrogate model has got slow, and we can't tell
which descriptor is to blame. This adds a `@timed` decorator that records how
long each featuriser takes, and a `@register` decorator so `featurise()` picks
up every descriptor automatically instead of us keeping a list in sync. A
featuriser maps one crystal structure (lattice vectors plus a list of element
symbols) to one number that the model uses as an input feature.
"""
import logging
import time
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

METALS = {"Zn", "Cu", "Co", "Ni", "Mg", "Zr", "Al", "Fe", "Cr", "Mn"}

FEATURISERS = {}
TIMINGS = defaultdict(float)  # featuriser name -> total seconds


def timed(func):
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        elapsed = time.time() - start
        TIMINGS[func.__name__] += elapsed
        logger.debug("%s took %.2f ms", func.__name__, 1000 * elapsed)

    return wrapper


def register(func):
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
