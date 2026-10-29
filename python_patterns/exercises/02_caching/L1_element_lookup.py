"""
PR #221: Cache element lookups and pair energies in the featuriser

Profiling the featurisation job showed most of its time going into element
lookups (one database query per atom) and into the Lennard-Jones pair energies
behind our host-guest descriptor, so this puts functools caches on both. The
Lennard-Jones formula gives the interaction energy of two atoms a distance r
apart from two per-element parameters, epsilon and sigma. For two different
elements the parameters are combined: the geometric mean of the epsilons and
the arithmetic mean of the sigmas. The descriptor sums this energy between a
probe atom and every framework atom.
"""
import sqlite3
from functools import lru_cache

import numpy as np

ELEMENT_DB = "data/elements.sqlite"


def _read_element_row(symbol: str) -> tuple | None:
    with sqlite3.connect(ELEMENT_DB) as conn:
        return conn.execute(
            "SELECT mass, electronegativity, epsilon, sigma FROM elements WHERE symbol = ?",
            (symbol,),
        ).fetchone()


@lru_cache(maxsize=None)
def element_properties(symbol: str) -> dict:
    mass, electronegativity, epsilon, sigma = _read_element_row(symbol)
    return {"mass": mass, "electronegativity": electronegativity, "epsilon": epsilon, "sigma": sigma}


@lru_cache(maxsize=1024)
def mean_property(symbols: list[str], prop: str) -> float:
    return float(np.mean([element_properties(s)[prop] for s in symbols]))


@lru_cache(maxsize=None)
def pair_energy(r: float, epsilon: float, sigma: float) -> float:
    sr6 = (sigma / r) ** 6
    return 4 * epsilon * (sr6 * sr6 - sr6)


def probe_energy(symbols: list[str], distances: np.ndarray, probe: str = "C") -> float:
    """Summed probe-framework energy in kcal/mol; distances[i] (angstrom) is to atom i."""
    p = element_properties(probe)
    total = 0.0
    for symbol, r in zip(symbols, distances):
        params = element_properties(symbol)
        # Lorentz-Berthelot mixing rules for the unlike pair
        params["epsilon"] = (params["epsilon"] * p["epsilon"]) ** 0.5
        params["sigma"] = (params["sigma"] + p["sigma"]) / 2
        total += pair_energy(r, params["epsilon"], params["sigma"])
    return total
