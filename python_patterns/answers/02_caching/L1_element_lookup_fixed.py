"""
Reference fix for PR #221 (python_patterns/exercises/02_caching/L1_element_lookup.py).
See the answer key for the reasoning.
"""
import contextlib
import sqlite3
from functools import lru_cache
from types import MappingProxyType

import numpy as np

ELEMENT_DB = "data/elements.sqlite"


def _read_element_row(symbol: str) -> tuple | None:
    # `with sqlite3.connect(...)` only commits or rolls back; closing() closes the connection.
    with contextlib.closing(sqlite3.connect(ELEMENT_DB)) as conn:
        return conn.execute(
            "SELECT mass, electronegativity, epsilon, sigma FROM elements WHERE symbol = ?",
            (symbol,),
        ).fetchone()


@lru_cache(maxsize=None)  # about a hundred possible symbols, so unbounded is fine
def element_properties(symbol: str) -> MappingProxyType:
    row = _read_element_row(symbol)
    if row is None:
        raise KeyError(f"unknown element {symbol!r}")
    mass, electronegativity, epsilon, sigma = row
    # Read-only: every caller shares this one cached object.
    return MappingProxyType(
        {"mass": mass, "electronegativity": electronegativity, "epsilon": epsilon, "sigma": sigma}
    )


def mean_property(symbols: list[str], prop: str) -> float:
    return float(np.mean([element_properties(s)[prop] for s in symbols]))


def pair_energy(r, epsilon, sigma):
    """Lennard-Jones energy; works elementwise on numpy arrays as well as on floats."""
    sr6 = (sigma / r) ** 6
    return 4 * epsilon * (sr6 * sr6 - sr6)


def probe_energy(symbols: list[str], distances: np.ndarray, probe: str = "C") -> float:
    """Summed probe-framework energy in kcal/mol; distances[i] (angstrom) is to atom i."""
    if len(symbols) != len(distances):
        raise ValueError(f"{len(symbols)} symbols but {len(distances)} distances")
    p = element_properties(probe)
    epsilon = np.array([element_properties(s)["epsilon"] for s in symbols])
    sigma = np.array([element_properties(s)["sigma"] for s in symbols])
    # Lorentz-Berthelot mixing rules for the unlike pairs, in new arrays
    mixed_epsilon = np.sqrt(epsilon * p["epsilon"])
    mixed_sigma = (sigma + p["sigma"]) / 2
    return float(np.sum(pair_energy(np.asarray(distances, dtype=float), mixed_epsilon, mixed_sigma)))
