"""
PR #272: Frozen Structure value object for generated crystals

A periodic crystal is a 3x3 lattice (rows are the cell vectors, in Å) plus atoms
at fractional coordinates (each position as fractions of the three cell vectors,
so values in [0, 1) lie inside the cell). The generative model hands us raw
arrays; this wraps them in an immutable `Structure` so we can deduplicate
samples, cache featurisation by structure, and make noisy copies ("rattling")
to augment the surrogate model's training set.
"""
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Structure:
    lattice: np.ndarray  # (3, 3), Å
    species: tuple[str, ...]
    frac_coords: np.ndarray  # (n_atoms, 3)
    energy: float | None = None  # DFT total energy, eV
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.lattice.shape != (3, 3):
            raise ValueError(f"lattice must be 3x3, got {self.lattice.shape}")
        if self.frac_coords.shape != (len(self.species), 3):
            raise ValueError("frac_coords must have shape (n_atoms, 3)")
        # wrap atoms back into the unit cell
        self.frac_coords[:] = self.frac_coords % 1.0

    def __hash__(self):
        return hash((self.lattice.tobytes(), self.species, self.frac_coords.tobytes()))

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @cached_property
    def cart_coords(self) -> np.ndarray:
        return self.frac_coords @ self.lattice

    def with_energy(self, energy: float) -> "Structure":
        return replace(self, energy=energy)

    @classmethod
    def from_dict(cls, d: dict) -> "Structure":
        return cls(
            lattice=np.array(d["lattice"]),
            species=d["species"],
            frac_coords=np.array(d["frac_coords"]),
            energy=d.get("energy"),
            metadata=d.get("metadata", {}),
        )


def rattle(s: Structure, sigma: float, rng: np.random.Generator) -> Structure:
    """A copy of `s` with Gaussian noise (in fractional units) added to every atom."""
    coords = s.frac_coords
    coords += rng.normal(0.0, sigma, size=coords.shape)
    out = replace(s, frac_coords=coords, energy=None)
    out.metadata["rattle_sigma"] = sigma
    return out


def augment(structures: list[Structure], n_copies: int, sigma: float = 0.01, seed: int = 0) -> list[Structure]:
    rng = np.random.default_rng(seed)
    return [rattle(s, sigma, rng) for s in structures for _ in range(n_copies)]


def deduplicate(structures: Iterable[Structure]) -> list[Structure]:
    """Drop exact repeats, e.g. the same crystal pulled from two source databases."""
    seen: set[Structure] = set()
    unique = []
    for s in structures:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique
