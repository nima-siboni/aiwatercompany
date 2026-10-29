"""
Reference fix for PR #272 (python_patterns/exercises/07_dataclasses/L2_frozen_structure.py).
See the answer key for the reasoning.
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
        # frozen=True only blocks attribute assignment, so take our own read-only float copies.
        lattice = np.array(self.lattice, dtype=float)
        coords = np.asarray(self.frac_coords, dtype=float) % 1.0  # % returns a new array
        species = tuple(self.species)
        if lattice.shape != (3, 3):
            raise ValueError(f"lattice must be 3x3, got {lattice.shape}")
        if coords.shape != (len(species), 3):
            raise ValueError("frac_coords must have shape (n_atoms, 3)")
        lattice.flags.writeable = False
        coords.flags.writeable = False
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "frac_coords", coords)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @cached_property
    def _key(self) -> tuple:
        # Identity is species plus geometry; energy and metadata don't make a different crystal.
        return (self.species, self.lattice.tobytes(), self.frac_coords.tobytes())

    def __eq__(self, other):
        if not isinstance(other, Structure):
            return NotImplemented
        return self._key == other._key

    def __hash__(self):
        return hash(self._key)

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @cached_property
    def cart_coords(self) -> np.ndarray:
        cart = self.frac_coords @ self.lattice
        cart.flags.writeable = False
        return cart

    def with_energy(self, energy: float) -> "Structure":
        return replace(self, energy=energy)

    @classmethod
    def from_dict(cls, d: dict) -> "Structure":
        return cls(
            lattice=d["lattice"],
            species=d["species"],
            frac_coords=d["frac_coords"],
            energy=d.get("energy"),
            metadata=d.get("metadata", {}),
        )


def rattle(s: Structure, sigma: float, rng: np.random.Generator) -> Structure:
    """A copy of `s` with Gaussian noise (in fractional units) added to every atom."""
    noise = rng.normal(0.0, sigma, size=s.frac_coords.shape)
    return replace(
        s,
        frac_coords=s.frac_coords + noise,
        energy=None,
        metadata={**s.metadata, "rattle_sigma": sigma},
    )


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
