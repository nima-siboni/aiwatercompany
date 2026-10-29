"""
PR #222: Cache derived properties on Framework

Screening scripts ask a framework for its volume, density and predicted CO2
uptake over and over (once per filter, once per plot), and the uptake call goes
to the surrogate model, which takes about 50 ms. This caches them on the
object. `remove_guests()` deletes the solvent molecules left in the pores after
synthesis ("activation"), which we do before screening. Uptake is in mmol/g at
a temperature in K and a CO2 pressure in bar; flue gas is about 0.15 bar of
CO2. Generated candidates have no mof_id until they are registered.
"""
import copy
from functools import cached_property, lru_cache

import numpy as np

ATOMIC_MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Mg": 24.305,
    "Al": 26.982, "S": 32.06, "Cl": 35.45, "Ti": 47.867, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Zr": 91.224,
}
AVOGADRO = 6.02214076e23


class Framework:
    def __init__(self, lattice, species, frac_coords, predictor, mof_id: str | None = None):
        self.lattice = np.asarray(lattice, dtype=float)  # rows are the cell vectors, in angstrom
        self.species = list(species)
        self.frac_coords = np.asarray(frac_coords, dtype=float)
        self.predictor = predictor  # (descriptor, temperature, pressure) -> uptake
        self.mof_id = mof_id

    def __eq__(self, other):
        return isinstance(other, Framework) and self.mof_id == other.mof_id

    def __hash__(self):
        return hash(self.mof_id)

    @cached_property
    def volume(self) -> float:
        """Cell volume in cubic angstrom."""
        return abs(float(np.linalg.det(self.lattice)))

    @cached_property
    def density(self) -> float:
        """Density in g/cm^3."""
        mass_g = sum(ATOMIC_MASS[s] for s in self.species) / AVOGADRO
        return mass_g / (self.volume * 1e-24)

    def descriptor(self) -> np.ndarray:
        return np.array([self.density, self.volume / len(self.species), len(self.species)])

    def uptake(self, temperature: float, pressure: float) -> float:
        """Predicted CO2 uptake in mmol/g."""
        return self._uptake(round(temperature), round(pressure, 1))

    @lru_cache(maxsize=None)
    def _uptake(self, temperature: int, pressure: float) -> float:
        return float(self.predictor(self.descriptor(), temperature, pressure))

    def remove_guests(self, guest_indices) -> None:
        """Delete the solvent atoms at these indices (activation), in place."""
        keep = np.setdiff1d(np.arange(len(self.species)), guest_indices)
        self.species = [self.species[i] for i in keep]
        self.frac_coords = self.frac_coords[keep]

    def strain(self, factor: float) -> None:
        """Scale the cell by `factor` in every direction, e.g. for thermal expansion."""
        self.lattice = self.lattice * factor

    def activated(self, guest_indices) -> "Framework":
        """A copy with the solvent removed; this framework is left unchanged."""
        new = copy.copy(self)
        new.remove_guests(guest_indices)
        return new
