"""
Reference fix for PR #222 (python_patterns/exercises/02_caching/L2_framework_properties.py).
See the answer key for the reasoning.
"""
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
        # Per-object cache: it goes away with the object, and every mutator clears it.
        self._uptake_cache: dict[tuple[float, float], float] = {}

    def __eq__(self, other):
        if not isinstance(other, Framework):
            return NotImplemented
        if self.mof_id is None or other.mof_id is None:
            return self is other  # unregistered candidates are only equal to themselves
        return self.mof_id == other.mof_id

    def __hash__(self):
        return hash(self.mof_id) if self.mof_id is not None else id(self)

    # Volume and density take microseconds, so they're plain properties: never stale.
    @property
    def volume(self) -> float:
        """Cell volume in cubic angstrom."""
        return abs(float(np.linalg.det(self.lattice)))

    @property
    def density(self) -> float:
        """Density in g/cm^3."""
        mass_g = sum(ATOMIC_MASS[s] for s in self.species) / AVOGADRO
        return mass_g / (self.volume * 1e-24)

    def descriptor(self) -> np.ndarray:
        return np.array([self.density, self.volume / len(self.species), len(self.species)])

    def uptake(self, temperature: float, pressure: float) -> float:
        """Predicted CO2 uptake in mmol/g, at exactly the conditions asked for."""
        key = (float(temperature), float(pressure))
        if key not in self._uptake_cache:
            self._uptake_cache[key] = float(self.predictor(self.descriptor(), temperature, pressure))
        return self._uptake_cache[key]

    def remove_guests(self, guest_indices) -> None:
        """Delete the solvent atoms at these indices (activation), in place."""
        keep = np.setdiff1d(np.arange(len(self.species)), guest_indices)
        self.species = [self.species[i] for i in keep]
        self.frac_coords = self.frac_coords[keep]
        self._uptake_cache.clear()

    def strain(self, factor: float) -> None:
        """Scale the cell by `factor` in every direction, e.g. for thermal expansion."""
        self.lattice = self.lattice * factor
        self._uptake_cache.clear()

    def activated(self, guest_indices) -> "Framework":
        """A copy with the solvent removed; this framework is left unchanged."""
        # A new object, not copy.copy: a shallow copy would share the original's cache dict.
        new = Framework(self.lattice, self.species, self.frac_coords, self.predictor, self.mof_id)
        new.remove_guests(guest_indices)
        return new
