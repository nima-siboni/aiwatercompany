"""
Reference fix for PR #251 (python_patterns/exercises/05_factory_registry/L1_featuriser_factory.py).
See the answer key for the reasoning.
"""
import numpy as np


class CompositionFeaturiser:
    """Fraction of atoms of each listed element (other elements are ignored)."""

    def __init__(self, elements: tuple[str, ...] = ("H", "C", "N", "O", "Zn", "Cu", "Zr")):
        self.elements = elements

    def featurise(self, structure) -> np.ndarray:
        counts = np.array([structure.symbols.count(el) for el in self.elements], dtype=float)
        return counts / len(structure.symbols)


class RDFFeaturiser:
    """Histogram of interatomic distances up to r_max (Å), per atom."""

    def __init__(self, r_max: float = 8.0, n_bins: int = 32):
        self.edges = np.linspace(0.0, r_max, n_bins + 1)

    def featurise(self, structure) -> np.ndarray:
        hist, _ = np.histogram(structure.pair_distances(self.edges[-1]), bins=self.edges)
        return hist / len(structure.symbols)


class ZeoppFeaturiser:
    """Zeo++ pore descriptors for the whole database (~1.5 GB), looked up by MOF id."""

    def __init__(self, path: str = "data/zeopp_descriptors.npz"):
        with np.load(path) as data:
            self.rows = dict(zip(data["mof_id"], data["descriptors"]))

    def featurise(self, structure) -> np.ndarray:
        return self.rows[structure.mof_id]


# Classes, not instances: only the featuriser that was asked for gets built.
FEATURISERS = {
    "composition": CompositionFeaturiser,
    "rdf": RDFFeaturiser,
    "zeopp": ZeoppFeaturiser,
}


def make_featuriser(config: dict):
    """Build the featuriser named by config["kind"] with config["params"]. Doesn't modify config."""
    unknown = config.keys() - {"kind", "params"}
    if unknown:
        raise ValueError(f"unknown featuriser config keys {sorted(unknown)}; expected 'kind' and 'params'")
    kind = config.get("kind")
    if kind not in FEATURISERS:
        raise ValueError(f"unknown featuriser {kind!r}; expected one of {sorted(FEATURISERS)}")
    return FEATURISERS[kind](**config.get("params", {}))
