"""
PR #251: Choose the featuriser from the experiment config

train.py used to hard-code the composition featuriser. This adds a factory so
an experiment JSON can pick one by name, e.g.
    "featuriser": {"kind": "rdf", "params": {"n_bins": 64}}
A featuriser turns a structure into a fixed-length vector for the surrogate
model; "zeopp" looks up pore descriptors (pore diameter, surface area) that
Zeo++ precomputed for every MOF in the database. train.py calls
make_featuriser(cfg["featuriser"]) once per cross-validation fold.
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


def make_featuriser(config: dict):
    kind = config.pop("kind", "composition")
    params = config.pop("params", {})
    featurisers = {
        "composition": CompositionFeaturiser(**params),
        "rdf": RDFFeaturiser(**params),
        "zeopp": ZeoppFeaturiser(**params),
    }
    return featurisers[kind]
