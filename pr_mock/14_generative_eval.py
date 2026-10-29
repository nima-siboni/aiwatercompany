"""
PR #273: Metrics for the crystal generator and the MOF-linker generator

Adds the evaluation we report in the paper and on the team dashboard.
- Crystals (diffusion model, 10k samples per checkpoint): structural validity, uniqueness and
  novelty against the MP-20 training set, and the S.U.N. rate (stable, unique and novel), with
  energies per atom from our MLIP and the MP reference entries for the convex hull.
- Linkers (SMILES generator): validity, uniqueness, novelty and FCD against the training
  linkers, in the same format as the MOSES baselines table in the paper.
"""
import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from rdkit import Chem
from scipy.spatial.distance import pdist

from metrics_utils import fcd  # Fréchet ChemNet distance between two lists of SMILES


# ------------------------------------------------------------------------------- crystals
def structure_valid(structure, min_dist=0.5):
    """CDVAE criterion: no two atoms closer than 0.5 Å."""
    return pdist(structure.cart_coords).min() > min_dist


def unique(samples, matcher):
    kept = []
    for x in samples:
        if not any(matcher.fit(x["structure"], k["structure"]) for k in kept):
            kept.append(x)
    return kept


def novel(samples, train, matcher):
    return [x for x in samples if not any(matcher.fit(x["structure"], t) for t in train)]


def e_above_hull(samples, reference_entries):
    entries = [PDEntry(x["structure"].composition, x["e_per_atom"]) for x in samples]
    diagram = PhaseDiagram(reference_entries + entries)
    return np.array([diagram.get_e_above_hull(e) for e in entries])


def crystal_metrics(samples, train, reference_entries):
    """samples: list of {"structure": Structure, "e_per_atom": float (eV/atom, from the MLIP)}."""
    matcher = StructureMatcher()
    valid = [x for x in samples if structure_valid(x["structure"])]
    uniq = unique(valid, matcher)
    nov = novel(uniq, train, matcher)
    ehull = e_above_hull(nov, reference_entries) if nov else np.array([])
    n = len(samples)
    return {
        "validity": len(valid) / n,
        "uniqueness": len(uniq) / n,
        "novelty": len(nov) / n,
        "sun_rate": float(np.sum(ehull < 0.0)) / n,
    }


# ------------------------------------------------------------------------------- linkers
def linker_metrics(generated, train_smiles, n_fcd=1000):
    mols = [Chem.MolFromSmiles(s) for s in generated]
    valid = [s for s, m in zip(generated, mols) if m is not None]
    canonical = {Chem.MolToSmiles(m) for m in mols if m is not None}
    novel_smiles = canonical - set(train_smiles)
    n = len(generated)
    return {
        "validity": len(valid) / n,
        "uniqueness": len(set(valid)) / n,
        "novelty": len(novel_smiles) / len(canonical),
        "fcd": fcd(valid[:n_fcd], train_smiles),
    }
