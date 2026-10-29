"""
PR #245: Build the MLIP training set from our VASP relaxations of CO2 in MOFs

Walks runs/<mof>_<n>co2/. Each directory is one VASP relaxation; the inputs were written with
pymatgen's MPRelaxSet from the framework with the n CO2 molecules appended at the end, and most
runs keep the framework atoms fixed (selective dynamics) and relax only the CO2. Produces:
- train.extxyz: every ionic step, with REF_energy and REF_forces for MLIP training, and
  tags = 1 on the CO2 atoms so the trainer can up-weight them;
- relaxed.csv: the final energy of each finished run and the binding energy
  E_bind = E(MOF + n CO2) - E(MOF) - n * E(CO2), using our reference runs for the empty MOFs
  and for gas-phase CO2.
Runs that did not finish are skipped.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from ase.io import read, write
from pymatgen.io.vasp import Vasprun

E_CO2 = -22.96  # eV, PBE, from our gas-phase reference run
E_MOF = pd.read_csv("reference/empty_mof_energies.csv", index_col="mof")["energy_ev"]


def finished(run_dir):
    vr = Vasprun(os.path.join(run_dir, "vasprun.xml"), parse_dos=False, parse_eigen=False,
                 exception_on_bad_xml=False)  # a few runs leave a truncated file; don't crash on them
    return vr.converged, vr.final_energy


def labelled_frames(run_dir, n_co2):
    out = []
    for atoms in read(os.path.join(run_dir, "OUTCAR"), index=":"):
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        atoms.calc = None
        atoms.info["REF_energy"] = energy
        atoms.arrays["REF_forces"] = forces
        tags = np.zeros(len(atoms), dtype=int)
        tags[len(atoms) - 3 * n_co2:] = 1  # CO2 atoms were appended last
        atoms.set_tags(tags)
        out.append(atoms)
    return out


def dedupe(frames, decimals=3):
    """Relaxations revisit the same geometry near convergence; keep one frame per energy."""
    seen, kept = set(), []
    for atoms in frames:
        key = round(atoms.info["REF_energy"], decimals)
        if key not in seen:
            seen.add(key)
            kept.append(atoms)
    return kept


def build(root="runs"):
    train, relaxed = [], []
    for run_dir in sorted(glob.glob(os.path.join(root, "*_*co2"))):
        mof, n_co2 = re.match(r"(.+)_(\d+)co2$", os.path.basename(run_dir)).groups()
        n_co2 = int(n_co2)
        converged, energy = finished(run_dir)
        if not converged:
            continue
        train += labelled_frames(run_dir, n_co2)
        relaxed.append({"mof": mof, "n_co2": n_co2, "energy_ev": energy,
                        "e_bind_ev": energy - E_MOF[mof] - n_co2 * E_CO2})
    write("train.extxyz", dedupe(train))
    pd.DataFrame(relaxed).to_csv("relaxed.csv", index=False)
