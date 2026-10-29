"""
PR #239: Fine-tune the foundation potential on our r2SCAN data for CO2 in Mg-MOF-74

Fine-tunes the pretrained foundation potential (PBE data, 89 elements) on our r2SCAN single
points of CO2 in Mg-MOF-74: 1,200 configurations from MD snapshots at different loadings, in an
extxyz file with energies and forces. The backbone is frozen and only the readout is trained,
with the reference energies refitted to r2SCAN. Also wraps the result as an ASE calculator for
the GCMC/MD workflow, and adds a quick relaxation helper.
"""
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

# FoundationPotential: .embedding (one row per element in .z_table, sorted by Z), .interactions,
# .readout, .atomic_energies, .avg_num_neighbors, .index(z) -> embedding rows for numbers z,
# .backbone(z, pos, cell, edge_index, shifts) -> node features, .energy(...) -> total energy
from foundation import FoundationPotential
from graphs import radius_graph_pbc  # (edge_index, shifts) for one periodic structure

CUTOFF = 6.0  # Å, as in pretraining


def load_dataset(path):
    data = []
    for atoms in read(path, index=":"):
        energy = atoms.info.get("energy")
        if energy is None:
            continue  # unlabelled snapshot
        data.append({"atoms": atoms, "energy": energy, "forces": atoms.arrays["forces"]})
    print(f"loaded {len(data)} labelled structures")
    return data


def adapt(model, data):
    z_table = sorted({int(z) for d in data for z in d["atoms"].numbers})  # H, C, O, Mg
    model.embedding.weight.data = model.embedding.weight.data[: len(z_table)].clone()
    model.z_table = z_table
    counts = np.array([[np.sum(d["atoms"].numbers == z) for z in z_table] for d in data])
    energies = np.array([d["energy"] for d in data])
    e0, *_ = np.linalg.lstsq(counts, energies, rcond=None)
    model.atomic_energies = torch.tensor(e0, dtype=torch.float32)
    model.avg_num_neighbors = float(np.mean(
        [radius_graph_pbc(d["atoms"], CUTOFF)[0].shape[1] / len(d["atoms"]) for d in data]))
    for block in [model.embedding, *model.interactions]:
        block.requires_grad = False
    return model


def energy_and_forces(model, atoms, training=False):
    edge_index, shifts = radius_graph_pbc(atoms, CUTOFF)
    z = torch.as_tensor(atoms.numbers)
    pos = torch.tensor(atoms.positions, dtype=torch.float32, requires_grad=True)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
    with torch.no_grad():  # the backbone is frozen, so don't track it
        h = model.backbone(z, pos, cell, edge_index, shifts)
    energy = (model.readout(h).squeeze(-1) + model.atomic_energies[model.index(z)]).sum()
    grad = torch.autograd.grad(energy, pos, create_graph=training, allow_unused=True)[0]
    forces = -grad if grad is not None else torch.zeros_like(pos)
    return energy, forces


def finetune(model, data, epochs=50, lr=1e-4, w_forces=100.0):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for d in data:
            energy, forces = energy_and_forces(model, d["atoms"], training=True)
            f_ref = torch.tensor(d["forces"], dtype=torch.float32)
            loss = ((energy - d["energy"]) / len(d["atoms"])) ** 2 + w_forces * ((forces - f_ref) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


class FinetunedCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model.eval()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.atoms = atoms
        energy, forces = energy_and_forces(self.model, atoms)
        self.results = {"energy": energy.item(), "forces": forces.detach().numpy()}


def relax(model, atoms, steps=500, step_size=0.01, fmax=0.05):
    """Steepest descent; good enough to clean up GCMC snapshots before MD."""
    edge_index, shifts = radius_graph_pbc(atoms, CUTOFF)
    z = torch.as_tensor(atoms.numbers)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
    pos = torch.tensor(atoms.positions, dtype=torch.float32, requires_grad=True)
    for _ in range(steps):
        energy = model.energy(z, pos, cell, edge_index, shifts)
        energy.backward()
        forces = -pos.grad
        if forces.norm(dim=1).max() < fmax:
            break
        with torch.no_grad():
            pos += step_size * forces
    atoms.positions = pos.detach().numpy()
    return atoms
