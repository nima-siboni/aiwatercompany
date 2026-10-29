"""
PR #236: Train the MOF potential on stresses as well as energies and forces

We want to use the potential for variable-cell relaxations of flexible MOFs (MIL-53
breathing, CO2-loaded Mg-MOF-74), so this adds the stress to the training targets. Labels
come from our VASP single points: total energy (eV), forces (eV/Å) and the stress line from
the OUTCAR ("in kB", in kBar). The energy loss is per atom so that large cells don't
dominate. Also adds a validation metrics report.
"""
import numpy as np
import torch
import torch.nn as nn
from ase.neighborlist import neighbor_list

CUTOFF = 5.0                           # Å
KBAR_TO_EV_PER_A3 = 0.1 / 160.21766    # 1 kBar = 0.1 GPa, and 1 eV/Å^3 = 160.2 GPa


def voigt_to_matrix(s):
    xx, yy, zz, yz, xz, xy = s
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def to_graph(atoms, energy, forces, outcar_stress_kbar):
    i, j, shifts = neighbor_list("ijS", atoms, CUTOFF)
    stress = voigt_to_matrix(outcar_stress_kbar) * KBAR_TO_EV_PER_A3
    return {
        "numbers": torch.as_tensor(atoms.numbers),
        "pos": torch.as_tensor(atoms.positions, dtype=torch.float32),
        "cell": torch.as_tensor(atoms.cell.array, dtype=torch.float32),
        # messages flow from the neighbour j (source) to the centre atom i (target)
        "edge_index": torch.as_tensor(np.stack([j, i])),
        "shifts": torch.as_tensor(shifts, dtype=torch.float32),
        "energy": torch.tensor(energy, dtype=torch.float32),
        "forces": torch.as_tensor(forces, dtype=torch.float32),
        "stress": torch.as_tensor(stress, dtype=torch.float32),
    }


def collate(graphs):
    offset, edge_index, graph = 0, [], []
    for k, g in enumerate(graphs):
        n = len(g["numbers"])
        edge_index.append(g["edge_index"] + offset)
        graph.append(torch.full((n,), k))
        offset += n
    cat = lambda key: torch.cat([g[key] for g in graphs])
    stack = lambda key: torch.stack([g[key] for g in graphs])
    return {
        "numbers": cat("numbers"), "pos": cat("pos"), "shifts": cat("shifts"),
        "forces": cat("forces"), "cell": stack("cell"), "energy": stack("energy"),
        "stress": stack("stress"), "edge_index": torch.cat(edge_index, dim=1),
        "graph": torch.cat(graph),
        "n_atoms": torch.tensor([len(g["numbers"]) for g in graphs]),
    }


def cosine_cutoff(d):
    return 0.5 * (torch.cos(torch.pi * d / CUTOFF) + 1.0) * (d < CUTOFF)


class MOFPotential(nn.Module):
    def __init__(self, hidden=64, n_rbf=32, n_layers=3, max_z=100):
        super().__init__()
        self.embed = nn.Embedding(max_z, hidden)
        self.register_buffer("centres", torch.linspace(0.0, CUTOFF, n_rbf))
        self.filters = nn.ModuleList(
            nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_layers))
        self.updates = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(n_layers))
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, pos, cell, batch):
        src, dst = batch["edge_index"]
        offsets = torch.einsum("ek,ekl->el", batch["shifts"], cell[batch["graph"][src]])
        d = (pos[dst] - pos[src] + offsets).norm(dim=-1)
        rbf = torch.exp(-((d[:, None] - self.centres) ** 2) / 0.1) * cosine_cutoff(d)[:, None]
        h = self.embed(batch["numbers"])
        for filt, update in zip(self.filters, self.updates):
            agg = torch.zeros_like(h)
            agg[dst] += filt(rbf) * h[src]
            h = h + update(agg)
        e_atom = self.readout(h).squeeze(-1)
        n_graphs = cell.shape[0]
        counts = torch.bincount(batch["graph"], minlength=n_graphs)
        energy = torch.zeros(n_graphs).index_add(0, batch["graph"], e_atom) / counts
        return energy  # energy per atom, to match the per-atom loss


def energy_forces_stress(model, batch, training):
    pos = batch["pos"].requires_grad_(True)
    eps = torch.zeros(3, 3, requires_grad=True)
    strain = 0.5 * (eps + eps.T)
    cell = batch["cell"] + batch["cell"] @ strain
    energy = model(pos + pos @ strain, cell, batch)
    grad_pos, grad_eps = torch.autograd.grad(energy.sum(), (pos, eps), retain_graph=training)
    volume = torch.linalg.det(batch["cell"])
    forces = -grad_pos
    stress = grad_eps / volume[:, None, None]
    return energy, forces, stress


def train(model, train_graphs, val_graphs, epochs=200, lr=1e-3, batch_size=8,
          w_energy=1.0, w_forces=100.0, w_stress=10.0, seed=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_graphs))
        for start in range(0, len(order), batch_size):
            batch = collate([train_graphs[i] for i in order[start:start + batch_size]])
            energy, forces, stress = energy_forces_stress(model, batch, training=True)
            loss = (w_energy * ((energy - batch["energy"] / batch["n_atoms"]) ** 2).mean()
                    + w_forces * ((forces - batch["forces"]) ** 2).mean()
                    + w_stress * ((stress - batch["stress"]) ** 2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            print(epoch, evaluate(model, val_graphs))
    return model


def evaluate(model, graphs, batch_size=32):
    model.eval()
    e_err, f_err, n_structures, n_atoms = 0.0, 0.0, 0, 0
    for start in range(0, len(graphs), batch_size):
        batch = collate(graphs[start:start + batch_size])
        energy, forces, _ = energy_forces_stress(model, batch, training=False)
        per_atom_ref = batch["energy"] / batch["n_atoms"]
        e_err += ((energy - per_atom_ref).abs().sum() / len(batch["pos"])).item()
        f_err += (forces - batch["forces"]).norm(dim=-1).sum().item()
        n_structures += len(batch["energy"])
        n_atoms += len(batch["pos"])
    return {"energy_mae_meV_per_atom": 1000 * e_err / n_structures,
            "force_mae_meV_per_A": 1000 * f_err / n_atoms}
