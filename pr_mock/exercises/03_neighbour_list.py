"""
PR #163: Neighbour list for crystal graphs

Builds edges between atoms within a cutoff radius, respecting periodic
boundary conditions, for use as input to the crystal GNN. Edge features are
Gaussian-expanded distances.
"""
import numpy as np


def frac_to_cart(frac, lattice):
    # lattice: 3x3, rows are the lattice vectors a, b, c (Angstrom)
    return lattice @ frac.T


def neighbour_list(frac_coords, lattice, cutoff=5.0):
    """
    frac_coords: (N, 3) fractional coordinates
    lattice:     (3, 3) lattice matrix, rows are lattice vectors
    Returns a list of (i, j, distance) tuples for all pairs within cutoff.
    """
    n = len(frac_coords)
    edges = []
    for i in range(n):
        for j in range(n):
            d = frac_coords[j] - frac_coords[i]
            d = d - np.round(d)  # minimum image
            cart = frac_to_cart(d, lattice)
            dist = np.sqrt(np.sum(cart ** 2))
            if dist < cutoff:
                edges.append((i, j, dist))
    return edges


def gaussian_expand(dist, n_basis=50, cutoff=5.0):
    centres = np.linspace(0, cutoff, n_basis)
    width = centres[1] - centres[0]
    return np.exp(-((dist[:, None] - centres[None, :]) ** 2) / width ** 2)


def build_graph(structure, cutoff=6.0):
    edges = neighbour_list(structure["frac_coords"], structure["lattice"], cutoff)
    src = np.array([e[0] for e in edges])
    dst = np.array([e[1] for e in edges])
    dist = np.array([e[2] for e in edges])
    return {
        "src": src,
        "dst": dst,
        "edge_attr": gaussian_expand(dist),
        "num_nodes": len(structure["frac_coords"]),
    }
