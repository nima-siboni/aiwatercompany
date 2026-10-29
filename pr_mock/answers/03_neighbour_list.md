# Answer key: 03_neighbour_list.py

**Good clarifying questions:** What cell sizes and atom counts do we expect (MOF unit cells
can have hundreds to thousands of atoms; inorganic cells can be 3–5 Å)? Are lattices
always orthogonal? Does the GNN need image offsets (for forces/stress or directional
features)? Are self-loops wanted? Why not use `pymatgen`/`ase`/`matscipy`?

## Blockers

1. **`frac_to_cart` uses the wrong matrix convention.** The comment says the rows are
   lattice vectors, so Cartesian = `frac @ lattice`. `lattice @ frac.T` uses the *columns*
   instead. For cubic/orthorhombic cells (a diagonal or symmetric matrix) both give the same
   answer, so tests on simple cells pass. For hexagonal, monoclinic or triclinic cells
   (common in MOFs) every distance is wrong. For an (N, 3) input it also returns (3, N).
   *Fix:* `frac @ lattice`, and add a test on a triclinic cell checked against a
   reference library.

2. **Single-image minimum image is wrong for small or skewed cells.** `d - round(d)` keeps
   only one periodic image per pair, and it's only the true nearest image for orthogonal
   cells. Whenever `cutoff` is more than half the cell's perpendicular width (here 6 Å,
   so any cell under 12 Å, which covers most inorganic crystals and many MOFs), neighbours
   are missed: other images of *j* and periodic images of *i* itself. The graph silently
   loses edges. *Fix:* loop over image offsets
   `n_k = ceil(cutoff / perp_width_k)` in each direction, or use a library neighbour list.

3. **The cutoff isn't passed through.** `build_graph` defaults to `cutoff=6.0`, but
   `gaussian_expand` is called with its own default of `cutoff=5.0`. Edges at 5–6 Å fall
   outside the basis centres, so their features decay towards zero and they're nearly
   indistinguishable from each other. Nothing errors. *Fix:* pass `cutoff` through, and
   keep one source of truth (a config value).

## Major

4. **Self-edges.** For `i == j` the distance is 0 < cutoff, so every node gets a self-loop
   with distance 0. Most crystal GNNs exclude these, and they can dominate message passing.
   Exclude them explicitly; the periodic images of *i* are valid edges and should stay
   once fix 2 is in.

5. **Performance.** It's an O(N²) Python double loop with a NumPy call per pair. A
   2,000-atom MOF means 4M iterations, which is tens of seconds per structure, times a
   screening library of thousands. Naive vectorisation (an N×N×3 array) runs out of memory
   for large cells. *Fix:* a cell list or periodic KD-tree, or a library
   (`ase.neighborlist.neighbor_list`, `pymatgen`'s `get_neighbor_list`, `matscipy`).
   Rewriting this is itself a review point: "do we need to own this?"

6. **No image offsets in the output.** Without the integer offset of each edge you can't
   compute forces or stresses, build directional features, or tell apart multiple edges
   between the same (i, j) pair through different images. Return `(src, dst, offset, dist)`.

## Minor

7. **Empty graphs.** If no pairs are within the cutoff (an isolated atom, a sparse cell),
   `np.array([])` has dtype float64, so `src` and `dst` aren't integer indices and
   downstream indexing breaks. Use `np.array(..., dtype=np.int64)`, or build arrays
   directly instead of tuples.
8. **`gaussian_expand` with `n_basis=1`** gives `width = 0`, which divides by zero, and
   `centres[1]` fails anyway. Validate `n_basis >= 2`. Also, the usual parameterisation is
   `exp(-gamma * (d - mu)^2)`; check that the width definition is intentional.
9. **No input validation** of shapes, dtypes, or a degenerate (singular) lattice.
10. **Data structure:** a list of tuples, then three list comprehensions; return NumPy arrays
    or a typed container. A dict with magic string keys is easy to misuse.

## Red herrings
- `np.round` rounds half to even at exactly 0.5. In an orthogonal cell both images are
  equidistant at that point, so it doesn't matter. It's only a problem in combination with
  the skewed-cell issue (#2).
- `dist < cutoff` vs `<=` is a convention, not a bug; just be consistent with the basis/envelope.

## Model 1-minute summary
"The edges are wrong for any non-orthogonal or small cell, for two reasons: the
fractional-to-Cartesian conversion uses the transposed lattice, and the single-image minimum
image misses neighbours when the cutoff exceeds half the cell width. Tests on cubic cells won't
catch either. There's also a cutoff mismatch between graph building and the edge features.
After those: drop self-loops, return image offsets, and replace the O(N²) loop with a
cell list or an existing library. I'd add a test comparing against `ase` on a triclinic MOF."
