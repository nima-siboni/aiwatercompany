# Answer key: 09_potential_training.py

**Good clarifying questions:** Are all training structures periodic, or are there molecules
and slabs too? Do we parse the OUTCAR stress line ourselves, and which sign and component order
does the model expect? Will the potential run MD and relaxations (smoothness, float64), and on
GPUs? How big are the cells (how many edges cross the cell boundary)? What energy reference do
the labels use? Is validation split by framework rather than by MD frame?

Six blockers is a lot, so rank them by blast radius: 1 and 2 corrupt every energy, 3 and 4
every force, and 5 and 6 only the stress. Also note that 3 hides 5 and 6 during training: once
the force and stress terms reach the weights, the wrong stress labels start doing damage.

## Blockers

1. **`agg[dst] += ...` keeps one message per atom** (`MOFPotential.forward`). Indexed
   assignment with repeated indices writes; it doesn't accumulate: three messages into node 0
   give `[1, 1, 1]` instead of `[3, 2, 1]`. In a 2×2×2 primitive Cu test cell every atom has 42
   neighbours within 5 Å, and the model sees one of them. Nothing errors and the loss still
   falls. PyTorch documents which duplicate wins as undefined. *Fix:*
   `agg = torch.zeros_like(h).index_add_(0, dst, msg)` (or `index_put_(..., accumulate=True)`).
   Test it against an explicit loop on a three-atom graph.

2. **Every periodic edge gets the wrong vector** (`to_graph` and `forward`). ASE defines
   `D = pos[j] - pos[i] + S @ cell`. The PR swaps the order so that j is the source, but then
   computes `pos[dst] - pos[src] + offsets`, which is `pos[i] - pos[j] + S @ cell`. When S = 0
   only the direction flips, which a distance-based model can't see, so tests on molecules
   pass. For every edge that crosses the cell boundary the length is wrong. In the test cell,
   282 of 336 edges were periodic, all 282 had the wrong length (up to 11 Å off), and all of
   them landed beyond the cutoff: the model saw no periodic neighbours at all. *Fix:*
   `pos[src] - pos[dst] + offsets`, or negate S when you swap i and j. Test the edge lengths
   against `neighbor_list("d")` on a triclinic cell.

3. **The force and stress losses never reach the weights** (`energy_forces_stress`).
   `autograd.grad` without `create_graph=True` returns plain tensors (`forces.requires_grad` is
   False), so the force and stress terms are constants. With and without them, the parameter
   gradients differed only by run-to-run noise. The logged loss still falls, because the
   energy term trains, so it looks healthy. *Fix:* `create_graph=training`, as MACE does. At
   evaluation, leaving it off is right.

4. **Forces and stresses come from the per-atom energy.** Mean pooling returns E/N, and forces
   must be −∂E_total/∂r, so every force and stress is divided by the atom count: 0.25× for a
   4-atom cell, 0.037× for 27 atoms, and 0.0025× for a 400-atom MOF cell. The energy and force
   terms then pull against each other. The per-atom loss is a good idea; it belongs in the
   loss, not in the model. *Fix:* sum-pool to the total energy, take forces and stress from it,
   and divide by `n_atoms` only in the energy loss. Test: a 2×2×2 supercell must give 8× the
   energy and the same forces.

5. **One strain tensor for the whole batch.** `eps` is (3, 3), so ∂E/∂ε is summed over every
   structure in the batch and then divided by each structure's own volume. In a batch of two,
   the stresses differed from the single-structure ones by 681% and 15%. Tests with batch
   size 1 pass. *Fix:* `eps` of shape (B, 3, 3), applied as `strain[graph]` to the positions and
   per structure to the cells (MACE's `get_symmetric_displacement`). With that fix, batched and
   single stresses agreed to 1e-8, and matched finite differences to 1e-5 eV/Å³.

6. **The stress labels have the wrong sign and permuted shear components**
   (`voigt_to_matrix`). The OUTCAR line is XX YY ZZ XY YZ ZX, with compression positive. The
   model's stress, (1/V) ∂E/∂ε, follows ASE: compression negative, Voigt order xx yy zz yz xz
   xy. The PR unpacks the OUTCAR line as if it were ASE order and doesn't negate it. For a
   sheared, compressed cell every component came out with the wrong sign and the off-diagonal
   terms in the wrong places. Unsheared test cells hide the permutation but not the sign.
   ASE's own parser does `-stress[[0, 1, 2, 4, 5, 3]] * 0.1 * GPa`. *Fix:* use ASE's reader, or
   negate and reorder; test that a compressed cell gets negative stress.

## Major

7. **The reported energy MAE is divided by the batch's atom count** (`evaluate`). The errors are
   already per atom; dividing each batch's sum by `len(batch["pos"])` (atoms in the whole batch)
   and then by the number of structures divides twice. It reported 1.1 meV/atom when the true
   per-atom MAE was 108 meV/atom, with 96 atoms in the batch. The number also changes with the
   batch size. *Fix:* accumulate `(energy - per_atom_ref).abs().sum()` and divide by the number
   of structures.

8. **Tensors created without a dtype or device.** `torch.zeros(n_graphs)` in `forward` and `eps`
   in `energy_forces_stress` are float32 on the CPU. A float64 model, which you want for
   relaxations, fails with `index_add_(): self (Float) and source (Double) must have the same
   scalar type`, and the first GPU run fails the same way with a device mismatch. Loud, but it
   breaks the first real run. *Fix:* `e_atom.new_zeros(n_graphs)`, and
   `torch.zeros(..., dtype=cell.dtype, device=cell.device)`.

## Minor

9. **Which force MAE?** `norm(dim=-1).mean()` is the per-atom vector error. Most papers and the
   MACE and NequIP logs report the per-component MAE, which is about half as large for
   isotropic errors (the ratio was 2.02 here). Name the metric for what it is, or report both,
   or comparisons with baselines are off by 2×.
10. **Signed volume.** `torch.linalg.det(cell)` is negative for left-handed cells, so the stress
    flips sign for exactly those structures. CIFs from some programs are left-handed. Use
    `.abs()`.
11. **No energy reference.** VASP totals are about −5 to −8 eV/atom and the model starts near 0.
    Subtract per-element reference energies, or at least the mean per-atom energy, before
    training. It isn't a correctness bug, but the first epochs go on learning an offset.
12. **No tests of the physics.** Finite-difference forces and stress, rotation and translation
    invariance, supercell consistency, and batched-equals-single would each have caught at
    least one of the blockers.
13. **Small things:** the `cat`/`stack` lambdas, a dict of tensors with implicit shapes, the
    magic width `0.1` in the radial basis, and `requires_grad_` on the caller's tensor.

## Red herrings (say why they're fine)

- **The collate offsets are right**: `edge_index + offset`, and the cells are stacked to
  (B, 3, 3). Batched energies equal the single-structure ones.
- **`autograd.grad(energy.sum(), pos)` over a whole batch** gives each structure's own forces,
  because no energy depends on another structure's atoms.
- **No `torch.no_grad()` in `evaluate`** is deliberate: forces need gradients. Under `no_grad`
  it raises `element 0 of tensors does not require grad`. What matters at evaluation is
  `create_graph=False`, and that is the case.
- **`KBAR_TO_EV_PER_A3 = 0.1 / 160.21766`** is correct; it equals `0.1 * ase.units.GPa`.
- **The 100× force weight** looks arbitrary, but MACE and NequIP use similar ones.
- **Symmetrising the strain** isn't needed for an invariant model, but it's harmless and a
  common guard.

## Model 1-minute summary
"This doesn't train what it says. The force and stress losses never reach the weights, because
`create_graph` is missing, so only the energy trains. The energy itself is computed on a broken
graph: the indexed `+=` keeps one message per atom instead of summing 42 of them, and the edge
direction was swapped without negating the shift, so every periodic edge has the wrong length.
In my test case all of them fell outside the cutoff. Mean pooling then makes forces and stresses
N times too small. The stress path adds two more: one strain tensor shared by the whole batch,
and OUTCAR labels used without the sign flip and reordering. Finally, the reported energy MAE is
divided by the batch's atom count, so it looks about 100 times better than it is. Tests I'd ask
for: finite-difference forces and stress, batched-equals-single, a supercell test, and edge
lengths against ASE on a triclinic cell."
