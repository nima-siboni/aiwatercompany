# Answer key: 10_potential_finetune.py

**Good clarifying questions:** Which ASE version wrote the extxyz file, and where are the labels
in it? What do we need from the model: binding energies only, or absolute energies across other
frameworks, water and so on? Will the calculator drive MD and ASE optimisers? How is the data
split, given that it comes from MD snapshots? Is the foundation model's element table available?

## Blockers

1. **The fine-tuning set is empty** (`load_dataset`). Since ASE 3.23, extxyz energies and forces
   are read into a `SinglePointCalculator`, not into `atoms.info`. `atoms.info.get("energy")` is
   therefore None for every frame, so the job prints "loaded 0 labelled structures", trains for
   zero steps, and ships the foundation model as if it had been fine-tuned. *Fix:* read the
   labels with `atoms.get_potential_energy()` and `atoms.get_forces()`, or store them under
   explicit keys (`REF_energy`, `REF_forces`, as MACE does), and fail loudly when nothing
   loads.

2. **Element embeddings are picked by position** (`adapt`). The foundation's embedding has one
   row per element of its table, sorted by Z: H, He, Li, Be, and so on. `weight[:len(z_table)]`
   keeps rows 0 to 3, so H keeps its own row, but C now uses helium's embedding, O lithium's and
   Mg beryllium's. The pretrained knowledge of C, O and Mg is gone, and nothing errors. *Fix:*
   `weight[[foundation_z_table.index(z) for z in z_table]]`, or keep the full table.

3. **The backbone runs under `torch.no_grad()`** (`energy_and_forces`). "Frozen" here also cuts
   the path from the energy back to the positions: the energy came out as 6.78 and every force
   as exactly 0. `allow_unused=True` with the zeros fallback hides the error that would have
   said so. Training can't fit forces, and MD or relaxations with this calculator see zero
   forces. *Fix:* freeze with `requires_grad_(False)` on the parameters, keep the forward pass
   under autograd, and remove `allow_unused` and the fallback.

4. **The calculator returns stale results** (`FinetunedCalculator.calculate`).
   `self.atoms = atoms` stores a reference to the caller's Atoms, so ASE's change check compares
   the object with itself and never sees a change. After every atom was moved,
   `get_potential_energy()` still returned 9.6309; a fresh calculator gave 8.6377. Every MD step
   after the first would reuse the first step's forces. *Fix:* call
   `super().calculate(atoms, properties, system_changes)` first, which stores a copy.

5. **`relax()` accumulates gradients.** `energy.backward()` adds into `pos.grad` at every step
   and nothing clears it, so each step uses the sum of all previous gradients: 1.0×, 2.05×,
   3.15× and 4.33× the true gradient over the first four steps. On a simple quadratic it never
   converged: after 500 steps it was exactly as far from the minimum as at the start. It also
   accumulates gradients into the model's parameters. *Fix:*
   `forces = -torch.autograd.grad(energy, pos)[0]`, or ASE's BFGS or FIRE with the calculator.

## Major

6. **Freezing with a plain attribute** (`adapt`). `block.requires_grad = False` sets an
   attribute on the module; its parameters still require gradients (True for the embedding and
   every interaction block in the check). Today the `no_grad` in 3 hides this. Once it's fixed,
   the whole foundation model trains on 1,200 configurations of one MOF, the opposite of what
   the PR intends. *Fix:* `block.requires_grad_(False)`, and build the optimiser from the
   parameters that still require gradients.
7. **Reference energies fitted by least squares on one framework.** Every structure is the same
   framework plus n CO2 molecules, so the composition matrix has rank 2 for 4 elements, and
   `lstsq` returns one arbitrary solution out of infinitely many. It reproduces the training
   energies, and it got CO2 right (CO2 is a direction present in the data). But the
   per-element values were off by up to 1 eV (Mg −2.64 against −1.6), so anything with another
   stoichiometry is shifted: H2O came out at −10.24 eV instead of −11.70. *Fix:* isolated-atom
   reference energies at the r2SCAN settings, or the foundation's values plus a fitted shift.
8. **`avg_num_neighbors` recomputed on the new data.** The foundation's normalisation (30.0) was
   replaced by 22.4 from the fine-tuning set, which scales every message in every pretrained
   layer by about 1.34 before training even starts. *Fix:* keep the foundation's value, as MACE
   does when fine-tuning.

## Minor

9. **float32 for relaxations.** It's fine for MD, but tight `fmax` thresholds and
   finite-difference checks need float64.
10. **The new element table and reference energies live only on the in-memory model.** The
    checkpoint must store them, or the calculator can't be rebuilt.
11. **`relax()` keeps the neighbour graph fixed** while atoms move, so atoms that come within the
    cutoff are ignored. Rebuild the graph every few steps.
12. **No batching and no validation split.** Split MD snapshots by trajectory segment, not at
    random, because neighbouring frames are nearly identical.

## Red herrings (say why they're fine)

- **Passing every parameter to Adam, frozen ones included,** is fine once freezing works:
  frozen parameters keep `grad = None`, and Adam skips them.
- **Reference energies cancel in adsorption energies** when the atoms are conserved
  (MOF+CO2 − MOF − CO2). That's why the CO2 numbers in 7 still look right.
- **float32 for MD** is fine; MACE recommends it for MD.
- **`create_graph=training`** is right: it's True in the training loop and False in the
  calculator.
- **`CUTOFF = 6.0`, the same as in pretraining,** is right. Changing it would invalidate the
  pretrained radial functions.

## Model 1-minute summary
"The job can finish without fine-tuning anything, and the calculator would give wrong dynamics
even if it did. The labels are read from `atoms.info`, where ASE no longer puts them, so zero
structures load. The element embeddings are sliced by position, so C, O and Mg get helium's,
lithium's and beryllium's. The frozen backbone runs under `no_grad`, which makes every force
exactly zero. The calculator stores a reference to the caller's Atoms, so it never notices that
atoms moved and returns stale results. And `relax()` accumulates `pos.grad`, so it never
converges. Behind those: freezing by attribute freezes nothing (masked by the `no_grad` for
now), the reference energies are fitted from a rank-deficient system, and the neighbour
normalisation is changed under the pretrained layers. I'd ask for a test that fine-tuning on
two structures changes the readout and nothing else, a finite-difference force check of the
calculator, and an MD run that checks the forces change between steps."
