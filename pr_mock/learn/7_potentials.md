# 7. ML potentials: graphs, forces and stresses

An ML potential predicts one energy per structure and differentiates it for forces and stresses.
The bugs are rarely in the network. They sit in the bookkeeping around it: which atom receives a
message, which periodic image an edge points to, which structure a strain or an atom count
belongs to, and which sign and unit a label uses. The results have the right shapes and the wrong
values, often only for periodic cells or batches of several structures, which small tests skip.

Outputs come from PyTorch 2.15, NumPy 2.5 and ASE 3.29 on Python 3.14, on a CPU unless noted.

## Quick table

| You see | What goes wrong | Fix |
|---|---|---|
| `agg[dst] += msg` | each atom keeps one of its messages | `index_add_`, or `index_put_(..., accumulate=True)` |
| edges swapped to `(j, i)`, ASE's shifts `S` unchanged | periodic edges get the wrong length; molecule tests still pass | negate `S` when you swap |
| `autograd.grad(E, pos)` without `create_graph=True` in training, or under `no_grad` at inference | the force loss never reaches the weights; "element 0 of tensors does not require grad" | `create_graph=self.training`; `torch.enable_grad()` at inference |
| mean pooling for a total energy, sum pooling for an intensive target | a supercell changes the answer; forces from a per-atom energy are N times too small | sum for extensive, mean for intensive; forces from the total |
| one `eps` for the whole batch, or Cartesian shifts stored in the dataset | wrong stresses, while energies and forces stay right | one strain per structure; shifts from the strained cell |
| OUTCAR stress times `0.1 * units.GPa` | wrong sign, shear components in the wrong order | ASE's conversion; check the sign on a compressed cell |
| "force MAE", or "per atom" divided by `len(batch)` | two definitions 2× apart; the whole batch's atom count | name the definition; `torch.bincount(batch)` |
| `torch.zeros(n)` inside `forward` | float32 on the CPU, in a float64 or GPU model | `x.new_zeros(n)`, `torch.zeros_like` |
| `module.requires_grad = False`, or `strict=False` loading | nothing is frozen, or nothing is loaded, and nothing raises | `requires_grad_(False)`; strip key prefixes, keep `strict=True` |
| E0s, `avg_num_neighbors` or scale refitted on fine-tuning data | arbitrary references, rescaled pretrained layers | keep the foundation model's (or isolated-atom E0s at your DFT settings); map embeddings by Z |
| `E.backward()`, then `-pos.grad`, in a loop | forces accumulate across steps | `torch.autograd.grad(E, pos)` |
| float32 finite differences or energy comparisons | steps of 0.5 meV near 7000 eV | float64 |

## Examples

### Accumulating messages

```python
msg = torch.ones(6, 4)                           # one message per edge
dst = torch.tensor([0, 0, 0, 1, 1, 2])           # the atom that receives each message
agg = torch.zeros(3, 4)
agg[dst] += msg                                  # agg[:, 0] is tensor([1., 1., 1.])
torch.zeros(3, 4).index_add_(0, dst, msg)[:, 0]  # tensor([3., 2., 1.])
```

`agg[dst] += msg` gathers, adds and writes back, so repeated indices overwrite each other, and
PyTorch documents the result as undefined. NumPy's `a[idx] += 1` does the same. **Fix:**
`index_add_`, `index_put_(..., accumulate=True)` or `scatter_add_`; in NumPy, `np.add.at`.

### Edge vectors and image shifts

```python
i, j, S, d = neighbor_list("ijSd", atoms, 4.0)  # ASE: D = pos[j] - pos[i] + S @ cell
src, dst = j, i                                 # messages flow from neighbour j to centre i
D = pos[dst] - pos[src] + S @ cell              # i and j swapped, S kept
err = np.abs(np.linalg.norm(D, axis=1) - d)     # max 5.686 Å; 10 of the 36 edges are wrong
```

Swapping i and j reverses the vector, so the shift must flip too (here, a rattled 2-atom
rock-salt cell). Molecules have no shifts, so molecule tests pass. **Fix:** negate the shift,
`pos[dst] - pos[src] - S @ cell` (error 0.0), or keep ASE's `D` and pick where to aggregate.

### Forces through autograd

```python
F = -torch.autograd.grad(E, pos, retain_graph=True)[0]
loss = (E - E_ref) ** 2 + 100 * ((F - F_ref) ** 2).mean()
loss.backward()     # weight gradients equal those of (E - E_ref) ** 2 alone: True

@torch.no_grad()
def predict(model, pos):
    pos = pos.clone().requires_grad_(True)
    E = model(pos)
    return E, -torch.autograd.grad(E, pos)[0]
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

Without `create_graph=True`, F is a constant to the weights: the force term trains nothing but
still shows in the loss. Without `retain_graph` too, `backward` fails with "Trying to backward
through the graph a second time". **Fix:** `create_graph=self.training`. At inference, put the
force call in `torch.enable_grad()`; it works inside `no_grad`, not inside `inference_mode()`.

### Pooling: total or per atom

```python
e_atom = model(atoms, pos)           # per-atom energies, shape (N,)
E = e_atom.mean()                    # "energy per atom, comparable across cells"
F = -torch.autograd.grad(E, pos)[0]  # |F| / |F_true|: 0.25 for a 4-atom cell, 0.0312 for 2x2x2
# E = -2.1006 for both cells; sum pooling gives -8.4025 and -67.2204
```

Mean pooling hides the factor of 8, and forces from a per-atom energy are 1/N of the real ones.
Sum pooling an intensive target (uptake in mol/kg, a band gap) makes primitive and conventional
cells of one MOF disagree. **Fix:** sum and differentiate the total; mean for intensive targets.

### One strain per structure

```python
eps = torch.zeros(3, 3, requires_grad=True)             # one strain for the whole batch
sym = 0.5 * (eps + eps.T)
pos = pos + pos @ sym
cell = cell + cell @ sym                                 # (B, 3, 3)
E = model(pos, cell, edges, unit_shifts, batch)          # (B,)
stress = torch.autograd.grad(E.sum(), eps)[0] / volume.view(-1, 1, 1)
stress[:, 0, 0]      # [0.0328, 0.0342]; each structure on its own: [0.0239, 0.0093]
```

A shared strain collects the whole batch's gradient, and a batch of one hides it. Cartesian shifts
precomputed in the dataset ignore the strain: [0.0004, -0.0001], energies and forces exact.
**Fix:** `eps` of shape `(n_graphs, 3, 3)`, `sym[batch]` for atoms, shifts from the strained cell.

### Stress sign, units and Voigt order

```python
kbar = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])   # OUTCAR "in kB" line: XX YY ZZ XY YZ ZX
stress = kbar * 0.1 * units.GPa                       # kBar to eV/Å³:
# [0.00624 0.01248 0.01872 0.00062 0.00125 0.00187]
convert_vasp_outcar_stress(kbar)                      # what ase.io.read applies:
# [-0.00624 -0.01248 -0.01872 -0.00125 -0.00187 -0.00062]
```

VASP prints compression as positive; ASE flips the sign and orders shears yz, xz, xy. 1 eV/Å³
is 160.22 GPa, and the virial is -stress × volume. **Fix:** convert once with ASE and test the
sign: EMT Cu at a = 3.4 Å (compressed) gives -0.185 eV/Å³, and at 3.8 Å +0.100.

### Which force MAE?

```python
err = rng.normal(0.0, 0.05, size=(100_000, 3))   # force errors, eV/Å
np.abs(err).mean()                               # 0.0399: per component
np.linalg.norm(err, axis=1).mean()               # 0.0799: per atom, norm of the error vector
e_err = torch.tensor([0.10, 0.40])               # |ΔE| of two structures, 10 and 40 atoms
(e_err / len(batch)).mean()                      # 0.005 eV/atom: divided by all 50 atoms
(e_err / torch.bincount(batch)).mean()           # 0.01 eV/atom
```

The force MAEs differ by 2.0× and the RMSEs by √3 (0.0501 against 0.0867); both appear in papers.
A metric on magnitudes alone ignores direction: forces all rotated by 30° score 0.0. **Fix:** put
the definition in the name (`force_mae_per_component`), and count atoms per structure.

### New tensors inherit nothing

```python
e_atom = model(pos)                                     # float64 model, for MD
E = torch.zeros(n_graphs).index_add_(0, batch, e_atom)
# RuntimeError: index_add_(): self (Float) and source (Double) must have the same scalar type
E = torch.zeros(n_graphs)
E[0] += e_atom[:2].sum()       # no error: -7.323456764221191 is stored for -7.3234567890123
```

A strain made by `torch.zeros(B, 3, 3)` fails in the einsum: "expected scalar type Double but
found Float". On a GPU (tested on Apple MPS, in float32) line 2 fails: "self, index and source
expected to be in the same device, but got (self) cpu, ...". **Fix:** `e_atom.new_zeros(n_graphs)`.

### Freezing layers and loading checkpoints

```python
for block in model.blocks:
    block.requires_grad = False                  # PR: freeze the backbone
model.blocks[0].weight.requires_grad             # True; one Adam step (lr=1e-3) moves it by 0.001
sd = torch.load(path)                            # from a DDP model: keys start with "module."
res = model.load_state_dict(sd, strict=False)
len(res.missing_keys), len(res.unexpected_keys)  # (8, 8): not one tensor loaded
list(torch.compile(model).state_dict())[0]       # '_orig_mod.blocks.0.weight'
```

The loop sets a plain attribute on each Module and freezes nothing; `strict=False` hides every
mismatch in a return value. **Fix:** `requires_grad_(False)` and a logged trainable count (225
before, 9 after). Strip prefixes (`consume_prefix_in_state_dict_if_present`); keep `strict=True`.

### Fine-tuning statistics and element tables

```python
e0 = np.linalg.lstsq(counts, energies, rcond=None)[0]   # E0s fitted on MD frames of one MOF
np.linalg.matrix_rank(counts)                           # 1
e0                              # [-0.33 -1.06 -1.96 -0.98]; true: [-1.2 -1.9 -1.3 -1.1]
z_table = sorted(set(train_Z))                          # [1, 6, 8, 30]
emb = foundation_emb[: len(z_table)]                    # rows for H, He, Li, Be
```

One composition can't separate its elements, so other compositions get an arbitrary offset: the
E0 sum for CO2 is -4.08 eV, not -5.1 eV. Refitted `avg_num_neighbors` or scale and shift rescale
every pretrained layer. **Fix:** keep the foundation model's statistics and E0s; pick rows by Z.

### Relaxation loops and pos.grad

```python
pos = start.clone().requires_grad_(True)
for step in range(3):
    E = model(pos)
    E.backward()
    F = -pos.grad                      # |F| / |true force|: 1.0, 2.0, 2.999
    with torch.no_grad():
        pos += 1e-3 * F
```

`.grad` accumulates, so each step adds the new gradient to all the old ones, and `E.backward()`
also fills `.grad` on every weight. **Fix:** `F = -torch.autograd.grad(E, pos)[0]`.

### float32 and finite differences

```python
def energy(x):                                  # total energy of a large cell, eV
    return -7000.0 + 2.5 * (x - 1.2) ** 2       # the force at x = 1.0 is 1.0 eV/Å
x, h = torch.tensor(1.0), 1e-3
-(energy(x + h) - energy(x - h)) / (2 * h)      # 0.9766 in float32; 1.0 in float64
# with h = 1e-2: 1.001; with h = 1e-4: -0.0
```

Near 7000 eV, float32 values are 0.00049 eV apart, a quarter of the 0.002 eV difference being
measured. The autograd force is fine (1.0000002), but a line search comparing energies is not.
**Fix:** float64 for finite differences, relaxations and phonons.

## Looks wrong, is fine

- **`autograd.grad(E.sum(), pos)` over a batch.** Cross terms between structures are zero, so
  each atom gets its own structure's force (checked against per-structure gradients: equal).
- **An unsymmetrised strain.** For an invariant energy dE/dε is already symmetric (asymmetry
  1.6e-16 against entries of 0.84); MACE and NequIP symmetrise as a guard.
- **Frozen parameters left in the optimiser.** Version-dependent: since PyTorch 2.0, `zero_grad`
  sets them to `None` and AdamW skips them (moved 0.0; `set_to_none=False`: 2.3e-2 in 5 steps).
- **E0s left out of an adsorption energy.** Both sides hold the same atoms, so the E0s cancel
  (-0.4 eV with either set above). They matter only for absolute energies.
- **`torch.compile` on varying sizes.** Version-dependent: on PyTorch 2.15, `dynamic=None` compiled
  twice for 12 sizes (`dynamic=False`: 8, its limit). Training a force loss through it does fail.
