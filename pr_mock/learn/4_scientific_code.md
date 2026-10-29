# 4. Scientific code: units, numerics and periodic structures

Bugs in scientific code rarely raise an error. They produce numbers of the right type and the
wrong value. Outputs come from NumPy 2.5.

## Quick table

| You see | What goes wrong | Fix |
|---|---|---|
| constants without units, or mixed unit systems | factors of 1000, or of 10²³ | one system per module (eV, Å and K are usual); units in names |
| `np.cos(angle)` with angles from a CIF | CIF files store degrees | `np.radians` first |
| `np.exp(-E / kT)` with large \|E\| | overflow to inf, then NaN weights | subtract the minimum, or use `logsumexp` |
| `mean(x**2) - mean(x)**2` | cancellation; the "variance" can go negative | `np.var`, or Welford's method for streams |
| `==` on floats, or `np.isclose` with its defaults | the default `atol=1e-8` swamps small quantities | explicit tolerances, in the problem's units |
| totals compared across cells of different sizes | per-cell and per-atom values mixed up | normalise per atom, per formula unit or per gram |
| float32 for large totals | only about 7 significant digits | float64 for totals; take differences before casting down |
| chemical formulas parsed with a regex | repeated elements, parentheses, fractional occupancies, hydrates | pymatgen `Composition` |
| Python loops over atom pairs | O(N²) interpreter work | cell lists or KD-trees (ASE, pymatgen, matscipy) |

## Examples

### Units

```python
kB = 1.380649e-23                  # J/K
E = 0.10                           # eV
np.exp(-E / (kB * 300))            # 0.0: the exponent is about -2.4e19
kB_eV = 8.617333262e-5             # eV/K
np.exp(-E / (kB_eV * 300))         # 0.0209
```

Conversions worth knowing: 1 eV = 96.485 kJ/mol = 23.06 kcal/mol; kT at 300 K is 0.02585 eV;
1 bohr = 0.5292 Å; 1 mmol/g = 1 mol/kg. Pressure turns up in bar, Pa and atm, and uptake in
mmol/g, mol/kg and cm³(STP)/g. **Fix:** units in names (`energy_ev`, `pressure_bar`), and
constants from `scipy.constants` or `ase.units`.

### Degrees and radians

```python
gamma = 120.0                  # from a CIF, in degrees
np.cos(gamma)                  # 0.814: treated as radians
np.cos(np.radians(gamma))      # -0.5
```

### Overflow in Boltzmann weights

```python
E = np.array([-30.0, -29.9, -29.5])     # eV
kT = 0.025852
w = np.exp(-E / kT)
w / w.sum()                             # [nan nan nan]: overflow, then inf/inf
w = np.exp(-(E - E.min()) / kT)
w / w.sum()                             # [0.9795 0.0205 0.    ]
```

NumPy only prints warnings here. **Fix:** shift by the minimum, or use
`scipy.special.logsumexp` or `softmax`.

### Cancellation and float32

```python
x = 1e8 + np.array([0.1, 0.2, 0.3])
np.mean(x**2) - np.mean(x)**2          # 0.0; the true variance is 0.00667

E = np.array([-12345.6, -12345.8, -12346.1], dtype=np.float32)
(E**2).mean() - E.mean()**2            # -16.0: a negative variance
np.var(E)                              # 0.0422

np.spacing(np.float32(123456.789))     # 0.0078: nothing finer than ~8 meV at this magnitude
```

PyTorch defaults to float32. Keep large totals in float64, and compute differences (formation
or adsorption energies) before casting down.

### Tolerances

```python
np.isclose(1e-10, 2e-10)                         # True: the default atol=1e-8 dominates
np.isclose(1e-10, 2e-10, rtol=1e-6, atol=0.0)    # False
```

Choose tolerances in the problem's units (say, 1e-4 Å for positions and 1 meV for energies).
In tests, use `np.testing.assert_allclose` with explicit `rtol` and `atol`.

### Periodic boundaries

Start with the convention. In ASE and pymatgen, the **rows** of the lattice matrix are the
vectors a, b and c, so `cart = frac @ lattice`. Unless the code stores vectors as columns,
`lattice @ frac.T` gives a different, wrong vector:

```python
lattice = np.array([[10.0, 0, 0], [5.0, 8.66, 0], [0, 0, 12.0]])   # rows are a, b, c
frac = np.array([[0.5, 0.5, 0.0]])
frac @ lattice           # [[7.5, 4.33, 0.0]], which is 0.5a + 0.5b
lattice @ frac.T         # [[5.0, 6.83, 0.0]]: wrong
```

The usual nearest-image shortcut rounds the fractional difference:

```python
d = frac_j - frac_i
d -= np.round(d)
r = np.linalg.norm(d @ lattice)
```

This is exact only when the cutoff is **below half the cell's smallest perpendicular width**
(for an orthorhombic cell, half its shortest side). Beyond that it fails in two ways:

- **It picks the wrong image in skewed cells.** Take a cell with a = b = c = 10 Å and
  γ = 60°. Atoms at fractional (0, 0, 0) and (0.45, 0.45, 0) are 5.07 Å apart, but rounding
  reports 7.79 Å. The cell's perpendicular widths are 8.66, 8.66 and 10 Å, so rounding is only
  safe below 4.33 Å, and with a 6 Å cutoff this edge is missing.
- **It misses extra images.** In a small cell, an atom can see several copies of the same
  neighbour, and copies of itself. One image per pair misses them. Note that `i == j` with a
  zero offset is a self-edge to drop, while `i == j` with a non-zero offset is a real edge.

**Fix:** loop over image offsets, or use a neighbour-list library: ASE's `neighbor_list`,
pymatgen's `Structure.get_neighbor_list`, or matscipy. (`scipy.spatial.cKDTree` with `boxsize=`
handles only rectangular periodic boxes.) This matters for MOFs: many cells are
non-orthogonal, and some have one short axis (MOF-74's hexagonal cell has c ≈ 7 Å), so a
typical graph cutoff of 5–8 Å already breaks the shortcut.

### Scaling

A Python loop over atom pairs is fine for 50 atoms and unusable at 10,000 (10⁸ pairs). NumPy
broadcasting makes it fast but needs N² memory. Cell lists or KD-trees make it O(N). In a
review, say at what size it breaks: "fine for the 20-atom test fixture; at a few thousand atoms
the pair loop dominates".

### Per-atom versus per-cell

The same structure, as a 1×1×1 cell and as a 2×2×2 supercell, has total energies that differ by
a factor of 8. Ranking by total energy, or comparing it with a per-atom model output, is
silently wrong. Check what each energy column is normalised by.
