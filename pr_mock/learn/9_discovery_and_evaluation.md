# 9. The discovery loop: active learning, generative models and benchmarks

Code that picks the next candidates to simulate, or scores a generative model, rarely crashes.
Nothing says that the GP only saw its prior, that a "valid" molecule is an empty string, or that
a confidence interval has zero width. Most of these bugs are conventions: which direction is
better, what units a parameter is in, whether an index is a label, and what the denominator is.

Outputs come from Python 3.14, scikit-learn 1.9.1, SciPy 1.18, NumPy 2.5, pandas 3.1, RDKit 2026.09, pymatgen 2026.10 and PyTorch 2.15.

## Quick table

| You see | What goes wrong | Fix |
|---|---|---|
| a GP on raw descriptors (Å, m²/g) | the kernel underflows, most candidates get the prior, EI ties and `argmax` returns row 0 | scale X; anisotropic kernel; `normalize_y=True` |
| EI copied from a maximisation script into a minimisation loop | the worst candidates rank first | improvement is `best - mu` when minimising; test on a toy |
| `alpha=noise_var` with `normalize_y=True` | `alpha` is in standardised units, so the noise is scaled by `y.var()` | `alpha = noise_var / y.var()`, or a `WhiteKernel` |
| `df.drop(picks)`, or `np.delete` in a loop | labels are not positions; indices shift after each deletion | `.iloc`, one `np.delete` call, or a mask over IDs |
| the top-k of one acquisition vector as a batch, and `rng.choice(n, k)` as the baseline | k near-copies of one candidate; a baseline that repeats picks | joint batch acquisition; `replace=False` |
| `Chem.MolFromSmiles(s) is not None`, and sets of raw SMILES | `""` is a valid molecule; `OCC` and `CCO` count as different | reject empty molecules; canonicalise both sides with the same RDKit |
| `pdist(structure.cart_coords)`, or per-atom energies in `PDEntry` | periodic images are ignored; the hull expects the energy of the whole formula | `structure.distance_matrix`; multiply by `num_atoms` |
| `GroupKFold(groups=df.formula)` | `Fe2O3` and `Fe4O6` are different groups | group on `Composition(f).reduced_formula` |

## Examples

### A GP on unscaled inputs: EI ties, and `argmax` walks the file

```python
gp = GaussianProcessRegressor().fit(X, y)    # X: pore diameter (Å), surface area (m²/g)
gp.kernel_                                   # 0.318**2 * RBF(length_scale=1)
mu, sd = gp.predict(pool, return_std=True)
z = (y.min() - mu) / sd                      # y is minimised
ei = (y.min() - mu) * norm.cdf(z) + sd * norm.pdf(z)
(ei == ei.max()).sum(), np.argmax(ei)        # (1896, 0): 1896 of 2000 candidates tie
```

The RBF is exactly zero between MOFs whose surface areas differ by hundreds, so 62% of the pool
gets the prior mean of 0, which beats every observation; the length-scale stays at 1 even though
scikit-learn 1.9 optimises it. **Fix:** scale X, use an anisotropic RBF and `normalize_y=True`
(the picked MOF's true value fell from 0.193 to the pool's best, 0.101); `assert np.ptp(ei) > 0`.

### The sign of EI

```python
y = np.array([0.30, 0.12, 0.45])                # observed energies; lower is better
mu, sd = np.array([0.05, 0.60]), np.array([0.05, 0.05])
z = (mu - y.min()) / sd                         # the maximisation form
ei = (mu - y.min()) * norm.cdf(z) + sd * norm.pdf(z)
ei.round(3), ei.argmax()                        # ([0.002, 0.48], 1): picks the 0.60 candidate
```

Minimising needs `y.min() - mu`, which gives `[0.072, 0.0]`; negating `y` but not `best` gave 0
for both. **Fix:** one convention per code path, tested on a toy with a known minimum.

### `alpha` under `normalize_y`

```python
X, y = np.array([[0.0], [100.0]]), np.array([0.0, 10.0])   # y.std() is 5
kernel = ConstantKernel(1.0, "fixed") * RBF(1.0, "fixed")
gp = GaussianProcessRegressor(kernel, alpha=1.0, normalize_y=True).fit(X, y)   # noise var 1
gp.predict(X, return_std=True)[1]       # [3.536 3.536]; a noise variance of 1 gives 0.981
```

`alpha` is added after `y` is standardised, so it is in units of `y.var()`: 25 times the noise
here, and too little noise when `y.std()` < 1. **Fix:** `alpha = noise_var / y.var()`.

### Pool bookkeeping

```python
df = pd.DataFrame({"mof_id": [f"m{i}" for i in range(8)], "ok": [1, 0, 1, 1, 0, 1, 1, 1]})
pool = df[df.ok == 1]                  # index labels 0, 2, 3, 5, 6, 7
picks = np.array([2, 3])               # positions from argsort: m3 and m5
pool.drop(picks).mof_id.tolist()       # ['m0', 'm5', 'm6', 'm7']: dropped m2 and m3
pool.loc[[1, 3]]                       # KeyError: '[1] not in index'
ids = np.array([f"m{i}" for i in range(8)])
for i in [1, 3, 4]:                    # meant to remove m1, m3 and m4
    ids = np.delete(ids, i)
ids                                    # ['m0' 'm2' 'm3' 'm5' 'm7']: m3 stays, m4 and m6 go
```

On a six-row pool the loop raises `IndexError`. Labelled rows left in the pool get picked again
(a UCB pick on a 1-D toy). **Fix:** a mask over stable IDs, `.iloc`, and one `np.delete` call.

### Batch diversity and the random baseline

```python
x = np.linspace(0, 1, 1001)                                    # candidates on a 1-D grid
acq = norm.pdf(x, 0.3, 0.05) + 0.9 * norm.pdf(x, 0.7, 0.05)    # two peaks of similar height
np.sort(x[np.argsort(-acq)[:5]])                               # [0.298 0.299 0.3 0.301 0.302]
rng = np.random.default_rng(0)
len(np.unique(rng.choice(2000, size=500)))                     # 435: the random baseline
```

The batch spends five simulations on one peak, and the baseline repeats 65 of its 500 picks.
**Fix:** joint batch acquisition, or pick, penalise the neighbourhood, repeat; `replace=False`.

### Hyperparameters at cold start

```python
X = rng.uniform(0, 1, (5, 2))                    # five initial simulations
gp = GaussianProcessRegressor(normalize_y=True).fit(X, f(X))     # f runs the simulation
gp.kernel_                                       # 1**2 * RBF(length_scale=1e-05)
mu, sd = gp.predict(rng.uniform(0, 1, (1000, 2)), return_std=True)
np.ptp(mu), np.ptp(sd)                           # (0.0, 0.0): every candidate looks the same
```

The length-scale hit its bound in 11 of 20 starts, so EI is flat and `argmax` returns 0 (new in
scikit-learn 1.9). **Fix:** bounds or priors (0.05 to 2 gave 0.067 and 0.41); random picks first.

### SMILES validity, uniqueness and novelty

```python
gen = ["CCO", "OCC", "", "C1CC", "c1ccccc1", "C1=CC=CC=C1"]
mols = [Chem.MolFromSmiles(s) for s in gen]
sum(m is not None for m in mols) / len(gen)     # 0.833: "" gives a molecule with 0 atoms
len(set(gen)) / len(gen)                        # 1.0
[Chem.MolToSmiles(m) for m in mols]             # ArgumentError: ... MolToSmiles(NoneType)
canon = [Chem.MolToSmiles(m) for m in mols if m is not None and m.GetNumAtoms() > 0]
len(set(canon)) / len(canon)                    # 0.5
train = ["OCC", "C1=CC=CC=C1"]                  # stored as drawn, never canonicalised
len(set(canon) - set(train)) / len(set(canon))  # 1.0; 0.0 against a canonicalised train set
```

True validity is 0.667; MOSES's `get_mol` rejects `""` (read in its source, not run here).
`sanitize=False` accepts pentavalent carbon; stereo changes the string (`C[C@H](N)O`, `CC(N)O`).
**Fix:** canonicalise both sets with the same RDKit version and options; state the denominators.

### Crystal checks: periodic images and the convex hull

```python
s = Structure(Lattice.cubic(5.0), ["Na", "Cl"], [[0.01, 0, 0], [0.99, 0, 0]])
pdist(s.cart_coords).min(), s.distance_matrix[0, 1]     # (4.9, 0.1)
e_pa = {"Li": -1.90, "O": -4.90, "Li2O": -4.75}          # model output, eV/atom
entries = [PDEntry(Composition(f), e) for f, e in e_pa.items()]
PhaseDiagram(entries).get_e_above_hull(entries[2])       # 1.317; with total energies 0.0
e_above_hull, pred = np.array([0.0, 0.0, 0.012, 0.3]), np.array([-0.02, 0.01, -0.01, 0.2])
(e_above_hull < 0).sum(), f1_score(e_above_hull < 0, pred < 0)    # (0, 0.0), no warning
```

The `pdist` check passes two atoms 0.1 Å apart. `PDEntry` takes the energy of the formula you
pass, so Li2O lands 1.317 eV/atom above the hull instead of on it. Materials Project stores 0 for
stable entries. **Fix:** `s.distance_matrix`; `e * Composition(f).num_atoms`; `<= 0`.

### StructureMatcher: cost, defaults and order

```python
sm = StructureMatcher(scale=False)        # defaults: ltol=0.2, stol=0.3, angle_tol=5
a, b, c = nacl(5.0), nacl(5.5), nacl(4.5)                # rock salt, a = 5.0, 5.5 and 4.5 Å
sm.fit(a, b), sm.fit(a, c), sm.fit(b, c)                 # (True, True, False)
```

`fit` is not transitive, so a greedy unique count is 1 in the order a, b, c and 2 in b, c, a. It
is O(N²): about 6.5 ms per `fit` on 32-atom cells, so 10,000 × 10,000 pairs take about 180 hours.
**Fix:** bucket by reduced formula first; fix and report the order and tolerances.

### Deep ensembles and MC dropout

```python
members = []
for k in range(5):
    torch.manual_seed(0)                      # the same init and batch order for every member
    members.append(train(make_net(), loader))
torch.stack([m(x) for m in members]).std(0).max().item()       # 0.0
mu, sig = np.array([1.0, 1.4, 0.6, 1.2, 0.8]), np.full(5, 0.1)  # five members, one input
sig.mean(), np.sqrt(np.mean(sig**2 + mu**2) - mu.mean()**2)    # (0.1, 0.3)
mc_model.eval()
torch.stack([mc_model(x) for _ in range(30)]).std(0).max().item()   # 0.0: dropout is off
```

Distinct seeds gave a mean σ of 0.597. `train()` also switches BatchNorm to batch statistics
(a row moved from 0.024 to −0.058 with its batch), and `F.dropout(h, p)` ignores `eval()`.
**Fix:** seed `base + k`; σ² = mean(σₖ² + μₖ²) − μ̄²; `eval()`, then `train()` on dropout only.

### Coverage quantiles and ECE bins

```python
y = rng.normal(0.0, 1.0, 100_000)                     # errors of a calibrated model, sd = 1
[np.mean(np.abs(y) <= norm.ppf(p)) for p in (0.5, 0.9, 0.95)]     # [0.0, 0.801, 0.901]
idx = np.digitize(conf, np.linspace(0, 1, 11)) - 1   # confidence 1.0 gets index 10
np.nanmean([abs(correct[idx == b].mean() - conf[idx == b].mean()) for b in range(10)])  # 0.028
```

The one-sided quantile makes a calibrated model look overconfident, and the ECE saw only the 100
rows below 1.0. **Fix:** `norm.ppf(0.5 + p/2)`, `np.clip(idx, 0, 9)`, count-weighted bins (0.046).

### Bootstrap intervals

```python
boot = []
for _ in range(1000):
    rng = np.random.default_rng(0)                    # reseeded inside the loop
    idx = rng.integers(0, len(err), len(err))
    boot.append(err[idx].mean())
np.percentile(boot, [2.5, 97.5])                      # [0.0733 0.0733]
diff = lambda a, b, axis=-1: b.mean(axis) - a.mean(axis)
stats.bootstrap((err_a, err_b), diff, rng=0).confidence_interval      # (-0.0035, 0.015)
stats.bootstrap((err_a, err_b), diff, paired=True, rng=0).confidence_interval   # (0.0047, 0.007)
```

`replace=False` also gives zero width. Both models were scored on the same structures, so only
the paired interval shows the difference. `random.randint(0, n)` includes n (631 of 1000
resamples raised `IndexError`). **Fix:** one generator outside the loop, and `paired=True`.

### Test-set reuse, metric slips and grouping keys

```python
t = ths[np.argmax([f1_score(e_test <= 0, p_test <= th) for th in ths])]   # tuned on test
r2_score(pred, y), r2_score(y, pred)                  # (-0.061, 0.699); MAE is symmetric
pearsonr(y, y + 0.3)[0] ** 2, r2_score(y, y + 0.3)    # (1.0, -0.525)
mean_absolute_percentage_error([0.0, 0.0, 1.1, 2.3], [0.05, 0.1, 1.0, 2.0])   # 1.69e+14
mean_squared_error(y, pred, squared=False)  # TypeError: ... unexpected keyword argument 'squared'
[Composition(f).reduced_formula for f in ["Fe2O3", "Fe4O6", "O3Fe2"]]   # all 'Fe2O3'
```

Over 200 test sets of 300 rows, tuned thresholds reported a mean F1 of 0.791 and scored 0.760 on
fresh data; the best of 10 seeds reports an MAE 6.7% too low. `GroupKFold` on raw strings put
Fe2O3 on both sides of every fold. **Fix:** tune on validation data; report the seed spread.

## Looks wrong, is fine

- **`GaussianProcessRegressor()` with the default kernel.** Version-dependent: scikit-learn 1.9
  fits it (`2.56**2 * RBF(length_scale=0.934)` on a toy); 1.8 keeps `1**2 * RBF(length_scale=1)`.
- **`Chem.MolToSmiles(mol)` without `canonical=True`.** Canonical by default: `OCC`, `C(O)C` and
  `[H]OC([H])([H])C` all give `CCO`; both benzene spellings give `c1ccccc1`.
- **`normalize_y=True` with one observation, or a constant target.** No division by zero; the
  fits predict 4.2 and 3.0.
- **MC dropout inside `torch.no_grad()`.** `no_grad` does not turn dropout off (spread 0.161).
- **`np.var(member_means)` with `ddof=0`.** That is the 1/M term of the mixture variance.
- **StructureMatcher matching a supercell, or a copy with 10% more volume.** Intended: the
  defaults are `primitive_cell=True` and `scale=True`.
