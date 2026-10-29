# 3. ML code: NumPy, pandas, PyTorch and evaluation

You know the ML. This chapter is about what the mistakes look like in code: the classics as a
table, then the ones that are easy to read past. Outputs come from NumPy 2.5, pandas 3.1,
scikit-learn 1.9 and PyTorch 2.14.

## Quick table

| You see | What goes wrong | Fix |
|---|---|---|
| a scaler, imputer or PCA fitted before the split | test statistics leak into training | fit inside a `Pipeline`, on the training fold only |
| a random row split when one entity has several rows (a MOF at several pressures, polymorphs, near-duplicate structures) | near-copies of training rows in the test set; inflated scores | a group split (`GroupShuffleSplit`, `GroupKFold`) |
| the test set used for early stopping or model selection | an optimistic test score | separate validation and test sets |
| no `opt.zero_grad()` | gradients accumulate across steps | zero them every step |
| no `model.eval()` before validation | dropout and batch norm stay in training mode | `eval()` for validation, `train()` afterwards |
| no `torch.no_grad()` in evaluation | graphs are built and kept; memory grows | `@torch.no_grad()` or `torch.inference_mode()` |
| `total += loss` with a tensor | autograd history accumulates across batches, and memory grows | `total += loss.item()` |
| a mean of per-batch means | a small last batch gets too much weight | `reduction="sum"`, then divide by N |
| `X / X.std(0)` | a constant column gives 0/0 = NaN, with only a warning | guard zeros, or drop constant columns |
| `np.log(y)` | `log(0)` is `-inf`, with only a warning | `log1p`, or filter, and ask why zeros exist |
| unseeded `shuffle`, `sample` or splits | runs can't be compared | one seeded generator, and log the seed |
| `torch.tensor(np_array)` | a float64 tensor, then "mat1 and mat2 must have the same dtype, but got Double and Float" | `torch.as_tensor(x, dtype=torch.float32)` |
| a hard-coded `"cuda"` | a crash on a CPU-only machine | a `device` parameter |
| `torch.load(..., weights_only=False)` or `pickle.load` on files you didn't write | arbitrary code runs on load | keep the default `weights_only=True` (PyTorch 2.6+), or use safetensors |

## The reference training loop

Know the correct shape, and deviations stand out.

```python
import copy, math
import torch, torch.nn.functional as F

def train_epoch(model, loader, opt, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(xb).squeeze(-1), yb)        # (B,) against (B,)
        loss.backward()
        opt.step()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        pred = model(xb.to(device)).squeeze(-1)
        total += F.mse_loss(pred, yb.to(device), reduction="sum").item()
        n += len(yb)
    return total / n                                         # a per-sample mean

best, best_state, bad = float("inf"), None, 0
for epoch in range(max_epochs):
    train_epoch(model, train_loader, opt, device)
    val = evaluate(model, val_loader, device)
    if not math.isfinite(val):
        raise RuntimeError(f"non-finite validation loss at epoch {epoch}")
    if val < best:
        best, bad = val, 0
        best_state = copy.deepcopy(model.state_dict())      # a snapshot, not a reference
    else:
        bad += 1
        if bad >= patience:
            break
model.load_state_dict(best_state)
```

## Examples

### Broadcasting (N, 1) against (N,)

```python
pred = np.array([[1.0], [2.0], [3.0]])     # model output, shape (3, 1)
y = np.array([1.0, 2.0, 3.0])              # targets, shape (3,)
(pred - y).shape                           # (3, 3)
((pred - y) ** 2).mean()                   # 1.333, although every prediction is exact
```

`F.mse_loss` with the same shapes also returns 1.333, and only warns: "Using a target size
(torch.Size([3])) that is different to the input size (torch.Size([3, 1])). This will likely
lead to incorrect results due to broadcasting." Trained on that loss, a model learns to predict
the mean target, and it looks as if it's just underfitting. **Fix:** `squeeze(-1)`, and
`assert pred.shape == y.shape`.

### Views and live references

```python
a = np.arange(6).reshape(2, 3)
col = a[:, 0]           # basic slicing returns a view
col *= 10               # a is now [[0, 1, 2], [30, 4, 5]]
b = a[a > 20]           # boolean or fancy indexing returns a copy
b *= 0                  # a is unchanged
```

```python
best = model.state_dict()       # tensors that share storage with the live parameters
opt.step()                      # updates the parameters in place, so `best` changes too
model.load_state_dict(best)     # "restores" the current weights, not the best ones
```

**Fix:** `copy.deepcopy(model.state_dict())`.

### pandas aligns on the index

```python
df = pd.DataFrame({"mof_id": ["A", "B", "C", "D"], "sa": [400, 900, 1200, 800]})
df = df[df.sa > 500]                       # the index is now 1, 2, 3
df["pred"] = pd.Series([0.1, 0.2, 0.3])    # this Series has index 0, 1, 2
#   mof_id    sa  pred
# 1      B   900   0.2     <- shifted by one row
# 2      C  1200   0.3
# 3      D   800   NaN
```

Assigning a Series matches rows by index label, not by position. **Fix:** assign a NumPy array
(`df["pred"] = preds`), or `reset_index(drop=True)` first. Arithmetic between two Series and
`pd.concat(axis=1)` align the same way.

### Merges that duplicate rows

```python
props = pd.DataFrame({"mof_id": ["A", "B"], "sa": [900, 1200]})
labels = pd.DataFrame({"mof_id": ["A", "A", "B"], "uptake": [2.1, 2.3, 4.0]})
props.merge(labels, on="mof_id")                         # 3 rows: A appears twice
props.merge(labels, on="mof_id", validate="one_to_one")
# MergeError: Merge keys are not unique in right dataset; not a one-to-one merge
```

A duplicated key gives that entity double weight, and a row split can put one copy in train
and the other in test. **Fix:** state the expected cardinality with `validate=`, and aggregate
duplicates before merging.

### `iterrows` changes types, and is slow

```python
df = pd.DataFrame({"n_atoms": [52], "energy": [-310.2]})
row = next(df.iterrows())[1]
row["n_atoms"]                   # np.float64(52.0): the row is one Series with one dtype
range(row["n_atoms"])            # TypeError: 'numpy.float64' object cannot be interpreted as an integer
next(df.itertuples()).n_atoms    # 52: itertuples keeps each column's type
```

**Fix:** vectorise (a merge, a groupby, `map`). If you must loop, use `itertuples`.

### NaN fails every comparison

```python
x = np.array([1.0, np.nan, 3.0])
x[x > 2]                          # [3.]: the NaN row is dropped
x[~(x <= 2)]                      # [nan, 3.]: "the same filter" keeps it
max([float("nan"), 1.0, 2.0])     # nan
max([1.0, float("nan"), 2.0])     # 2.0
np.mean(x), pd.Series(x).mean()   # nan, 2.0: pandas skips NaN by default, NumPy doesn't
```

In early stopping, `val < best` is False forever once the loss turns NaN, so the checkpoint
silently stops updating. **Fix:** check `math.isfinite` and fail fast.

### Integer arrays truncate on assignment

```python
occupancy = np.array([1, 1, 1])      # int64
occupancy[0] = 0.5                   # stored as 0, silently
occupancy += 0.5                     # this one raises UFuncTypeError
np.array([-1.7, 1.7]).astype(int)    # [-1, 1]: truncation toward zero
```

### Averaging batch means

```python
batch_sizes = [100, 100, 1]
batch_means = [1.0, 1.0, 10.0]
sum(batch_means) / 3                                           # 4.0
sum(m * n for m, n in zip(batch_means, batch_sizes)) / 201     # 1.04, the real mean
```

### Leakage that matters, and leakage that's only a smell

Fitting a `StandardScaler` on all rows is a real correctness issue, but its effect is usually
small. The leaks that change conclusions are near-duplicates across splits (one framework at
several pressures, one structure from two CIF files, polymorphs), features derived from the
target, and choosing models on the test set. Saying this out loud is good prioritisation.

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(X, y, groups=df["mof_id"]))
assert set(df.mof_id.iloc[train_idx]).isdisjoint(df.mof_id.iloc[test_idx])
```

Also ask what the baseline is (predicting the training mean), whether errors are reported in
the units that matter (mmol/g rather than log space), and how much they vary across seeds.
