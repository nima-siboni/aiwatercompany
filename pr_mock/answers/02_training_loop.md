# Answer key: 02_training_loop.py

**Good clarifying questions:** What dtypes and shapes do `X_*`/`y_*` have (numpy float64?
y of shape (N,) or (N, 1)?)? Does this ever run on CPU-only machines or CI? How big is the
validation set? Is the returned `history` used for plotting or logging?

## Blockers

1. **Missing `opt.zero_grad()`.** Gradients accumulate across every step, so training
   diverges or behaves erratically, and nothing errors. *Fix:* call `opt.zero_grad()`
   before `loss.backward()` (`set_to_none=True` is the default).

2. **Shape broadcasting in the loss.** `pred` is (B, 1). If `y` is (B,), as it is when it
   comes from `df[col].values`, then `MSELoss` broadcasts to (B, B) and compares every
   prediction with every target. The model learns roughly the mean. PyTorch only emits a
   UserWarning, which is easy to miss in logs. The same problem is in `evaluate`. *Fix:*
   `pred.squeeze(-1)` or `y.view(-1, 1)`, plus an `assert pred.shape == yb.shape`.

3. **`best_state = model.state_dict()` stores references, not a copy.** The tensors in
   that dict are the live parameters, so they keep changing as training continues, and
   `load_state_dict(best_state)` at the end just restores the *final* weights. Early
   stopping silently does nothing. *Fix:*
   `best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}`
   (or `copy.deepcopy`).

4. **`evaluate` runs with dropout on.** There's no `model.eval()`, so the validation loss
   is noisy and biased, and model selection and early stopping are driven by noise.
   *Fix:* `model.eval()` inside `evaluate`, and switch back with `model.train()` at the
   start of each epoch (that call is already there).

## Major

5. **No `torch.no_grad()` in `evaluate`, and loss tensors are accumulated.** Each epoch builds
   an autograd graph over the whole validation set. `total` stays a graph-attached tensor,
   and it's stored in `history` and as `best_val`, so memory grows every epoch. That
   eventually means OOM on a large val set. *Fix:* `with torch.no_grad():` and accumulate
   `loss.item()`.

6. **Biased validation average.** `total / len(loader)` averages per-batch means, which
   overweights a smaller last batch. *Fix:* sum `loss * batch_size`, then divide by
   `len(loader.dataset)`, or use `reduction="sum"`.

7. **dtype mismatch.** `torch.tensor(numpy_float64)` gives float64 tensors, while the model
   is float32, so the matmul raises a RuntimeError. It's loud, but it would fail the first
   real run. *Fix:* `torch.as_tensor(X, dtype=torch.float32)` (or `from_numpy(...).float()`).

8. **Crash when validation is NaN.** If the loss is NaN from the first epoch,
   `val_loss < best_val` is never true, `best_state` stays `None`, and `load_state_dict(None)`
   crashes after a long run. Also add a guard for non-finite loss and gradient clipping.

9. **Hard-coded `device = "cuda"`** breaks on CPU/MPS and in CI. Pass the device in or
   auto-detect it.

## Minor

10. **The logged "train loss" is only the last batch.** Track the epoch mean.
11. **No seeding**, and `shuffle=True` on the val loader adds nondeterminism (the loss
    itself is order-invariant, so it's a nit on its own).
12. **Off-by-one in patience.** `bad_epochs > patience` waits `patience + 1` epochs. Pick a
    convention and document it.
13. **The loss is in normalised/log space** (from the data prep), so the reported numbers
    aren't in physical units. Log the metric in real units (e.g. MAE in mmol/g) as well.
14. **Config and design:** hyperparameters are scattered across function arguments and the
    model's defaults; the function builds loaders, trains and selects all at once. Consider a
    config dataclass, separating loader creation, and returning `best_val` and the best epoch.

## Red herrings
- Formatting tensors with `f"{loss:.4f}"` works for 0-d tensors (PyTorch defines `__format__`),
  so it isn't a bug. Once you switch to `.item()`, it goes away anyway.
- `model.train()` inside the epoch loop is correct and needed; it's what undoes the
  `eval()` you're adding in `evaluate`.

## Model 1-minute summary
"Four issues mean this doesn't do what it says, and none of them error: no
`zero_grad`, the (B,1)-vs-(B,) broadcast in the loss, `state_dict()` aliasing that
turns early stopping into a no-op, and dropout left on during validation. Next, `no_grad`
and `.item()` in `evaluate` to stop the memory growth, then the dtype and device issues. For
tests: overfit a tiny batch to near-zero loss, and assert that the returned weights equal
the best epoch's weights."
