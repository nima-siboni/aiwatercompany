# 11. Training at scale: DDP, mixed precision and resumable runs

Distributed, mixed-precision training code rarely crashes when it is wrong. Ranks drift apart,
metrics come from a quarter of the data, a resumed run warms up again, and every number looks
plausible. Ask what each rank, each loader worker and each dtype sees. Chapter 3 has the basics.

Outputs come from PyTorch 2.15 and NumPy 2.5 on Python 3.14 (CPU; DDP over gloo, 1 to 4 processes).

## Quick table

| You see | What goes wrong | Fix |
|---|---|---|
| every micro-batch in `model.no_sync()`, or training via `model.module.method()` | gradients are never all-reduced; the replicas drift apart | sync the last micro-batch; train through `model(...)` |
| `clip_grad_norm_` right after `scaler.scale(loss).backward()` | the clip sees gradients multiplied by the scale | `scaler.unscale_(opt)` first |
| `load_state_dict(..., strict=False)` on a DDP checkpoint | `module.` keys match nothing; the model stays random | save `model.module.state_dict()`; check the returned keys |
| `all_gather` then `[:len(dataset)]`; `err / len(loader.dataset)` | padded duplicates, round-robin order, one rank's sum over the full count | gather indices too; all-reduce a sum and a count |
| `.mean()` per micro-batch, then `/ accum` or `/ world_size` | small micro-batches weigh more; DDP already averages | sum, then divide by the window's global count |
| `plateau.step`, `continue` or `break` on a rank-local value | LRs differ and replicas diverge, or collectives hang | all-reduce the value first |
| a new scheduler, or numpy objects, in resume code | the warm-up restarts; `torch.load` fails (2.6+) | save every `state_dict` and the step; plain floats and dicts |
| `self.rng` set in `__init__`; no `set_epoch`; attributes set with `persistent_workers=True` | workers repeat each other; the same order every epoch; workers miss the change | a generator per worker; `set_epoch`; pass values through the sampler |
| `random_split` without `generator=` | the split moves when unrelated code changes | its own seeded generator, or saved indices |
| an EMA update written as `e = decay * e + ...` | rebinds a name; the EMA keeps its initial weights | `e.lerp_(p, 1 - decay)`; include buffers |
| `model.to(torch.bfloat16)`, or energies summed in bf16 | small updates round away; totals lose hundreds of eV | fp32 weights and fp32 sums |

## Examples

### Gradients that are never synced: `no_sync` and `model.module`

```python
for i, (x, y) in enumerate(loader):                 # model = DDP(net)
    with model.no_sync():                           # skip the all-reduce while accumulating
        (F.mse_loss(model(x), y) / accum).backward()
    if (i + 1) % accum == 0:
        opt.step(); opt.zero_grad()
# 2 ranks, 3 optimiser steps: the parameters differ by up to 0.11 between ranks
loss = F.mse_loss(model.module.predict(x), y)       # a named method instead of forward
# 2 ranks, 3 steps: the parameters differ by up to 0.097 (by 0 through model(x))
```

`no_sync` defers the all-reduce to the first backward outside it, and here there is none. DDP's
`forward` prepares the all-reduce, so calling the inner module skips it too. Nothing raises.
**Fix:** `ctx = model.no_sync if (i + 1) % accum else contextlib.nullcontext`; call `model(...)`.

### GradScaler and clipping

```python
with torch.autocast("cpu", dtype=torch.float16):
    loss = F.mse_loss(model(x), y)
scaler.scale(loss).backward()
clip_grad_norm_(model.parameters(), 1.0)     # returns 82485.61, the norm of the scaled gradients
scaler.step(opt); scaler.update()
# weight update norm 1.481e-06; with scaler.unscale_(opt) before the clip, 0.09702
```

The clip cuts the scaled gradients to norm 1, then `step` divides by the scale (65536), so
training crawls with no NaN to notice. `opt.step()` in place of `scaler.step(opt)` gives an update
norm of 8003, and a later `scaler.update()` raises "AssertionError: No inf checks were recorded
prior to update." **Fix:** `scaler.unscale_(opt)` before clipping.

### `module.` prefixes and `strict=False`

```python
torch.save(ddp_model.state_dict(), "ckpt.pt")    # keys: ['module.lin.weight', 'module.lin.bias']
net = Net()
net.load_state_dict(torch.load("ckpt.pt"), strict=False)
# _IncompatibleKeys(missing_keys=['lin.weight', 'lin.bias'],
#                   unexpected_keys=['module.lin.weight', 'module.lin.bias'])
# no exception, and net keeps its random initial weights
```

`strict=True` would raise "Missing key(s) in state_dict"; `strict=False` turns that into a return
value nobody reads. **Fix:** save `ddp_model.module.state_dict()` (or strip the prefix with
`torch.nn.modules.utils.consume_prefix_in_state_dict_if_present`), and assert on the returned keys.

### Distributed evaluation

```python
[list(DistributedSampler(range(10), num_replicas=4, rank=r, shuffle=False)) for r in range(4)]
# [[0, 4, 8], [1, 5, 9], [2, 6, 0], [3, 7, 1]]: padded to 12, dealt round-robin
preds = torch.cat([model(x) for x, _ in loader])        # this rank's share
out = [torch.zeros_like(preds) for _ in range(world)]
dist.all_gather(out, preds)
preds = torch.cat(out)[: len(val_set)]                  # samples 0, 4, 8, 1, 5, 9, 2, 6, 0, 3
mae = (preds - val_set.y).abs().mean()                  # 3.5, for a model that is exactly right
```

The slice drops sample 7 and keeps sample 0 twice. On 4 ranks, `err_sum / len(loader.dataset)`
gave 0.30 for a model whose error is always 1.0, and `all_reduce` (a sum) of per-rank means gave
4.0; `all_gather` with uneven sizes (per-atom outputs) aborted gloo with "gloo::EnforceNotMet".
**Fix:** gather indices, sort, de-duplicate; all-reduce sums and counts; pad or `all_gather_object`.

### Loss normalisation with gradient accumulation

```python
for i, (x, y) in enumerate(loader):          # micro-batches of 12 and 300 atoms
    loss = F.mse_loss(model(x), y) / accum   # a per-atom mean for each micro-batch
    loss.backward()
    if (i + 1) % accum == 0:
        opt.step(); opt.zero_grad()
# against one batch of all 312 atoms, the gradient has a relative error of 0.79
```

Each atom of the 12-atom micro-batch weighs 25 times one in the 300-atom micro-batch, and DDP
does the same across ranks; an extra `/ world_size` is wrong, since DDP already averages.
`zero_grad()` inside the loop keeps only the last micro-batch (half the gradient norm here).
**Fix:** `reduction="sum"` over the window's atom count (error 2.5e-07); under DDP, all-reduce
the count and multiply by the world size. Step the scheduler with the optimiser.

### Ranks that take different branches

```python
plateau.step(val_loss)          # rank-local: lr stays 0.1 on rank 0, drops to 0.05 on rank 1
# 3 steps later the replicas differ by up to 0.037
if not torch.isfinite(loss):    # a NaN on rank 1 only
    continue                    # RuntimeError: [...] Timed out waiting 10000ms [...]
if rank == 0:
    model(x_val)                # the DDP wrapper, around a net with BatchNorm
dist.barrier()                  # both ranks time out
```

DDP syncs gradients, never parameters, and a skipped step leaves the other rank waiting in an
all-reduce. DDP's `forward` broadcasts BatchNorm buffers from rank 0, so the wrapper on one rank
hangs (without BatchNorm it ran). The tests used a 10 s timeout; NCCL's default is 10 minutes.
**Fix:** all-reduce the value or a flag before branching; on one rank, evaluate `model.module`.

### Resuming a run: scheduler state and `torch.load`

```python
torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
            "best_mae": np.float64(0.12)}, "last.pt")
ckpt = torch.load("last.pt")
# _pickle.UnpicklingError: [...] Unsupported global: GLOBAL numpy._core.multiarray.scalar [...]
sched = LambdaLR(opt, warmup)          # a new scheduler starts at step 0
start_epoch = ckpt["epoch"]            # the last epoch that finished
# with float(best_mae): LR 0.1 before the restart, 0.001 after it
```

Since 2.6, `torch.load` defaults to `weights_only=True`, so numpy scalars and
`np.random.get_state()` fail at the first restart. A fresh `CosineAnnealingLR` resumed halfway
through 10 epochs ended at 0.025, not 0.0, and an unrestored GradScaler restarts at 65536.0.
**Fix:** save scheduler, scaler and EMA state and the step counter; resume at `epoch + 1`; store
`float()` and `bit_generator.state`. Under DDP pass `map_location` (a GPU is needed to show why).

### Data-loader workers: RNG, `set_epoch` and persistent workers

```python
class Augmented(Dataset):
    def __init__(self, data, seed=0):
        self.data, self.rng = data, np.random.default_rng(seed)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i] + self.rng.normal()
# 2 workers: epochs 0 and 1 both give [0.126, 0.126, -0.132, -0.132, 0.64, 0.64]
loader = DataLoader(train_set, num_workers=2, persistent_workers=True)
for epoch in range(3):
    train_set.noise_scale = float(epoch)     # the workers see 0.0, 0.0, 0.0
```

Each worker copies the dataset, generator included, and non-persistent workers copy the original
again every epoch; persistent ones keep their first copy, so later changes never arrive. Without
`set_epoch`, a `DistributedSampler` returned `[4, 7, 3, 0, 6]` every epoch. **Fix:** create the
generator in the worker (from `get_worker_info().seed`); pass per-epoch values via the sampler.

### `random_split` without a generator

```python
torch.manual_seed(0)
train, val = random_split(range(10), [0.8, 0.2])    # val indices [2, 6]
torch.manual_seed(0)
model = nn.Linear(4, 4)                             # one more user of the global RNG
train, val = random_split(range(10), [0.8, 0.2])    # val indices [1, 8]
```

The split moves whenever earlier code changes (a resume path that skips initialisation, say), and
validation samples leak into training. **Fix:** `generator=torch.Generator().manual_seed(seed)`.

### An EMA that never updates

```python
ema = copy.deepcopy(model)
@torch.no_grad()
def update_ema(decay=0.99):
    for e, p in zip(ema.parameters(), model.parameters()):
        e = decay * e + (1 - decay) * p     # rebinds the name e
# after 500 steps, ema still equals the initial weights: True
```

Every "EMA" result comes from the untrained model. An in-place update over `parameters()` alone
misses the BatchNorm buffers: eval MSE was 0.024 for the live model, 185.574 for that EMA and 0.045
with buffers copied. **Fix:** `e.lerp_(p, 1 - decay)`, include buffers, update per optimiser step.

### bf16 weights and energy sums

```python
model = model.to(torch.bfloat16)                    # halves the memory
# 100 SGD steps with lr * grad = 1e-3 leave a weight of 1.0 at 1.0, not 0.9
e_atom = model(batch)                               # 400 atoms, about -7 eV each
E = torch.zeros(1, dtype=e_atom.dtype).index_add_(0, batch.batch, e_atom)
# bf16: -2096.00 eV, fp16: -2794.00 eV, fp32: -2787.51 eV
```

bf16 has 8 significant bits (epsilon 0.0078): updates below about 0.4% of a weight vanish, and a
running sum stalls once the spacing between values passes the per-atom energy. CUDA autocast
runs `sum` in fp32 but not `index_add_` (per its op lists; a GPU is needed to show it).
**Fix:** fp32 weights under autocast; pool energies and compute losses in fp32.

## Looks wrong, is fine

- **`np.random` in `__getitem__` with several workers.** Version-dependent: fine since PyTorch
  1.9, which seeds NumPy per worker and per epoch (verified on 2.15); before 1.9 workers repeated.
- **A different `torch.manual_seed` on each rank before the model is built.** DDP broadcasts rank
  0's parameters and buffers when it wraps the model: a gap of 0.40 between ranks became 0.
- **No division by the world size after `backward()`.** DDP averages gradients: the gradient
  norm was 1.4261 for one process on the full batch and 1.4261 for two ranks on halves.
- **"Detected call of `lr_scheduler.step()` before `optimizer.step()`" with a GradScaler.** The
  scaler skips steps with inf gradients while it calibrates (the scale fell from 2³⁰ to 2²⁹).
- **No `torch.no_grad()` in validation when forces come from autograd.** Under `no_grad`,
  `autograd.grad` raised "element 0 of tensors does not require grad and does not have a grad_fn".
- **`torch.load(path)` with no `weights_only`.** Version-dependent: from PyTorch 2.6 the default
  is `weights_only=True`, which refuses arbitrary objects; before 2.6 the same line could run code.
