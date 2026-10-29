# Answer key: 17_distributed_training.py

**Good clarifying questions:** Has this run end to end on more than one GPU, and do the
single-GPU and 8-GPU loss curves agree? How are jobs pre-empted, and has a resume been tested?
Who loads `checkpoints/finetune.pt` afterwards, and how? Does the model have BatchNorm or other
buffers? How large is the validation set relative to 8 × the batch size?

The six blockers overlap in a confusing way: several of them make the replicas or the reported
numbers wrong on their own. Say that out loud, and rank them by what they break: training
itself (1–3), what we measure (4–5), and what we ship (6).

## Blockers

1. **`no_sync()` wraps every micro-batch** (training loop). `no_sync` is meant to skip the
   all-reduce on all but the *last* micro-batch of an accumulation window. Here it covers all of
   them, so gradients are never averaged across GPUs. Each rank trains its own replica on its
   own shard, nothing errors, and rank 0's weights are what get saved. In a 2-rank
   reproduction, the replicas differed by 0.255 after three steps (0.000 with the fix). *Fix:*
   `ctx = model.no_sync() if (i + 1) % cfg.accum else contextlib.nullcontext()`.

2. **Gradients are clipped while still scaled.** With a GradScaler, the gradients after
   `backward()` are multiplied by the loss scale (65,536 at the start). The clip saw a norm of
   76,559 where the true norm was 1.17, clipped it to 10 in scaled units, and `scaler.step` then
   divided by 65,536. The update was 7,655× smaller than it should have been, so training
   barely moves and no NaN tells you why. *Fix:* call `scaler.unscale_(optimizer)` before
   `clip_grad_norm_` (the pattern in the AMP docs).

3. **The EMA never updates** (`update_ema`). `e = decay * e + (1 - decay) * p` rebinds the loop
   variable; it doesn't change the tensor. In the check the model moved by 5.0 and the EMA
   still equalled its initial weights. Validation and `best_ema.pt` therefore always evaluate
   the pretrained starting model. *Fix:* `e.lerp_(p, 1 - decay)` (or `e.mul_(decay).add_(p,
   alpha=1 - decay)`), and copy or average the buffers too.

4. **Validation compares predictions with the wrong targets** (`evaluate`). `DistributedSampler`
   deals indices round-robin: rank 0 gets 0, 2, 4…, rank 1 gets 1, 3, 5…. It also pads to a
   multiple of the world size by repeating samples. Concatenating the gathered shards gives the
   order [0, 2, 4, 6, 8, 1, 3, 5, 7, 9], which is then compared with targets in dataset order.
   In the check, a *perfect* model scored an MAE of 2.00 on 10 samples. With padding, the
   `[:len(val_set)]` slice also drops real samples and keeps duplicates. Checkpoint selection
   runs on this number. *Fix:* gather the indices along with the predictions and sort, or gather
   per-sample errors and all-reduce a sum and a count.

5. **Each rank's sampler has its own seed, and `set_epoch` is never called.**
   `DistributedSampler` splits one shared permutation between ranks. With different seeds the
   ranks permute differently, so their shards overlap. Without `set_epoch`, every epoch reuses
   the same permutation. In the check (12 samples, 2 ranks), each epoch saw the same 9 samples,
   and 3 of the 12 were never trained on in any epoch. *Fix:* the same `seed` on every rank, and
   `train_loader.sampler.set_epoch(epoch)` at the start of each epoch.

6. **`load_for_screening` loads nothing.** The checkpoint holds `model.state_dict()` of the DDP
   wrapper, so every key starts with `module.`. Loading it into a bare `MOFPotential` with
   `strict=False` matches no key: in the check, missing keys were `['weight', 'bias']` and the
   weights didn't change. Screening then runs a randomly initialised potential, and nothing
   complains. *Fix:* save `model.module.state_dict()` (or the EMA's), load with
   `strict=True`, or at least check the returned missing keys.

## Major

7. **Resuming fails, so the pre-emption feature doesn't work.** `torch.load` defaults to
   `weights_only=True` since torch 2.6, and the checkpoint contains the `Config` dataclass. The
   load fails with "Unsupported global: GLOBAL __main__.Config was not an allowed global by
   default". Even once it loads, a resume:
   - doesn't restore the scaler or the scheduler;
   - restarts the EMA from the pretrained weights, because it is copied before loading;
   - repeats the saved epoch (`start_epoch = ckpt["epoch"]` instead of `+ 1`).
   *Fix:* save `dataclasses.asdict(cfg)`, save and restore every state dict, and start from
   `epoch + 1`.
8. **Every rank writes the same checkpoint files.** Eight processes call `torch.save` on the
   same path at the same time, so the file can end up interleaved or truncated, and a job
   killed mid-write leaves a broken checkpoint. *Fix:* save on rank 0 only, write to a temporary
   file and `os.replace` it, then `dist.barrier()`.
9. **The LR scheduler sees each rank's own loss.** `scheduler.step(epoch_loss)` uses the rank's
   local sum, so the ranks lower their learning rates at different epochs (0.0005 against 0.001
   in the check). DDP synchronises gradients, not parameters, so different learning rates make
   the replicas drift even after fix 1. *Fix:* all-reduce the epoch loss first, or drive the
   scheduler from the validation metric, which every rank shares.

## Minor

10. **`loss.item()` on every micro-batch** forces a GPU sync each time. Accumulate on the device
    and sync once per epoch.
11. **The loss averages per-micro-batch means**, so with mixed cell sizes an atom in a small cell
    weighs more than one in a large cell. Decide whether that's intended, and normalise by atom
    counts across the accumulation window if not.
12. **Small things:** the hard-coded `"checkpoints/best_ema.pt"` next to a configurable
    `cfg.ckpt`; no logging of the world size, effective batch or learning rate; and nothing
    checks that the dataset fits the 8 × 4 × 8 accumulation arithmetic.

## Red herrings (say why they're fine)

- **`torch.manual_seed(cfg.seed + rank)`** is fine. DDP broadcasts rank 0's parameters when it
  wraps the model, and different seeds per rank are what you want for dropout or augmentation.
  Only the *sampler* seed must match (5).
- **The optimiser is built before the DDP wrap.** DDP keeps the same parameter objects, so the
  optimiser still holds the right tensors.
- **No division by the world size.** DDP already averages the gradients; dividing again would
  be the bug.
- **`scaler.scale(loss).backward()` outside the `autocast` block** is what the AMP docs
  recommend.
- **No `torch.no_grad()` in `evaluate`:** the forces need autograd. The EMA copy is a plain
  module, so evaluating it on every rank triggers no DDP communication.
- **fp16 plus GradScaler** is the right choice on V100s, which don't support bf16.

## Model 1-minute summary
"Training, measurement and shipping are each broken. Training: `no_sync` covers every
micro-batch, so the GPUs never average gradients and each trains its own replica; the gradients
are clipped while still scaled, so updates are thousands of times too small; and the EMA never
changes, because the update rebinds a loop variable. Measurement: validation compares
round-robin-ordered predictions with targets in dataset order, so even a perfect model scores
badly, and per-rank sampler seeds without `set_epoch` mean a quarter of the data is never seen.
Shipping: `load_for_screening` loads nothing, because of the `module.` prefix and
`strict=False`. Then resume crashes on the dataclass under `weights_only`, every rank writes the
checkpoint, and the plateau scheduler sees per-rank losses. I'd ask for a 2-process CPU test
with gloo that checks the replicas stay identical and that the gathered validation of a perfect
model gives zero error."
