# Answer key: L2_training_callbacks.py — Observer, level 2 (20 min)

**Pattern focus:** callbacks are observers of the training loop. Check who owns the payload
(`logs`) and whether observers keep references to it, which state outlives a run, which hook
each callback uses, and when the loop actually reads the flags that callbacks set.

**Good clarifying questions:** Is the same trainer, with the same callback objects, reused
in every active-learning round? (The PR says yes.) Which metric should choose the "best"
checkpoint? How many batches per epoch, and how big is the real model? Does anything read
`History`, such as plots or AL diagnostics?

## Blockers (silently wrong training or model selection)

1. **One `logs` dict for the whole run, and `History` keeps references to it.** `fit` creates
   `logs` once and overwrites its keys on every batch and epoch. `History.on_epoch_end`
   appends that same object, so after 3 epochs `history.epochs` holds three references to one
   dict with the final epoch's numbers. Learning curves come out flat, and anything that
   picks an epoch from the history is wrong. Batch callbacks in epoch 2 also see the previous
   epoch's `val_loss`. *Fix:* build a fresh dict per batch and per epoch in `fit`, and have
   `History` store `dict(logs)`. An observer shouldn't keep a payload it doesn't own.

2. **`stop_training` is never reset.** It's set once, in `__init__`. After early stopping ends
   round 1, every later `fit` on the same trainer (the AL loop's next rounds) runs one epoch
   and stops. From round 2 on the surrogate barely trains, and nothing errors. *Fix:*
   `self.stop_training = False` at the start of `fit`.

3. **`ModelCheckpoint` picks "best" by single-batch training loss.** It hooks `on_batch_end`
   and compares `logs["loss"]`, the loss on one batch of training data. The weights it keeps
   are whichever happened to hit an easy batch, often late in training and overfit, rather
   than the weights with the best validation loss, and the AL loop reloads them. *Fix:* hook
   `on_epoch_end` and monitor `val_loss` (add a `monitor` argument).

## Major

4. **`EarlyStopping` ignores `mode`.** It always tests `value < self.best`. With
   `EarlyStopping(monitor="val_r2", mode="max")`, an R² rising through 0.1, 0.2, 0.3 counts as
   "no improvement", so training stops after `patience` epochs while still improving, and
   `best` holds the worst value. A typo such as `mode="maximum"` is silently accepted.
   *Fix:* compare according to `mode`, start `best` at +inf or -inf to match, and validate
   `mode` in `__init__`.

5. **Callback state isn't reset between fits.** `EarlyStopping.best` and `wait`, and
   `ModelCheckpoint.best`, are set in `__init__`, never in `on_train_begin`. When the
   callbacks are reused across AL rounds, round 2 is compared against round 1's best on a
   different validation set: early stopping fires after `patience` epochs even while the loss
   improves, and the checkpoint may never save, so the AL loop reloads round 1's weights.
   *Fix:* reset per-run state in `on_train_begin`, which is what that hook is for.

6. **Synchronous disk I/O inside the batch loop.** `np.save` runs whenever a batch beats the
   best batch loss so far, which is often (11 writes in the first 64 batches in the check).
   With the real model's weights and thousands of batches, the I/O dominates the run time.
   `np.save` also overwrites the only checkpoint in place, so a crash mid-write leaves a
   corrupt file. *Fix:* save at most once per epoch (#3), writing to a temp file and then
   `os.replace` it.

## Minor

7. **The stop flag is only checked after the epoch.** `TerminateOnNaN` sets `stop_training`
   in `on_batch_end`, but `fit` only checks it after the batch loop. The rest of the epoch
   keeps stepping with NaN weights, and the epoch-end callbacks run on NaN metrics. It wastes
   time rather than producing wrong results, since training does stop. *Fix:* check the flag
   after each batch's callbacks and break.

8. **`np.save` appends `.npy`.** `ModelCheckpoint("checkpoints/best")` writes
   `checkpoints/best.npy`, so reloading `cb.path` raises `FileNotFoundError`. *Fix:* the
   temp-file-and-rename from #6, renaming to exactly `self.path`.

## Red herrings (say why they're fine)
- **`self.callbacks = callbacks or []`** looks like the shared-mutable-default bug, but a new
  list is created whenever `callbacks` is None, so each trainer gets its own. A list that's
  passed in is used as given, which is what the caller expects.
- **No-op base hooks called on every callback for every batch.** An empty method call costs
  well under a microsecond, which is nothing next to a training step. Per-hook subscriber
  lists aren't worth the complexity.

## Model 1-minute summary
"Three things make this silently train or select the wrong model. `fit` reuses one `logs`
dict and `History` stores references to it, so the whole history shows the last epoch.
`stop_training` is never reset, so once early stopping fires, every later `fit` in the
active-learning loop runs a single epoch. And `ModelCheckpoint` chooses the 'best' weights by
single-batch training loss in `on_batch_end`, so the loop reloads an overfit snapshot. Then
the callbacks themselves: `EarlyStopping` ignores `mode`, so it stops early on max-mode
metrics, and neither it nor the checkpoint resets its state in `on_train_begin`, so round 2
is judged against round 1. The checkpoint also does synchronous, non-atomic disk writes inside
the batch loop. Smaller things: the stop flag is only checked after the whole epoch, and
`np.save` adds `.npy` to the path. `callbacks or []` is fine."

## Reference fix
`L2_training_callbacks_fixed.py`:
- `fit` resets `stop_training`, builds fresh `logs` dicts per batch and per epoch, and checks
  the flag after every batch.
- `History` stores a copy of each epoch's logs.
- `EarlyStopping` validates `mode`, compares according to it, and resets in `on_train_begin`.
- `ModelCheckpoint` hooks `on_epoch_end`, monitors `val_loss`, resets in `on_train_begin`,
  and writes atomically to exactly `path`.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_08_observer.py -k L2 -v`
