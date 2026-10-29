# Answer key: L2_streaming_dataset.py — Generators, level 2 (20 min)

**Pattern focus:** iterables vs iterators. Anything consumed more than once (a dataset, once per
epoch) must hand out a fresh iterator from `__iter__`. Also the lifetimes of lazy things: a
generator that outlives the `with` that opened its file, a buffer that's never flushed, and
errors swallowed inside a generator where nobody sees them.

**Good clarifying questions:** How are the shards written: by date, by generation campaign,
by composition family? (That decides how much the shard order matters.) Who calls `len()`: a
progress bar, an LR schedule? Apart from a truncated last line, what bad lines do we expect?
Has the record schema ever changed between old and new shards?

## Blockers (silently wrong results / data loss)

1. **`__iter__` returns `self`, so the dataset can be iterated exactly once.** The pipeline is
   built once in `__init__`, `__iter__` hands back the same object every time, and an exhausted
   generator stays exhausted. In the documented usage, `mean_energy(ds)` uses up the only pass,
   so *every* epoch, the first included, trains on nothing: `for record in ds` ends immediately
   and the loop runs zero steps, with no error. (Without `mean_energy`, epochs 2 onwards are empty.)
   *Fix:* make it an iterable, not an iterator. Drop `__next__`, and have `__iter__` build a
   fresh pipeline on every call.

2. **The shuffle buffer is never flushed.** `_shuffle` only emits when the buffer is full, so
   when the input ends, the last `buffer_size - 1` records are still in `buf` and are dropped:
   9,999 records per pass, mostly from the last shards (older records have had more chances to
   leave the buffer). A dataset smaller than the buffer (a debug run, a validation split) yields
   nothing at all. *Fix:* after the loop, shuffle what's left and `yield from buf`.

3. **`except Exception: continue` silently drops anything that goes wrong.** It's meant for the
   truncated last line, but it also swallows `KeyError` (a renamed field), `ZeroDivisionError`
   (`n_atoms` of 0) and `TypeError` (a null energy). If the pipeline renames `energy` to
   `energy_eV` in new shards, every record of every new shard disappears, with no log line and no
   counter, and the model quietly trains on old data only. *Fix:* catch only
   `json.JSONDecodeError`, log it with `path:line`, and let anything else raise with the location.

## Major

4. **`.gz` shards crash with `ValueError: I/O operation on closed file`.** `open_shard` returns a
   generator expression from inside `with gzip.open(...)`. The `return` leaves the `with`, which
   closes the file before the first line is read, so the first gzipped shard kills the epoch (and
   `len()`). Tests on plain shards pass; production, where the newer shards are gzipped, crashes.
   The plain branch returns a file that nothing closes explicitly. *Fix:* make `open_shard` a
   generator: `with opener(path, "rt") as f: yield from f`. The file then stays open while it's
   read and closes when the generator finishes or is closed.

5. **`__len__` rereads all 40 GB on every call.** A progress bar or LR schedule that calls
   `len()` once per epoch adds a full extra pass of I/O per epoch; anything that calls it per step
   never finishes. *Fix:* compute it once and cache it, or better, have the job that writes the
   shards record their counts in a manifest.

6. **The shard order never changes, so the shuffle barely shuffles.** `sorted(glob(...))` gives the
   same order every epoch, and a 10k buffer only mixes records that are close together in the
   stream. A shard holds far more than 10k records (40 GB over 300 shards is ~130 MB each), so
   each stretch of the epoch comes almost entirely from one shard. If shards are written per
   campaign or per composition family, minibatches are homogeneous and gradients are biased. Ask
   how shards are written. *Fix:* shuffle the shard order every epoch (optionally interleave a
   few shards at a time).

## Minor

7. **`__len__` counts lines, not records.** It includes the truncated lines and everything
   `_records` drops, so it disagrees with iteration (7 vs 6 in my check), and anything that
   derives steps per epoch from `len()` is off. *Fix:* count what iteration yields.

## Nits
- `buf.pop(i)` is O(buffer size): about 0.6 µs per record at 10k on a laptop (less than
  `json.loads`), but ~7 µs at 100k. Swap the chosen element with the last one, then `pop()`.
- `mean_energy` divides by zero on an empty dataset, which #1 and #2 make easy to hit. Raise a
  clear error.

## Red herrings (say why they're fine)
- **A private `random.Random(seed)` is the right choice** (better than `random.seed`): the order
  is reproducible whatever else in the process uses `random`. I checked: reseeding the global
  `random` between two runs doesn't change the order.
- **Emitting a uniformly random element of the buffer is the standard buffered shuffle** (it's
  what `tf.data`'s `shuffle(buffer_size)` does). The bias here comes from the fixed shard order
  (#6) and the missing flush (#2), not from the pop.

## Model 1-minute summary
"Three blockers, all silent. First, `__iter__` returns self over a generator built once in
`__init__`, so the dataset can be iterated once: in the documented usage `mean_energy` uses up
that pass and training sees zero records in every epoch. It should be an iterable that builds a
fresh pipeline per `__iter__`. Second, the shuffle buffer is never flushed, so each pass loses
9,999 records and a small dataset yields nothing. Third, `except Exception: continue`, meant for
truncated lines, also swallows key errors, so a renamed field silently drops whole shards. Then
the gzip branch returns a generator over a file the `with` already closed, so the first .gz shard
crashes; `__len__` rescans 40 GB per call; and the shard order is never shuffled, so minibatches
come from one shard at a time."

## Reference fix
`L2_streaming_dataset_fixed.py`:
- `__iter__` builds a fresh pipeline on each call (no `__next__`) and shuffles the shard order
  with the dataset's own RNG.
- `_shuffle` flushes the shuffled remainder and removes elements in O(1).
- `_records` skips only undecodable lines (logged with `path:line`) and raises `ValueError` with
  the location for anything else.
- `open_shard` is a generator around `with`, for plain and gzipped shards.
- `__len__` counts records once and caches the result.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_04_generators.py -k L2 -v`
