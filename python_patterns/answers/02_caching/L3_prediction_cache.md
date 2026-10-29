# Answer key: L3_prediction_cache.py — Caching, level 3 (35 min + 5 min summary)

**Pattern focus:** a hand-rolled cache in a real service, where the key must capture
everything the value depends on (the model, not only the input), stay stable across
processes and never collide. Also check that callers can't change cached values through the
references they get, that failures aren't cached, that "LRU" really is LRU, and that the
cache is safe under threads and when it's written to disk.

**Good clarifying questions:** Does every model carry a stable `version` (the training run
ID), and does `reload_model` run while requests are in flight? Do callers ever modify the
arrays they get back? Which failures does the model raise in practice: transient (GPU out of
memory, timeouts) or bad input? How many request threads are there, and how big is an AL
round's working set compared with `max_entries`? Is the warm-start file per replica or
shared, and who else can write to that directory? Is `cache=` used to share one cache
between services?

## Blockers (silently wrong results / data loss)

1. **The model version isn't in the key, so a reload keeps serving the old model.**
   `reload_model` swaps `self.model` but the cache neither changes key nor clears. After the
   AL loop publishes run-42, every candidate already cached is still scored by run-41, and
   only new candidates see the new model. Rankings mix two models without any sign of it.
   Once the cache is a real LRU (issue 5), hot stale entries would never age out at all.
   *Fix:* put `model.version` in the key, and read `self.model` once per request so the key
   and the prediction come from the same model. Clearing on reload is weaker: a request
   still running on the old model can insert its result just after the clear.

2. **Cached arrays are handed out by reference, and `predict_ucb` edits them.**
   `pred[0] += beta * pred[1]` writes into the cached array, so each call for a candidate
   adds `beta * std` again: the UCB goes 1.45, 2.45, 3.45, ... The more often a candidate is
   scored, the better it looks, which is a feedback loop in the acquisition step, and
   `predict_one` callers now get an inflated mean too. Any other caller that edits a
   returned array in place (a unit conversion, a clip) corrupts what everyone is served.
   *Fix:* compute the UCB into a new value; store copies and return copies (or mark stored
   arrays read-only with `arr.flags.writeable = False`, so in-place edits raise).

3. **Failures are cached as None, for good.** When the model raises (a transient GPU
   out-of-memory error, a timeout), `predict_one` stores `None`, and `key in self.cache` is
   then True for it, so every later request for that candidate gets `None` for as long as the
   entry lives, and `shutdown` writes it to the warm-start file. The AL loop silently drops
   those candidates. *Fix:* don't cache failures. If you need negative caching, give it a
   short TTL and keep it distinguishable from a value. Use a sentinel so a lookup can tell a
   miss from a stored `None`.

## Major

4. **Check-then-act under threads.** `if key in self.cache: return self.cache[key]` is two
   steps. If another request thread inserts between them and evicts this key, `[]` raises
   `KeyError`, which shows up as sporadic 500s under load. `hits += 1` and
   `key_hits[key] += 1` lose updates. (In CPython with the GIL, each single `OrderedDict`
   call is atomic for these int keys, so the structure itself won't corrupt. The multi-step
   sequences are the problem.) Two threads that miss the same key both call the model, which is waste, not a
   bug. *Fix:* a lock around each cache operation, and one `get(key, sentinel)` instead of
   `in` followed by `[]`.

5. **The "LRU" is FIFO.** `__getitem__` never calls `move_to_end`, so entries are evicted in
   insertion order however often they're read. The AL loop re-scores the same top
   candidates every round, which are the hottest entries, and FIFO evicts them on schedule.
   *Fix:* `self._data.move_to_end(key)` on every hit.

6. **`hash()` keys don't survive a restart.** Hashes of `str` and `bytes` are salted per
   process (`PYTHONHASHSEED`), so after a restart none of the warm-start keys match. The
   warm start loads up to 200k dead entries, gets no hits, and saves them again. Because
   only the 64-bit hash is stored, a collision would also silently return another input's
   prediction: unlikely, but impossible to detect. *Fix:* a SHA-256 of a canonical encoding
   (shape, dtype, bytes, conditions), stored as the key itself.

7. **`key_hits` grows forever.** It's bounded by nothing, not even `max_entries`: there's
   one counter for every key ever hit, about 80 bytes each, and evicted keys are never
   removed. That's roughly 80 MB per million distinct candidates, and `stats()` scans all
   of it (`most_common`) on every dashboard poll. *Fix:* delete a key's counter when the key
   is evicted, or use a bounded top-k structure.

8. **`predict_batch` calls the model once per row.** A 50k-candidate AL round makes 50k
   model calls with a batch size of 1 (plus 50k hash-and-lookups) instead of one batched
   call for the misses. On a GPU model that's typically minutes instead of seconds. *Fix:* look up all
   the keys, then call the model once with the rows that missed.

9. **An interrupted save stops the next start-up.** `open(path, "wb")` empties the previous
   good file before writing. If the write fails (a full disk, or the process killed
   mid-write), the file is left truncated. `__init__` loads it without a `try`, so the next
   start raises `UnpicklingError` and the service won't start until someone deletes the
   file. *Fix:* write to a temp file in the same directory and `os.replace` it over the old
   one; treat an unreadable warm-start file as empty (log it and carry on).

## Minor

10. **The key ignores shape and dtype.** `tobytes()` is the raw buffer, so a (2, 3) and a
    (3, 2) array of zeros get the same key, and so do a float32 `zeros(2)` and a float64
    `zeros(1)`. The same values in float32 and float64 never share an entry, either. With
    one fixed-length float64 vector per structure this won't bite today, but it's fragile.
    *Fix:* convert to float64 and C order, and include the shape in the key.
11. **`cache or PredictionCache()` throws away an empty cache.** `PredictionCache` defines
    `__len__`, so an empty one is falsy. A cache passed in to be shared, or one configured
    with a bigger `max_entries`, is silently replaced by a new default. *Fix:*
    `cache if cache is not None else PredictionCache()`.
12. **`load` bypasses the size limit.** `self._data.update(...)` adds every saved entry,
    and `__setitem__` only pops one entry per insert. If `max_entries` was lowered since
    the file was written, the cache stays over its limit for good: loading 100 entries into
    a `max_entries=10` cache leaves it at 100. *Fix:* load through the normal insert path,
    and evict in a `while` loop.

## Red herrings (say why they're fine)
- **`popitem(last=False)` is the correct eviction call.** It pops from the front of the
  `OrderedDict`, which holds the oldest entry. The bug is the missing `move_to_end` on hits
  (issue 5), not the eviction.
- **`pickle` for the warm-start file is fine** when only this service writes the file, in a
  directory only it can write. Flag it if the path could ever come from a user or be shared
  with other teams, because unpickling can run arbitrary code.
- **Keying on the feature vector rather than a structure ID is a good choice.** Generated
  candidates without IDs work, and a re-featurised structure automatically gets a new key.
  What's missing is the model version (blocker 1).

## Model 1-minute summary
"Three blockers. The key doesn't include the model version, and `reload_model` doesn't
invalidate anything, so after every retrain the AL loop ranks old candidates with the old
model and new ones with the new model. Second, cached arrays are returned by reference, and
`predict_ucb` adds beta times std into the cached array, so a candidate's UCB grows every
time it's scored. Third, failures are cached as None, so one transient GPU error drops a
candidate for good. Then robustness: `in` followed by `[]` raises KeyError when another
thread evicts in between, the 'LRU' is really FIFO, `hash()` keys make the warm start
useless after a restart, a failed save truncates the file so the service can't start, and
the per-key stats grow forever. And `predict_batch` should call the model once for all the
misses. I'd key on the model version plus a SHA-256 of the input, store copies, add a lock,
and write the file atomically."

## Reference fix
`L3_prediction_cache_fixed.py`:
- `make_key` is a SHA-256 of shape, float64 bytes and conditions; the service prefixes it
  with `model.version`, reading `self.model` once per request.
- `PredictionCache` has one lock, an atomic `get` with a sentinel, `move_to_end` on hits,
  stores and returns copies, and drops `key_hits` entries on eviction.
- Failures return `None` without being cached; `predict_ucb` computes a new value.
- `predict_batch` makes one model call for all the misses.
- `save` writes a temp file and `os.replace`s it; `load` inserts through `__setitem__`, so
  the size limit holds; an unreadable warm-start file is logged and ignored.
- `cache if cache is not None else PredictionCache()`.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_02_caching.py -k L3 -v`
(Issue 4 is checked by forcing the eviction between the two steps with a patched dict, not
with real threads. The lost counter updates and the duplicate model calls aren't checked.)
