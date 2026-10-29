# Answer key: L1_parallel_featurise.py — Concurrency, level 1 (10 min)

**Pattern focus:** The canonical thread-pool pitfalls: pairing results with their inputs,
choosing threads or processes for the kind of work, and futures whose exceptions nobody
retrieves.

**Good clarifying questions:** Where does the 70 ms go: Python code, or a C library such as
numpy? How many cores does the machine have? What backs `FeatureCache` (a file, SQLite, a
service), and is `put` safe to call from many threads at once? Does anything read the cache
while a run is still going?

## Blockers (silently wrong results)

1. **Features are filed under the wrong structures.** `as_completed` yields futures in the order
   they *finish*, but `zip(ids, as_completed(futures))` pairs them with IDs in *submission*
   order. As soon as one structure takes longer than the next (a large CIF, a big unit cell),
   every vector after it is attributed to a different MOF. The surrogate then trains on
   features that belong to other structures, with no error, and the cache hands the same wrong
   labels to other jobs. Sizes vary, so this happens on every real run. *Fix:* keep a mapping,
   `future_to_id = {pool.submit(featurise, cif): sid for ...}`, and look up
   `future_to_id[future]`; or use `pool.map`, which returns results in input order.

## Major

2. **Threads don't speed up pure-Python CPU work.** The PR says the parser and descriptors are
   pure Python. On the default (GIL) build of CPython only one thread runs Python bytecode at a
   time, so 32 threads give about one core's throughput, and the hour stays an hour. I measured
   0.99× for 32 threads on a pure-Python loop. (3.13 and later also ship an optional
   free-threaded build where this would scale, but that isn't the default.) *Fix:* a
   `ProcessPoolExecutor` (the featuriser must then be a picklable, module-level function), or
   vectorise the descriptors with numpy. Keep threads for I/O-bound work.

3. **Cache writes are fire-and-forget.** `pool.submit(cache.put, sid, vec)` throws the future
   away. If `put` raises (disk full, the cache service down), the exception sits in a future
   nobody looks at: no log, no error, and "other jobs can reuse it" silently stops being true.
   *Fix:* call `cache.put` in the main loop with error handling, or keep the futures and check
   them.

## Minor

4. **The cache isn't filled "as soon as it's ready".** The executor's queue is first in, first
   out, and all 50k featurisation tasks were submitted up front. Every `cache.put` therefore
   waits behind every featurisation task that hasn't started yet, so nothing is cached until
   the last structure has been picked up. A run killed at minute 50 has cached nothing. *Fix:*
   write to the cache from the main thread as each result arrives (as in #3).

5. **The warning drops the exception.** `log.warning("featurisation failed for %s", ...)`
   records *that* a structure failed but not *why*, so you can't tell a parser bug from a
   malformed file. *Fix:* `log.warning(..., exc_info=True)` or `log.exception(...)`. Consider
   returning the failed IDs too, so the caller can retry them.

## Red herring (say why it's fine)
- **`failed.append(structure_id)` runs in worker threads without a lock.** `list.append` is
  atomic in CPython (on the free-threaded build, lists have their own internal locking), and the
  list is only read after the `with` block has waited for every task. The thread-safety
  question worth asking is about `cache.put`, not this list.

## Model 1-minute summary
"The blocker is the zip over `as_completed`. Results arrive in completion order, so after the
first slow structure every feature vector is filed under the wrong MOF, silently, and the cache
passes the wrong labels on to other jobs. A dict from future to ID fixes it. Second, threads
won't make this faster: the work is pure-Python CPU, and with the GIL 32 threads still use about
one core, so it needs a process pool or vectorised descriptors. Third, the cache writes are
fire-and-forget, so their errors vanish, and because they queue behind all the remaining
featurisation work, nothing is cached until the very end. Smaller: the failure warning drops the
exception. The unlocked `failed.append` from worker threads is fine."

## Reference fix
`L1_parallel_featurise_fixed.py`:
- `ProcessPoolExecutor` with a `future_to_id` dict; `featurise` must be a module-level function
  so that it can be pickled.
- Each result is cached in the main process as it arrives, and cache errors are logged.
- Failures are logged with `exc_info=True`.
- The checks swap the process pool for a thread pool, because the test fakes can't be pickled.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_10_concurrency.py -k L1 -v`
