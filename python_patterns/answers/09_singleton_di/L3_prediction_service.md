# Answer key: L3_prediction_service.py — Singleton and dependency injection, level 3 (35 min + 5 min summary)

**Pattern focus:** In a service, module-level state is shared by every request on every thread
and lives as long as the process. Sort each global into "process-wide and read-only after
start-up" (fine: the model, the configured clients) or "request-scoped" (never a global: the
current user, this request's conditions). Then check initialisation order, fork safety, and what
tests have to do to swap a dependency.

**Good clarifying questions:**
- How does the platform server run this: threads per process, worker processes, replicas? Does
  it fork workers after the start-up hook (a pre-fork server with the app preloaded)?
- Do users from different projects share one deployment? If so, #1 is a data-isolation breach.
- Must job IDs be unique across restarts and replicas? What does `save_predictions` do with an
  ID that already exists?
- Can a structure change under the same ID (a re-relaxed geometry)? How many distinct structures
  does a screening campaign ask for?
- How big do requests get?

## Blockers (silently wrong results / data loss / security)

1. **Request state lives in a module global.** `_authenticate` stores the caller in
   `CURRENT_USER`, and `_require_project` reads it later. That fails in two ways, and both leak
   data across projects:
   - *Stale:* a request without an `Authorization` header doesn't touch `CURRENT_USER`, so it
     runs as whoever made the previous request in that process. A "no token gives 403" test
     passes on a fresh process, which is how this slips through.
   - *Shared:* alice's `handle_get_results` authenticates, then waits on
     `_store.load_predictions`. Meanwhile bob's request authenticates on another worker thread.
     Alice's `_require_project` now checks *bob's* projects, and alice receives bob's
     predictions.

   *Fix:* make the user a value. `_authenticate` returns a `User` (and raises when the header is
   missing), and the handler passes it to `_require_project`. If the framework needs ambient
   request context, use a `contextvars.ContextVar`, set at the start of each request and reset at
   the end: a pool thread runs many requests, so its context outlives any one of them.

2. **The model is published before its weights are loaded.** `get_model` assigns
   `_model = _model_factory()` and only then calls `_model.load_weights(...)`. A request that
   arrives during the load passes the unlocked `if _model is None` check and predicts with the
   model's initial, random weights. The uptakes look plausible, they're wrong, and the status is
   200. If the server only takes traffic after `warm_up()` returns, production may never hit
   this, but every lazy path does: a worker that skipped warm-up, a reconfiguration, a test. Lazy
   initialisation code has to be correct on its own. *Fix:* build and load into a local, then
   publish: `model = _model_factory(); model.load_weights(...); _model = model`.

3. **Request conditions are written into the module-level defaults.**
   `conditions = DEFAULT_CONDITIONS` doesn't copy, so `conditions.update(...)` mutates the
   global. After one request at 77 K, every later request in that process that doesn't send a
   temperature is predicted at 77 K, and the response even echoes the wrong conditions. One
   request with a misspelt key (`{"temp": 77}`) poisons the dict for good: if `predict` takes
   explicit keyword arguments, every later request fails with `TypeError` (a 500) until restart.
   Concurrent requests also overwrite each other's conditions mid-request. *Fix:*
   `conditions = {**DEFAULT_CONDITIONS, **given}`, reject unknown keys, and make the defaults
   read-only (`types.MappingProxyType`).

4. **Job IDs repeat across restarts and processes.** `_job_ids = itertools.count(1)` lives in
   process memory. Every restart starts again at `pred-000001`, and every worker process or
   replica issues the same sequence. A new prediction is then saved under an existing ID.
   Depending on `save_predictions`, that either overwrites another user's results or fails after
   the model has done the work, and a user fetching an old job ID gets someone else's
   predictions, or a 403 for their own job. Threads aren't the problem: on the default (GIL) build,
   `next()` on an `itertools.count` is atomic. *Fix:* `uuid.uuid4()`, or let the store allocate
   the ID.

## Major

5. **`warm_up()` opens the store connection, possibly before workers fork (ask).** It calls
   `_store.ping()`, which opens the store's connection in whichever process runs the start-up
   hook. If the server forks its worker processes after that (pre-fork with the app preloaded),
   every worker inherits the same socket or file handle. With a network database, two workers
   interleave messages on one socket and can read each other's responses. With SQLite, the
   SQLite docs warn that carrying an open connection across `fork()` can corrupt the database.
   *Fix:* open connections lazily in each worker (or in a post-fork hook), and keep `warm_up()`
   to the model. Loading the model before forking can share its memory between workers, but check
   with the platform team: a CUDA context, for one, can't be used in a forked child.

6. **`@functools.cache` on `_load_structure` grows without bound.** The cache is keyed by
   structure ID and never evicts. A long-running worker keeps every parsed structure any user has
   asked for, so a screening campaign over hundreds of thousands of hypothetical MOFs grows memory
   until the worker is OOM-killed. It never refreshes either: if a structure is re-relaxed under
   the same ID, the old geometry is served until restart (ask whether IDs are immutable). It
   also outlives `configure()` (#11). *Fix:* a bounded `lru_cache(maxsize=...)` that's cleared
   when the store changes; key by (ID, version) if structures can change.

7. **One model call per structure.** `predict` takes a batch (`predict([structure], ...)`), but
   the loop calls it once per structure. For a GNN, per-call overhead (building the graph batch,
   launching kernels) dominates small batches, so a 1,000-structure request takes many times
   longer than one batched call and holds a worker thread all that time. *Fix:* collect the
   structures and call `predict` once, in chunks if one batch doesn't fit in memory.

8. **No limit on request size.** `structure_ids` can be any length. One client sending 200k IDs
   ties up a worker for minutes, fills the unbounded cache (#6) and builds a huge response.
   *Fix:* cap it (`MAX_STRUCTURES_PER_REQUEST`), return 400 above the cap, and point bulk jobs at
   the batch pipeline.

## Minor

9. **The metrics lose updates under threads.**
   `METRICS["seconds"] += time.perf_counter() - started` reads the old value, calls
   `perf_counter`, then writes. On CPython 3.13 (default GIL build), a thread switch can happen
   at that call. In a quick experiment, 4 threads × 300k updates of
   `d[k] += time.perf_counter() * 0 + 1` lost more than half the increments, because one stale
   write-back erases everything the other threads added in between. A bare `x += 1` lost nothing
   in the same experiment, but nothing guarantees that, and the free-threaded build has no GIL to
   hide it. It's monitoring, not science, hence minor. *Fix:* compute `elapsed` first, then update
   under a lock, or use a metrics library's thread-safe counters.

10. **Hidden global dependencies, and test hooks in production code.** Every handler reaches
    into module globals (`_store`, `_verify_token`, `get_model()`). Tests have to call
    `configure()` on the module and remember `reset_for_tests()` between cases, they can't run in
    parallel, and a forgotten reset leaks state into the next test. `reset_for_tests()` ships to
    production, where calling it drops the model and zeroes the metrics. Before `configure()` has
    run, a handler fails deep inside with `'NoneType' object is not callable` or a similar
    `AttributeError`. *Fix:* keep the process-wide dependencies, but wire them in one place:
    `configure()` as the composition root that also resets everything derived from them (#11), or
    a small `PredictionService(verify_token, store, model_factory)` object that the server holds
    and tests construct fresh. Delete `reset_for_tests`.

11. **`configure()` keeps state built from the old dependencies.** It swaps `_model_factory` and
    `_store`, but the loaded `_model` and the `_load_structure` cache (filled from the old store)
    survive. Reconfiguring, or the next test that configures fakes, silently keeps the old model
    and the old store's structures. `reset_for_tests()` exists to paper over this and misses the
    structure cache, so tests are order-dependent. *Fix:* `configure()` resets `_model` and calls
    `_load_structure.cache_clear()`.

## Red herrings (say why they're fine)
- **A process-wide, lazily loaded model is the right design.** It's expensive to load and
  read-only once loaded, so one shared instance per process is exactly what you want. "Singletons
  are bad" isn't the point; the problems are how it's published (#2) and the request state kept
  next to it (#1, #3).
- **The double-checked locking shape is fine, and so is holding the lock during the load.** The
  unlocked `if _model is None` is a single reference read; on the default (GIL) build of CPython
  the worst it can do is send a thread into the lock unnecessarily. Holding `_model_lock` for the
  whole load is the point: requests that arrive meanwhile need the model and must wait anyway.
  Only the order of "assign" and "load" is wrong.
- **`handle_metrics` reads `METRICS` without the lock.** `dict(METRICS)` is a single C-level copy,
  so on the GIL build no other Python thread runs in the middle of it, and it returns a snapshot
  that callers can't use to mutate the live dict. The values can be one request apart from each
  other, which is fine for metrics. The writers are the problem (#9).

## Model 1-minute summary
"Four blockers, three with the same root cause: request data kept in module globals.
`CURRENT_USER` is process-wide, so an anonymous request runs as the previous caller, and under
concurrency alice can pass the access check with bob's identity and read his predictions.
`DEFAULT_CONDITIONS` is mutated by every request, so predictions silently use the previous
caller's temperature. Job IDs come from an in-memory counter, so they repeat after every restart
and across workers. Separately, the lazy loader publishes the model before its weights load, so
early concurrent requests get predictions from random weights. Then the majors: the store
connection may be opened before workers fork, the structure cache is unbounded, the model runs
once per structure, and request size isn't capped. I'd keep the model and clients as
process-wide singletons wired in `configure()`, pass the user and conditions explicitly, use
UUIDs, and delete `reset_for_tests`."

## Reference fix
`L3_prediction_service_fixed.py`:
- `_authenticate` returns the `User` and rejects a missing token; handlers pass it to
  `_require_project`. `CURRENT_USER` is gone.
- `get_model` loads into a local and publishes afterwards. `configure()` resets the model and
  clears the structure cache, and `reset_for_tests` is gone.
- Conditions are a fresh dict per request with validated keys; `DEFAULT_CONDITIONS` is a
  read-only `MappingProxyType`.
- UUID job IDs; one batched `predict` call; a 400 above `MAX_STRUCTURES_PER_REQUEST`; a bounded
  LRU for structures.
- Metrics: the elapsed time is computed first, then the update happens under `_metrics_lock`.
  `warm_up()` no longer touches the store.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_09_singleton_di.py -k L3 -v`
