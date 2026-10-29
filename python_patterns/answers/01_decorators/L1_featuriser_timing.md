# Answer key: L1_featuriser_timing.py — Decorators, level 1 (10 min)

**Pattern focus:** the four things every function-wrapping decorator must get right: return
the wrapped function's result, forward `*args, **kwargs`, copy the metadata with
`functools.wraps`, and do its bookkeeping in `try/finally`. Also learn to spot code
elsewhere that relies on `__name__`, because without `wraps` it sees the wrapper's name.

**Good clarifying questions:** Does anything besides `featurise` read `FEATURISERS`? Are
featurisers ever called with keyword arguments (notebooks, other pipelines)? Are the
timings for a one-off profiling run or for a long-running service? Do featurisers run in
threads?

## Blockers (silently wrong results)

1. **The registry keeps only the last featuriser** (`timed` has no `functools.wraps`).
   `register` receives the wrapper, whose `__name__` is `"wrapper"`, so
   `FEATURISERS[func.__name__] = func` writes the same key three times. `featurise` then
   returns `names == ["wrapper"]` and one column (the last registered featuriser,
   `n_elements`), and nothing errors. *Fix:* `@functools.wraps(func)` on `wrapper`, and make
   `register` raise on a duplicate name so a collision can never pass silently again.

2. **The wrapper drops the return value, and `dtype=float` hides it.** `wrapper` computes
   `result` but never returns it, so every featuriser returns `None`.
   `np.array(..., dtype=float)` converts `None` to `nan` instead of raising, so `featurise`
   returns an all-NaN matrix. With issue 1 on top, the PR output is
   `(["wrapper"], [[nan], [nan], ...])`. A model or imputer that tolerates NaN trains on
   nothing without complaint. *Fix:* `return func(*args, **kwargs)`.

## Major

3. **Keyword arguments crash.** `wrapper(*args)` doesn't accept `**kwargs`, so
   `volume_per_atom(s, include_hydrogen=False)`, the parameter this same PR adds, raises
   `TypeError: ...wrapper() got an unexpected keyword argument 'include_hydrogen'`. *Fix:*
   `def wrapper(*args, **kwargs)` and `func(*args, **kwargs)`.

## Minor

4. **`time.time()` is a wall clock.** NTP can step it backwards or forwards mid-call, which
   records a negative or inflated duration. *Fix:* `time.perf_counter()`, which is monotonic
   and meant for measuring intervals.
5. **Timing is lost when a featuriser raises.** `metal_fraction` on an empty structure
   raises `ZeroDivisionError` before `TIMINGS` is updated, so the failing featuriser, often
   the one you're profiling, never shows up. *Fix:* record the time in a `finally` block.

## Nit
- `TIMINGS[name] += elapsed` is a read-modify-write, so updates can be lost if featurisers
  ever run in threads. That's harmless for a profiling aid; mention it and move on.
  Storing a call count next to the total would also let you report the mean per call.

## Red herring (say why it's fine)
- **`@register` above `@timed` is the right order.** Decorators apply bottom-up, so
  `register` receives the timed function, and everything `featurise` calls through the
  registry is timed. Swapping them would register the untimed function. The bug is the
  missing `wraps`, not the order.

## Model 1-minute summary
"There are two blockers, and neither raises an error. First, `timed` doesn't use
`functools.wraps`, so every featuriser registers under the name 'wrapper'. The registry
ends up with one entry, and `featurise` returns one column instead of three. Second, the
wrapper computes `result` but never returns it, so every featuriser returns None, and
`dtype=float` quietly turns that into NaN. Together they give a one-column all-NaN feature
matrix. Next, the wrapper only forwards positional arguments, so the new
`include_hydrogen=False` keyword crashes with a TypeError. The minor points: time with
`perf_counter` rather than `time.time`, and record the time in a `finally` so failing
featurisers still show up. The decorator order is correct. I'd add a test that checks the
registry's names and that `featurise` returns finite values, one column per featuriser."

## Reference fix
`L1_featuriser_timing_fixed.py`:
- `@functools.wraps(func)` on the wrapper; `register` raises on a duplicate name.
- `return func(*args, **kwargs)` inside `try`, with the timing recorded in `finally`.
- `time.perf_counter()` instead of `time.time()`.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_01_decorators.py -k L1 -v`
