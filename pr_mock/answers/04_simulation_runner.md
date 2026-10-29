# Answer key: 04_simulation_runner.py

**Good clarifying questions:** How long does one simulation take, and how many cores does
the machine have? Does this run on a single node or should it go through the cluster scheduler?
Can two `screen` processes run at the same time (two people, or two notebooks)? Where do
`mof_id`s come from: user input, a database? What does `gcmc_sim` print on failure?

## Blockers (corrupted or wrong data, persisted)

1. **The cache is corrupted by concurrent writes.** Thirty-two threads mutate the global
   `_cache` and call `save_cache()` concurrently. `json.dump` iterates the dict while other
   threads insert into it, which can raise `RuntimeError: dictionary changed size during
   iteration`. Concurrent `open(..., "w")` calls truncate and interleave each other, so the
   JSON is left half-written. On the next run `load_cache` raises `JSONDecodeError`, and
   weeks of simulation results are effectively lost. *Fix:* guard with a `threading.Lock`;
   write atomically (temp file plus `os.replace`); better, use an append-only JSONL file or
   SQLite, one row per result, written from the main thread as futures complete.

2. **The return code isn't checked, so failures get cached as data.** If `gcmc_sim` fails
   but its last stdout line parses as a float (a cycle count, a partial progress number),
   that garbage is cached permanently and becomes a training label. *Fix:* `check=True` or
   check `returncode`, parse a well-defined output (JSON or a result file), and validate
   that the value is in range.

3. **The cache key depends on dict order and representation.** `str(params)` changes with
   key order (`{"pressure":..., "temperature":...}`) and with type (`298` vs `298.0`), so the
   same simulation is cached under different keys, and wasted compute is the good case.
   The key also omits the structure file contents and the simulator version, so editing a
   CIF or upgrading the code serves stale results. *Fix:*
   `json.dumps({"id": ..., **params}, sort_keys=True)` plus a hash of the CIF and the
   simulator version.

## Major

4. **`uptake` is unbound after three failures.** If all retries fail, `_cache[key] = uptake`
   raises `UnboundLocalError`, and because it goes through `ex.map`, the whole `screen` call
   dies. *Fix:* record the failure explicitly (a status plus an error message) and continue.

5. **Bare `except:`.** It swallows everything, including programming errors and
   `KeyboardInterrupt`, logs nothing, and retries deterministic failures such as a missing
   CIF three times. *Fix:* catch the specific exceptions (`subprocess.TimeoutExpired`,
   `CalledProcessError`, `ValueError`), log stderr, and only retry transient errors, with
   backoff.

6. **`shell=True` with an f-string means command injection, and it breaks on odd IDs.** A
   `mof_id` like `"x; rm -rf ~"`, or one containing spaces or parentheses (common in CSD
   refcodes and filenames), runs as shell code or breaks the command line. *Fix:* pass
   an argument list (`["gcmc_sim", "--structure", f"{mof_id}.cif", ...]`) without `shell=True`.

7. **One failure loses the whole batch.** `list(ex.map(...))` re-raises the first exception,
   and the results of everything else in flight are discarded from the return value.
   *Fix:* `as_completed` with per-item try/except, returning successes and failures.

8. **Oversubscription and timeouts.** 32 concurrent CPU-heavy simulations on, say, an 8-core
   box thrash the machine. A 1-hour timeout times 3 retries means up to 3 hours per bad MOF.
   *Fix:* default workers to the core count (or a resource argument); for thousands of MOFs
   use Slurm job arrays instead of threads on one node.

## Minor

9. **The whole cache is rewritten after every result,** which is O(N²) total I/O. It's
   negligible next to hour-long simulations but becomes noticeable with 50k entries.
10. **`json.load(open(...))`** leaks the file handle; a corrupt cache file crashes start-up
    with no recovery path.
11. **Relative `CACHE_FILE`** depends on the current working directory, so running from
    another directory silently starts an empty cache.
12. **Global mutable state** (`_cache`, `load_cache` as a hidden prerequisite of
    `run_simulation`) is hard to test. Use a small `ResultStore` class with injected paths.
13. **Check-then-run race:** two threads can both miss the cache for a duplicated ID and
    run it twice. That's minor (wasted compute, not wrong data); dedupe `mof_ids` up front.
14. **`if params:` branching** is awkward. Use one code path,
    `params = {**DEFAULT_PARAMS, **(params or {})}`.

## Red herrings (say why they're fine)
- **Mutable default `params={...}`:** a classic smell, but it's never mutated here, so it
  isn't a bug. Flag it as a nit and move the defaults into a `DEFAULT_PARAMS` constant.
- **"Threads are useless because of the GIL":** not true here. The work happens in
  subprocesses, and `subprocess.run` releases the GIL while waiting, so a thread pool is a
  reasonable choice. The real issues are the shared mutable state and oversubscription.

## Model 1-minute summary
"The top risk is data integrity. Concurrent unsynchronised writes can corrupt or lose the
whole cache, failed runs can be cached as real uptakes because the return code isn't checked,
and the cache key isn't canonical and ignores the structure and simulator version. Then
error handling: the unbound variable after retries, the bare except, and one failure killing
the whole screen. `shell=True` with interpolated IDs is an injection risk and a robustness
issue. I'd suggest SQLite or JSONL with atomic writes, an argument-list subprocess call, and
per-item results with explicit failure records."
