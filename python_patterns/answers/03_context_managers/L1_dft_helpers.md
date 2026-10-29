# Answer key: L1_dft_helpers.py — Context managers, level 1 (10 min)

**Pattern focus:** what a context manager does on the *error* path. Check what `__exit__`
returns, and check that a `@contextmanager` generator restores state when the block raises,
because that is exactly when nobody is watching.

**Good clarifying questions:** Can an OUTCAR be missing or unfinished (job still queued,
crashed during setup), and should the script then skip the job with a record or stop? Are the
job directories relative paths? Does anything call these helpers from threads?

## Blockers (silently wrong results / data loss)

1. **`Timer.__exit__` returns `True`, so it swallows every exception from the block.** A
   truthy return from `__exit__` tells Python the exception has been handled. In the
   collection loop, if `runs/mof-2` has no OUTCAR, `read_energy` raises `FileNotFoundError`,
   `Timer` swallows it, `energy` still holds mof-1's value, and `energies["runs/mof-2"]` gets
   mof-1's energy. Nothing is logged except the timing line. `__exit__` also receives
   `KeyboardInterrupt` and `SystemExit`, so Ctrl-C inside a timed block is swallowed too.
   *Fix:* return nothing (`None`) from `__exit__`. Log the failure if you like
   (`exc_type is not None`), but let it propagate.

## Major

2. **`in_dir` doesn't restore the directory when the block raises.** In a `@contextmanager`
   generator, the block's exception is re-raised at the `yield`; without `try/finally`,
   `os.chdir(prev)` never runs. Together with #1 the damage spreads silently: after the
   failure the process stays in `runs/mof-2`, so the next `in_dir("runs/mof-3")` resolves
   relative to it and raises `FileNotFoundError`, which `Timer` swallows again. I ran it with
   four jobs and one missing OUTCAR: all four came back with mof-1's energy.
   *Fix:* put the `yield` in `try/finally`. On Python 3.11+, `contextlib.chdir` does exactly this.

3. **`read_energy` returns `None` for a job that never finished an ionic step.** With no TOTEN
   line (crashed during setup, still running), it returns `None` despite the `-> float`
   annotation. The `None` lands in `energies` and fails far away (`min()` raises `TypeError`)
   or becomes an empty CSV cell. *Fix:* raise `ValueError` naming the file.

## Minor

4. **`open()` without `with` in `read_energy`.** The file is closed only when the object is
   garbage-collected. CPython usually does that at function return, but if `float()` raises,
   the traceback keeps the frame and the file alive while it's handled, and on PyPy files close
   only when the GC runs, so a loop over thousands of OUTCARs can hit "Too many open files".
   *Fix:* `with open(outcar_path) as f:`.

5. **`time.time()` measures a duration with the wall clock**, which jumps when NTP corrects
   it, so `elapsed` can be negative or far off. *Fix:* `time.perf_counter()`.

6. **`in_dir` isn't needed here.** `os.chdir` changes the directory of the whole process, so
   it breaks as soon as jobs are parsed from a thread pool. `read_energy(os.path.join(job_dir,
   "OUTCAR"))` needs no context manager at all. Keep `in_dir` for tools that insist on the cwd.

## Red herring (say why it's fine)
- **`float(line.split()[-2])` looks like an off-by-one, but it's right.** The last token is
  the unit (`eV`), so the number is second from the end, as the docstring's example line
  shows. Keeping the *last* match is also intended: VASP prints one TOTEN per ionic step and
  we want the final one. Suggest a one-line test against a real OUTCAR snippet.

## Model 1-minute summary
"The blocker is `Timer.__exit__` returning True. It silently swallows every exception,
Ctrl-C included, and in the collection loop a missing OUTCAR leaves `energy` holding the
previous job's value, which is then recorded for the wrong job. It compounds with `in_dir`,
which has no try/finally: after one failure the process stays in the wrong directory, every
later relative path fails, and the Timer swallows those errors too, so every later job gets
the same stale energy. Both fixes are tiny: return nothing from `__exit__`, and put the
`yield` in a try/finally. Next, `read_energy` should raise instead of returning None for
unfinished jobs. Smaller: `with open`, `perf_counter`, and passing paths instead of
chdir. I'd add tests that an exception propagates through both context managers and that
the cwd is restored afterwards."

## Reference fix
`L1_dft_helpers_fixed.py`:
- `Timer.__exit__` returns nothing (so exceptions propagate), logs failures, and uses `perf_counter`.
- `in_dir` restores the cwd in `finally` (equivalent to `contextlib.chdir` on 3.11+).
- `read_energy` reads with `with open(...)` and raises `ValueError` when there is no TOTEN line.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_03_context_managers.py -k L1 -v`
