# Answer key: L3_simulation_workspace.py — Context managers, level 3 (35 min + 5 min summary)

**Pattern focus:** a context manager that owns several resources (a lock, a directory, process
state). Check that each one is released exactly once on every path (success, an exception in
the block, an exception halfway through `__enter__`), that it only ever touches what it owns,
and that it doesn't hide process-global state behind a per-object API.

**Good clarifying questions:** Is `$SCRATCH` always set where this runs (cron, CI, laptops)?
Where do `job_id`s come from: are they plain names, unique per job? Do workers for the same
campaign run on several nodes (the lock lives in node-local `/tmp`)? How much do the codes
print to stdout over a day-long run? When a job fails, do we want its directory kept for
debugging?

## Blockers (silently wrong results / data loss)

1. **`os.chdir` and `os.environ` are process-wide, and `run_batch` runs eight workspaces at
   once in threads.** Each `__enter__` moves the cwd of the *whole process* and overwrites the
   one shared environment. Job A enters (cwd is A's directory), then job B enters (cwd is B's);
   A's `subprocess.run` now starts in B's directory with B's inputs, and `run` reads B's
   `results.json`. A's MOF is labelled with B's result, and nothing errors. A's simulation also
   runs with B's `OMP_NUM_THREADS`, and each `__exit__` restores whatever was current when
   *that* workspace entered, so the restores are scrambled too. I reproduced it with two nested
   workspaces in one thread, which is the interleaving the pool produces: A got
   `{"mof": "B", "omp": "8"}`. *Fix:* don't touch process state:
   `subprocess.run(cmd, cwd=self.path, env={**os.environ, "OMP_NUM_THREADS": str(self.threads)})`,
   and absolute paths everywhere (`self.path / "results.json"`).

2. **`__del__` deletes the running job's directory.** When a second worker hits `WorkspaceBusy`,
   its `__exit__` never runs, but `__del__` does, as soon as nothing references the object. Its
   `self.path` is the directory the *first* worker is running in (same `job_id`, same path), so
   it `rmtree`s a live simulation. With CPython's refcounting that happens right after the
   `except WorkspaceBusy` block in a retry loop (I checked, with the GC disabled), or when the
   losing batch returns, which can be hours before the winner (in another orchestrator on the
   node) finishes. The comment calls it a safety net; it's the opposite, and `__del__` also
   runs at unpredictable times such as interpreter shutdown. *Fix:* remove `__del__`. Cleanup
   belongs in `__exit__` and in `__enter__`'s own failure path (#6).

3. **The workspace adopts an existing directory and deletes it, under a root that defaults to
   the cwd.** `mkdir(parents=True, exist_ok=True)` silently reuses whatever already exists at
   `SCRATCH_ROOT / job_id`, and `__exit__` `rmtree`s it.
   - With `$SCRATCH` unset (cron, CI, a laptop) the root is `.`: `job_id="results"` deletes
     `./results`, and `job_id=""` makes `self.path` the root itself (`Path(".") / ""` is `.`),
     so the whole cwd goes. An absolute `job_id` replaces the root (`Path(".") / "/data/mofs"` is
     `/data/mofs`).
   - Even with `$SCRATCH` set, a job rerun after a `keep=True` run deletes the kept outputs.
   - With the relative root, `self.path` is also wrong after the `chdir`: `save_outputs` globs
     `./mof-5/mof-5` from inside the workspace and silently copies nothing (I checked).

   *Fix:* require `$SCRATCH`, validate `job_id` as a plain name, and create a fresh directory
   you own with `tempfile.mkdtemp(prefix=f"{job_id}-", dir=root)`. Only delete what you created.

## Major

4. **The lock is check-then-create, so it doesn't lock.** Between `self.lock_path.exists()` and
   `write_text`, a second worker starting the same job also sees no lock. Both proceed, and run
   in the same directory, overwriting each other's files. *Fix:* acquire atomically, with
   `fcntl.flock(fd, LOCK_EX | LOCK_NB)` on the lock file or `os.open(path, O_CREAT | O_EXCL)`.

5. **Locks go stale after a crash, and they are per node.** If a worker is OOM-killed, gets
   `kill -9`, or the node reboots, `__exit__` never runs and the lock file stays. Every later
   attempt raises `WorkspaceBusy`, which `run_batch` logs as "already running elsewhere", so the
   job silently never runs again. Nothing checks the PID written into the file. And
   `/tmp/sim-locks` is node-local, so it can't stop the same job running on two nodes (ask).
   *Fix:* `flock`. The kernel releases it when the process dies, so it fixes this and #4
   (`O_EXCL` fixes #4 but still goes stale). Across nodes, let the job database or the
   scheduler own "running"; `flock` over NFS isn't reliable.

6. **A failure halfway through `__enter__` leaks the lock.** If `__enter__` raises, `__exit__`
   is never called. Writing the inputs comes *after* taking the lock, so a full disk, a `bytes`
   value or a bad file name leaves the job locked forever, and leaves a half-made directory
   behind without a word. *Fix:* build `__enter__` on an `ExitStack`: register each cleanup as
   soon as its resource exists, and `pop_all()` once setup has fully succeeded. (The reference
   fix keeps a failed setup's directory and logs where it is, for the same reason as #10.)

7. **`__exit__` isn't exception-safe.** If `shutil.rmtree` fails (on NFS, `.nfsXXXX` files
   linger while something holds a file open; permissions), `unlink` never runs, so the lock
   leaks (#5 again), and the `OSError` raised from `__exit__` *replaces* the job's own exception:
   callers see `OSError: Directory not empty`, and the real error survives only as the
   `__context__` in the traceback. *Fix:* run each cleanup independently (an `ExitStack` does),
   and log cleanup failures instead of raising them.

8. **`capture_output=True` holds a day of output in memory, and loses it on timeout.** Long MD
   runs can print hundreds of MB, held in the orchestrator's memory until the process exits,
   eight times over in a full pool. On a timeout, `subprocess.run` raises before
   `stdout.log` is written. The partial output is attached to the `TimeoutExpired`, but nobody
   saves it, and `__exit__` then deletes the directory. *Fix:* stream stdout and stderr straight
   to files in the workspace (`stdout=f`).

## Minor

9. **`OMP_NUM_THREADS` leaks when it wasn't set before.** `__exit__` only restores it
   `if self._prev_threads is not None`, so an unset variable stays at the last job's value for
   the rest of the process, and the orchestrator's own numpy now uses that many threads. The
   fix for #1 removes this.

10. **Failures can't be debugged.** The `RuntimeError` drops stderr and the workspace path, and
    `__exit__` deletes the directory, with the simulation's own logs, on any exception.
    *Fix:* put the stderr tail and the path in the error, and keep the directory when the
    block raised (log where it is).

11. **Design.** `__enter__` does five unrelated things, and `run` silently depends on the cwd
    that `__enter__` set. Small single-purpose context managers (lock, directory) composed with
    an `ExitStack` are easier to get right and to test.

## Red herrings (say why they're fine)
- **`time.monotonic()` for `wall_time_s` is right:** it can't jump when the system clock is
  corrected.
- **`__exit__` returning `None` is right:** the job's exception propagates, and returning `True`
  would hide every failed simulation. Ignoring `exc_type` isn't the problem; the problem is
  what `__exit__` does.
- **`subprocess.run(cmd_list, timeout=...)` is right:** no shell, so no injection or quoting
  problems with odd job names, and on timeout `run` kills the child and waits for it (no
  zombie). Grandchildren, e.g. MPI ranks under `mpirun`, can survive; worth knowing, not a blocker.

## Model 1-minute summary
"Three blockers, all about state this class doesn't own. One: `chdir` and `os.environ` are
process-wide, and `run_batch` runs eight workspaces in threads, so a job can run in another
job's directory and return that job's results under the wrong MOF. Two: a worker that loses
the lock still runs `__del__`, and its path is the running job's directory, so it deletes a
live simulation. Three: `exist_ok=True` plus `rmtree`, on a root that defaults to the cwd, can
delete directories it never created. Then the lock: check-then-create races, it goes stale
after any crash, and a failure halfway through `__enter__` leaks it, so jobs get stuck as
'already running' forever; cleanup errors also mask the real exception. I'd pass `cwd` and
`env` to subprocess, use `mkdtemp` under a required `$SCRATCH`, `flock`, and an `ExitStack`
for setup and teardown."

## Reference fix
`L3_simulation_workspace_fixed.py`:
- No `chdir` and no `os.environ` writes: `run` passes `cwd=` and `env=` and uses absolute paths.
- `$SCRATCH` is required, `job_id` is validated, each workspace gets its own `mkdtemp`
  directory, and `__del__` is gone, so it only ever deletes what it created.
- The lock is an `fcntl.flock`: atomic, and released by the kernel if the process dies (POSIX only).
- `__enter__` builds an `ExitStack` and hands it to `__exit__` with `pop_all()`, so a failed
  setup releases the lock. Cleanup failures are logged, not raised, and a failed job's directory
  is kept and logged.
- stdout and stderr stream to files in the workspace; the error includes the stderr tail and the path.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_03_context_managers.py -k L3 -v`
