# Answer key: L2_results_store.py — Context managers, level 2 (20 min)

**Pattern focus:** commit or publish only on the success path. `finally` is for cleanup that
must happen either way, never for "make it permanent". Also: know what the library context
managers actually do (`with sqlite3.connect(...)` is a transaction, not a close), and know the
atomic-write recipe.

**Good clarifying questions:** Where do `/tmp` and the snapshot directory live on the cluster
(same filesystem)? Can two exports run at once (a cron export plus a manual refresh, or two
campaigns)? Does a retried job keep its `job_id`? Does the dashboard run as a different user?
Is the database on local disk or a network filesystem?

## Blockers (silently wrong results / data loss)

1. **`transaction()` commits in `finally`, so a failed job commits half an isotherm.** It only
   rolls back on `sqlite3.Error`. Any other exception (a point dict missing a key, a
   `TypeError`, Ctrl-C) skips the `except` and goes straight to `conn.commit()`. A job whose
   third point lacks `uptake_mmol_g` commits the first two points and then raises; the
   scheduler retries and inserts the full isotherm, so the first two points are now
   duplicated. That's the exact bug the PR set out to fix. The irony is that
   `with sqlite3.connect(...) as conn` alone *would* have rolled back, but `transaction()`
   exits first and commits, so the connection's rollback finds nothing to undo. (A hard kill
   is fine: SQLite's journal rolls back an uncommitted transaction. It's the Python-level
   exceptions that `finally` turns into commits.) *Fix:* `except BaseException: rollback;
   raise` and `else: commit`, which is what `with conn:` already does.

2. **`atomic_write` renames in `finally`, so a failed export replaces the last good snapshot
   with a partial one.** If anything raises inside the block, the half-written temp file is
   still moved over `path`. The "atomic" helper produces exactly the half-written file it was
   meant to prevent, and destroys the previous good one. It happens today: see #3. *Fix:* rename
   only after the block finished; on any exception, delete the temp file and re-raise.

## Major

3. **`round(uptake, 3)` crashes the export whenever any point has failed.** Failed points are
   stored with `uptake_mmol_g = None` (the docstring of `add_job` says so), and `round(None, 3)`
   raises `TypeError`. Combined with #2, every export after the first failed point publishes a
   truncated snapshot: I ran it with 5 good rows and then one failed point, and the dashboard
   file went from 5 lines to 4, with no error visible to the dashboard. *Fix:*
   `None if uptake is None else round(uptake, 3)`.

4. **The temp file is created in `tempfile.gettempdir()`, not next to the target.** On a cluster
   `/tmp` is node-local while the snapshot lives on shared storage (and many Linux machines
   mount `/tmp` as tmpfs). `os.rename` can't move a file between filesystems, so every export
   fails with `OSError: [Errno 18] Invalid cross-device link`. Copying instead (`shutil.move`)
   would "work" but wouldn't be atomic. *Fix:* create the temp file in the target's directory.

5. **The fixed temp name `<basename>.tmp` lets concurrent exports clobber each other.** Two
   exports whose targets share a file name (two campaigns both writing `snapshot.jsonl`, or the
   cron export and a manual refresh) open the same temp file. Both truncate it, and whichever
   renames first carries the other's writes along. In my check, campaign B's snapshot ended up
   holding campaign A's data, and A's export crashed with `FileNotFoundError`. *Fix:*
   `tempfile.mkstemp(dir=target_dir)` (unique name, same filesystem as #4). Note that `mkstemp`
   creates the file with mode 0600, so `chmod` it if the dashboard runs as another user.

## Minor

6. **`with sqlite3.connect(...) as conn` doesn't close the connection.** The connection's context
   manager commits or rolls back; it never closes. CPython usually frees it at function return
   (3.13 emits `ResourceWarning: unclosed database`, hidden by default), but an exception traceback keeps it
   alive, and the code shows the author believed `with` closes it. *Fix:*
   `with closing(sqlite3.connect(path)) as conn:`.

7. **`os.rename` instead of `os.replace`.** On Linux and macOS `rename` already replaces the
   target atomically, so this is about portability and intent: on Windows it raises
   `FileExistsError` when the target exists. `os.replace` behaves the same everywhere.

8. **No `flush` + `fsync` before the rename.** `close()` hands the data to the OS, so readers do
   see the whole file. What's missing is durability: after a power cut or kernel crash the
   rename can reach disk before the data, leaving an empty or truncated snapshot. *Fix:*
   `f.flush(); os.fsync(f.fileno())` before `os.replace`.

9. **Retries still duplicate data.** All-or-nothing fixes the half-isotherm case, but if a
   worker dies *after* the commit and before the scheduler records success, the retry inserts
   the whole isotherm again: there is no key to collide on. *Fix:*
   `PRIMARY KEY (job_id, mof_id, pressure_bar)` and `INSERT OR REPLACE`. The existing table needs
   a migration, because `CREATE TABLE IF NOT EXISTS` won't add the key.

## Red herrings (say why they're fine)
- **`fetchall()` before writing looks wasteful, but it's the right call.** The query finishes
  and SQLite drops its read lock before the slow file write. Streaming from the cursor while
  writing would hold that lock for the whole export, and in SQLite's default rollback-journal
  mode a held read lock stops the workers from committing ("database is locked"). I checked:
  after `fetchall()` another connection gets an exclusive lock at once; mid-iteration it
  can't. A few hundred thousand rows is well under 100 MB.
- **A new connection per call is fine.** SQLite connections are cheap to open (a local file,
  no server), and per-call connections sidestep sqlite3's rule that a connection belongs to the
  thread that created it. They just need closing (#6).

## Model 1-minute summary
"Two blockers, and they're the same mistake: making things permanent in `finally`.
`transaction()` commits in finally and only rolls back on sqlite3 errors, so a KeyError or a
Ctrl-C halfway through a job commits half an isotherm, and the retry duplicates it.
`atomic_write` renames in finally, so a failed export replaces the last good snapshot with a
truncated one, and exports do fail today, because `round(None)` crashes on any failed point.
Then the atomic write itself: the temp file is in /tmp, so on the cluster the rename crosses
filesystems and fails, and the fixed temp name lets two exporters overwrite each other. The fix
is commit and rename only on success, mkstemp in the target directory, fsync, and os.replace.
Smaller: `with sqlite3.connect()` doesn't close anything, and a primary key would make retries
idempotent."

## Reference fix
`L2_results_store_fixed.py`:
- `transaction()` commits only on success and rolls back on any exception (the semantics of `with conn:`).
- `atomic_write` uses `mkstemp` in the target directory, fsyncs, `os.replace`s only on success,
  deletes the temp file on failure, and sets mode 0644.
- Every connection is wrapped in `contextlib.closing`.
- `PRIMARY KEY (job_id, mof_id, pressure_bar)` with `INSERT OR REPLACE`; the export writes `null` for failed points.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_03_context_managers.py -k L2 -v`
