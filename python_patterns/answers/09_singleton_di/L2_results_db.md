# Answer key: L2_results_db.py — Singleton and dependency injection, level 2 (20 min)

**Pattern focus:** A lazily created module-level resource has a lifecycle (creation,
configuration, sharing between threads, transactions, close), and each stage can go wrong.
For every caller, ask what the hidden global connection means: which database it points at,
whose transaction it's in, and what happens after `close_db()`.

**Good clarifying questions:** How many rows per second does the runner write, and is batching
needed at all? How many runner processes write to this file, from how many nodes? Is "the shared
filesystem" NFS or Lustre? What should happen when a retried job reports again with the same
`job_id`? How do notebooks switch between campaigns?

## Blockers (silently wrong results / data loss)

1. **`get_db(path)` only honours `path` on the first call.** Once `_conn` is set, the argument
   is ignored. A notebook that looked at campaign 41 and then calls
   `get_db("campaign_42.sqlite")` gets campaign 41's connection, and `best_candidates()` shows
   the wrong campaign's MOFs with no error. In the runner, anything that touches the store before
   the start-up `get_db(path)` (a helper, an import-time call, a test) opens
   `gcmc_results.sqlite` in the working directory, and the real path is then silently ignored.
   *Fix:* raise if a different path is requested while a connection is open, or make the path
   part of an explicit store object (#7).

2. **One failed insert rolls back the whole batch.** The connection is shared, so its single
   transaction holds every row inserted since the last commit, from every worker thread. A
   duplicate `job_id` (a retried job reporting twice) or a NaN uptake raises `IntegrityError`.
   (SQLite stores NaN as NULL, which fails `NOT NULL`; I checked.) The handler's
   `conn.rollback()` then discards up to 199 other jobs' results. Only the caller with the bad
   row sees an exception; the others already returned as if their rows were saved. `_pending`
   also keeps counting the discarded rows. *Fix:* never roll back a shared transaction because
   of one row. On a constraint error SQLite already undoes just the failed statement. Better,
   give each call its own short transaction (#3, #4).

3. **`close_db()` doesn't commit, so the last partial batch is lost.** `Connection.close()` does
   not commit, and the open transaction is discarded. A run of 1,150 jobs persists 1,000. The
   `atexit` hook also doesn't run when the process is killed (the OOM killer, SIGKILL, or a
   scheduler's SIGTERM when no handler is installed), so a pre-empted runner loses up to 199
   finished simulations. *Fix:* commit each result (see #4), or at the very least commit in
   `close_db()`.

## Major

4. **The open batch holds SQLite's write lock between commits.** After the first insert of a
   batch, this connection holds the database's write lock until the 200th row commits. With
   hour-long GCMC jobs, that can be hours. Any other writer, such as a second runner process or a
   notebook that tags results, waits for its busy timeout (5 s by default) and then fails with
   `sqlite3.OperationalError: database is locked`. That's the batching optimisation interacting
   with the long-lived connection. *Fix:* commit each result. A commit costs milliseconds; the
   jobs take minutes to hours. If the write rate really is high (a backfill), batch explicitly:
   `executemany` in one short transaction that a single caller owns.

5. **`close_db()` leaves the closed connection in `_conn`.** `get_db()` keeps returning it, so
   after one `close_db()` every call raises
   `ProgrammingError: Cannot operate on a closed database.` until the process restarts. That hits
   a notebook that closes and reconnects, and a test suite that closes between tests. *Fix:* set
   `_conn = None` in `close_db()`, under the lock.

6. **The lazy initialisation isn't atomic.** `get_db()` checks `_conn is None` outside `_lock`,
   so two threads making their first call at the same time both connect, and the second
   assignment wins. A thread that already took the losing connection writes its row into it.
   Nobody commits that connection, and when it's garbage-collected the row is rolled back.
   CPython 3.13 only emits a `ResourceWarning`, which is hidden by default. In the runner as
   described, `get_db(path)` runs before the workers start, so this needs a caller that relies on
   "opened lazily on first use"; it's rarer than #1–#5, but it's what the function promises.
   *Fix:* do the check and the connect under the lock.

## Minor

7. **Hidden dependency on the global connection.** `record_result` and `best_candidates` call
   `get_db()` themselves, so a test can't hand them a temporary database without reaching into
   module globals (`_conn`, `DEFAULT_DB`), and because of #1 it can't even redirect them
   reliably. *Fix:* accept the connection as a parameter (the reference fix adds an optional
   `conn=`), or wrap this in a small `ResultsStore(conn)` class that the runner creates once
   and passes around.

8. **SQLite on a shared filesystem (ask first).** If processes on more than one node write to
   this file over NFS, that's a correctness problem, not a speed problem. The SQLite FAQ warns
   that file locking is broken on many NFS implementations, so concurrent writers can corrupt the
   database. WAL mode doesn't help, because it needs every process on the same host. If the
   answer is "several nodes", this becomes a blocker: use a client-server database or route all
   writes through one process.

## Red herrings (say why they're fine)
- **`_pending += 1` on a module global from worker threads** looks like the classic
  unsynchronised counter, but the read, the increment and the reset all happen inside
  `with _lock`. The trouble is what the batch does (#2–#4), not the counter; the reference fix
  only removes it because it removes the batching. The same goes for `check_same_thread=False`:
  once you turn that check off, the sqlite3 docs make serialising writes your job, and `_lock`
  does that.
- **The module-level `_lock` and `atexit.register(close_db)` run at import.** They're cheap and
  do no I/O. The PR rightly doesn't connect at import: that would create `gcmc_results.sqlite`
  in whatever directory imports the module, including every test run.

## Model 1-minute summary
"Three blockers, all about the shared connection's lifecycle. First, `get_db(path)` silently
ignores the path after the first call, so a notebook that switches campaigns shows the old
campaign's results. Second, the connection is shared, so every thread writes into one
transaction: a single duplicate or NaN result rolls back up to 199 other jobs' results while
their callers think they were saved. Third, `close_db()` doesn't commit, so the tail of every run
is lost, and a killed runner loses up to a whole batch. Then the majors: the open batch holds
SQLite's write lock for hours, `close_db()` leaves a dead connection behind, and the lazy init
races. I'd drop the implicit batching and commit per result, which costs milliseconds next to
hour-long simulations, put the init under the lock, and pass the connection in. And I'd ask
whether several nodes write to this file over NFS."

## Reference fix
`L2_results_db_fixed.py`:
- `get_db` does the `None` check and the connect under `_lock`, and raises `ValueError` if a
  different path is requested while a connection is open.
- `record_result` runs one short transaction per result (`with _lock, conn:`). There's no shared
  batch to roll back, nothing is left uncommitted at exit, and the write lock is released at once.
- `close_db` resets `_conn` under the lock, so the store can be reopened.
- `record_result` and `best_candidates` take an optional `conn`, so tests can inject a database.
- #8 isn't a code change; it's a deployment question.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_09_singleton_di.py -k L2 -v`
