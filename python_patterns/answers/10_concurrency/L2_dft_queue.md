# Answer key: L2_dft_queue.py — Concurrency, level 2 (20 min)

**Pattern focus:** A producer–consumer queue has to end correctly on every path: each item
marked done, every worker told to stop (including when the producer fails), backpressure so the
producer can't run away, and nothing slow done while holding a lock. Also check what the
results are keyed by.

**Good clarifying questions:** How long does a relaxation take, and how many jobs can the
cluster run at once (what should `n_workers` be)? Is the cluster client thread-safe? How fast
does the generative model produce candidates, and how many per campaign? Can two candidates
share a formula? What does `wait()` do when a job is stuck in the scheduler?

## Blockers (silently wrong results / data loss)

1. **Energies are keyed by formula, so polymorphs overwrite each other.** Generated MOFs often
   share a composition: the same metal, linker and ratio in a different topology.
   `self.energies[candidate["formula"]]` keeps whichever finished last, so the active-learning
   loop sees one energy for several structures and credits it to whichever structure it looks
   up. *Fix:* key by the candidate's unique ID; keep the formula as data if it's needed.

2. **One failed job hangs the campaign forever.** The `except` branch ends in `continue`, which
   skips `self.tasks.task_done()`. The queue's count of unfinished tasks never reaches zero, so
   `self.tasks.join()` in `run` blocks forever. DFT jobs fail routinely (SCF not converging, hit
   walltime), so in practice the first failure hangs every campaign. Everything computed so far
   exists only in `self.energies`, in memory, and is lost when someone kills the hung process.
   An exception *outside* the `try` (a result without `energy_ev`, a failed upload) also kills
   the worker thread, so the pool shrinks as well. *Fix:* call `task_done()` in a `finally`
   that covers the whole item, and catch errors per item so one bad job can't kill a worker.

## Major

3. **One sentinel for `n_workers` workers.** `run` puts a single `_STOP`, so one worker exits and
   the other `n_workers - 1` block on `get()` forever. They're daemon threads, so the process can
   still exit, but each `run` leaks seven threads that keep the campaign object alive, and
   nothing can ever join them. *Fix:* put one sentinel per worker, then `join()` the threads.

4. **The upload runs while holding the lock.** `upload_artifacts` (slow network I/O) is inside
   `with self._lock`, so only one worker at a time can record a result and upload. The others
   queue on the lock as soon as their jobs finish, and one hung upload stalls every worker.
   *Fix:* hold the lock only around the dict update and upload outside it.

5. **The queue is unbounded, so there's no backpressure.** `queue.Queue()` has no `maxsize`,
   so `run` drains the whole generator into memory at once: a million candidates with their CIF
   text sit in the queue while 8 workers chew through them, and the generative model has produced
   everything before the active-learning loop has learned from a single result. *Fix:*
   `queue.Queue(maxsize=2 * n_workers)`; `put()` then blocks the producer when the workers fall
   behind.

## Minor

6. **A failing producer leaves the workers running.** If the candidate generator raises (the
   generative-model service goes away), `run` propagates the exception without sending any
   sentinel. The workers keep submitting the already-queued candidates to the cluster in the
   background, spending cluster time nobody is tracking, and then block forever. *Fix:* send the
   sentinels and join the workers in a `finally`.

7. **No timeout on `client.wait`.** The client takes `timeout=None` and the PR passes nothing, so
   a job stuck in the scheduler blocks its worker forever; eight stuck jobs stall the whole
   campaign. *Fix:* pass a timeout (a few times the expected walltime) and treat expiry as a
   failure.

## Red herrings (say why they're fine)
- **Threads, not processes, are the right tool here.** The workers spend their time waiting on
  the cluster API, and a thread blocked on network I/O releases the GIL, so the waits overlap.
  Compare L1, where the work is pure-Python CPU.
- **`self.failures.append(...)` from several threads without the lock** is safe: `list.append`
  is atomic in CPython. `queue.Queue` does its own locking too, so `put` and `get` need no lock.

## Model 1-minute summary
"Two blockers. Energies are keyed by formula, so polymorphs silently overwrite each other and the
active-learning loop credits energies to the wrong structures; key by candidate ID. And the
failure path skips `task_done()`, so the first failed DFT job leaves `tasks.join()` waiting
forever, with all the results held only in memory. Then the queue mechanics: one sentinel for
eight workers leaks seven threads per run, the queue has no `maxsize`, so the producer drains the
whole stream into memory, and the upload runs under the lock, which serialises the workers on
slow I/O. Smaller: if the producer raises, the workers are never stopped, and `wait()` has no
timeout. Threads are the right choice here because the work is waiting on the cluster, and the
unlocked `failures.append` is fine."

## Reference fix
`L2_dft_queue_fixed.py`:
- Energies are keyed by candidate ID.
- `task_done()` runs in a `finally`, and each item's work is wrapped in one `try`, so a bad job
  can't kill its worker.
- `run` sends one `_STOP` per worker from a `finally` and then joins every worker.
- `queue.Queue(maxsize=2 * n_workers)` for backpressure; the upload is outside the lock;
  `wait(..., timeout=wait_timeout_s)`.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_10_concurrency.py -k L2 -v`
