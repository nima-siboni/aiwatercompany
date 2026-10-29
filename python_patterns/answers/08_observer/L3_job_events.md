# Answer key: L3_job_events.py — Observer, level 3 (35 min + 5 min summary)

**Pattern focus:** a pub/sub hub inside a multi-threaded service. Check how subscribers are
referenced (weak references, bound methods), which thread runs the callbacks, and what
happens when a callback fails, blocks, mutates the event, or emits again. Also check whether
the registry can change during delivery.

**Good clarifying questions:** How long do callbacks take (retraining takes minutes)? Is
`emit` meant to be synchronous? Are components created or connected while jobs are running
(dashboards, the next AL round's trigger)? May subscribers modify events? Do consumers rely
on per-job ordering (queued, running, failed, queued...)? Who keeps subscriber objects alive,
and do we ever subscribe lambdas?

## Blockers (silent data loss)

1. **`weakref.ref` to a bound method is dead immediately.** `self.on_finished` creates a new
   bound-method object on every access, and the hub's weak reference is the only reference to
   it, so `ref()` returns None before anything is emitted. `RetrainTrigger` never receives an
   event and the active-learning loop never retrains, with no error. An inline lambda dies the
   same way. Only module-level functions and callable objects that something else keeps alive
   work, which is probably what the author tested with. *Fix:* `weakref.WeakMethod` for bound
   methods (`inspect.ismethod`). The weak design has a second trap worth raising:
   `ResultsStore(events, path)` on a line of its own is collected at once and silently never
   runs. If components aren't always held elsewhere, strong references plus explicit
   unsubscribe are simpler.

2. **Every subscriber shares one mutable event and payload.** `ProgressLog` does
   `event.payload.pop("raw_output", None)` on the shared dict. Patterns are visited in
   subscription order, so if `ProgressLog` was created before `ResultsStore`, the store saves
   `raw_output` as NULL for every job. Whether data is lost depends on construction order. For
   `queued` and `running` events the payload *is* the job's `params` dict, so a subscriber that
   edits it changes what the worker runs. *Fix:* make `JobEvent` frozen and give it a read-only
   copy of the payload (`MappingProxyType(dict(payload))` in `__post_init__`), and have
   `ProgressLog` store a trimmed copy made with `dataclasses.replace`.

3. **The registry is mutated while worker threads iterate it.** `emit` loops over
   `self._subscribers.items()` on worker threads. If another thread subscribes to a pattern
   that isn't in the dict yet (a dashboard connects, the next AL round builds its trigger), the
   loop raises `RuntimeError: dictionary changed size during iteration`. That happens inside
   `_run`, whose future nobody reads, so the job silently vanishes: no `finished`, no `failed`,
   and the result is lost. Callbacks that do I/O widen the window. `unsubscribe` is also a
   read-modify-write, so a concurrent `subscribe` to the same pattern can be lost. *Fix:* take a
   lock for every registry change; in `emit`, copy the matching subscribers under the lock and
   call them outside it (holding the lock during callbacks deadlocks as soon as a callback
   subscribes).

4. **`ResultsStore` uses its SQLite connection from worker threads.** Callbacks run on whichever
   thread calls `emit`, which is a job worker, not the thread that created the store. By
   default a `sqlite3` connection refuses that (`ProgrammingError: SQLite objects created in a
   thread can only be used in that same thread`), so every insert fails, and #5 swallows the
   error. In production nothing is stored and nothing is logged, while a unit test that emits
   from the main thread passes. *Fix:* `check_same_thread=False` plus a lock around writes (or
   one connection per thread, or a single writer thread).

## Major

5. **Subscriber exceptions are swallowed without a trace.** `except Exception: pass` is what
   keeps #4 invisible, and it will hide every future subscriber bug too. Isolating subscribers
   is right; doing it silently isn't. *Fix:* `logger.exception(...)` with the subscriber, the
   status and the job id, plus a failure counter.

6. **Slow subscribers run on the job workers.** Delivery is synchronous, so each callback's run
   time is added to the job. Once #1 is fixed, `RetrainTrigger` calls `self.retrain()` (minutes
   of training) inside a GCMC worker's `emit`: that worker is blocked for the whole retrain, and
   if another `every` jobs finish meanwhile, a second worker starts a second retrain. *Fix:*
   keep callbacks cheap and give slow work its own executor (`RetrainTrigger` submits to a
   single-thread executor). With many slow subscribers, deliver from a dispatcher thread with a
   queue instead. Ask whether `emit` is meant to be synchronous.

7. **A re-entrant `emit` delivers events out of order.** `RetryOnFailure` calls `runner.submit`
   inside the `failed` delivery, and `submit` emits `queued` immediately, before the rest of the
   `failed` subscribers have run. With `RetryOnFailure` subscribed before a dashboard listener
   on `*`, the dashboard sees queued, running, **queued, failed**, running, finished, so from
   the moment the retry is queued until it starts, it shows the job as failed. *Fix:* if `emit`
   is called while the same thread is already delivering, queue the event and deliver it after
   the current one (a per-thread pending queue). A single dispatcher thread would also give
   ordering across threads.

8. **`RetrainTrigger`'s counter is shared by all workers without a lock.** `self.n_new += 1` is
   a read-modify-write, so concurrent `finished` events can lose increments, and two threads can
   both pass the `>= self.every` check before either resets it, which starts two retrains.
   *Fix:* a `threading.Lock` around the increment, the check and the reset.

## Minor

9. **Dead weak references are never pruned.** Every garbage-collected subscriber leaves a dead
   reference in its list, and every `emit` walks past it. In this PR every method
   subscription is dead on arrival. *Fix:* drop dead references while taking the snapshot.

10. **Futures are fire-and-forget.** `submit` drops the future from `pool.submit`. Anything
    raised outside the `try` in `_run` (an `emit` that fails as in #3, or a `SystemExit` from a
    CLI-style simulator wrapper, which `except Exception` doesn't catch) is stored in that
    future and never seen: the job disappears with no log line and no `failed` event. *Fix:* a
    done-callback that logs and emits `failed`.

11. **Failure details are thrown away.** `str(exc)` turns `KeyError('cif')` into `"'cif'"`,
    which loses the exception type, and the traceback is never logged. *Fix:*
    `logger.exception` in `_run` and `repr(exc)` in the event.

## Nit
- Statuses are free strings, so `emit(JobEvent(j, "finsihed"))` only reaches `*` subscribers.
  An `Enum` would catch the typo.

## Red herrings (say why they're fine)
- **`unsubscribe` compares with `!=` rather than `is`.** That's correct: `obj.method` creates a
  new bound-method object each time, so an identity check would never match, while equality
  compares the underlying function and instance. (It matters once #1 is fixed.)
- **`timestamp: float = field(default_factory=time.time)`** stamps each event when it's
  created; `= time.time()` would have frozen the import time.
- **`ProgressLog`'s `deque(maxlen=1000)`** is appended to from many threads without a lock.
  It's bounded, and `deque.append` is thread-safe.

## Model 1-minute summary
"This can lose data silently in four ways. The hub keeps `weakref.ref` to bound methods, which
die immediately, so `RetrainTrigger` never fires; it needs `WeakMethod`. Subscribers share one
mutable payload, so `ProgressLog` can delete `raw_output` before the results store saves it,
depending on construction order; events should be frozen, with a read-only payload. `emit`
iterates the subscriber dict on worker threads while other threads subscribe, which raises
inside a job whose future nobody reads, so the job vanishes; it needs a lock and a snapshot.
And `ResultsStore` uses its SQLite connection from worker threads, so every insert fails,
while `except Exception: pass` hides it. Next: log subscriber errors, move retraining off the
job workers, queue re-entrant emits so retries don't reorder events, and lock the retrain
counter. Smaller: prune dead references and check the futures. Equality in `unsubscribe` and
the deque are fine."

## Reference fix
`L3_job_events_fixed.py`:
- `JobEvent` is frozen with a read-only copy of its payload; `ProgressLog` stores trimmed copies.
- `JobEvents` uses `WeakMethod` for bound methods and a lock for every registry change. It
  delivers from a snapshot taken under the lock, prunes dead references, logs each failing
  subscriber, and queues re-entrant emits per thread.
- `JobRunner` logs failures with tracebacks, puts `repr(exc)` in the event, and reports
  runner crashes from a future done-callback.
- `ResultsStore` opens its connection with `check_same_thread=False` and serialises writes.
- `RetrainTrigger` updates its counter under a lock and runs `retrain` on its own
  single-thread executor.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_08_observer.py -k L3 -v`
