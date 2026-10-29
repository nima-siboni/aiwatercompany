# Checklist: worker pools, producer–consumer queues and asyncio

Concurrency overlaps waiting (threads, asyncio) or spreads CPU work over cores (processes). Every
design must answer the same questions: which result belongs to which input, what happens when
one item fails, how much runs at once, and who waits for or cancels each task.

**When you see it in a PR, check:**
- `zip(inputs, as_completed(...))`, or "input order" promised over completion-ordered results → mislabelled results.
- Threads for pure-Python CPU work → no speedup on the default (GIL) build; use processes or vectorise.
- `submit()` / `create_task()` results thrown away → exceptions never seen; tasks GC'd or cancelled by `asyncio.run`.
- One exception escaping `map`, `gather` or an `as_completed` loop → every other result lost.
- `queue.Queue()` without `maxsize` → no backpressure; the producer fills memory.
- `task_done()` outside a `finally`, or one sentinel for N workers → `join()` hangs, threads leak.
- No shutdown on the producer's error path → workers keep working, or block forever.
- A lock held across slow I/O → workers serialised; one hung call stalls everyone.
- Blocking calls in `async def` (`time.sleep`, `urllib`/`requests`, file or DB I/O, heavy CPU) → the loop freezes.
- Unbounded fan-out (`gather` over 100k coroutines with no semaphore) → API floods, sockets exhausted; bound it with a semaphore or a fixed pool of worker coroutines.
- No per-call timeout, or a deadline that drops pending work → hangs, partial results that look complete.
- Fixed retry delay, no jitter, or retrying non-transient errors → retry storms, wasted load.
- Fallback values on failure (`0.0`, `[]`, `None` used as data) → errors that look like results.
- `asyncio.run` inside library code → `RuntimeError` in Jupyter or any running loop.

**Usually fine (don't block on it):**
- `list.append` or `d[k] = v` from threads in CPython; synchronous code between two `await`s.
- Threads for I/O-bound work (network, subprocess waits): blocking I/O releases the GIL.
- `queue.Queue` without extra locks: it does its own locking.
- A `while True` background task that is referenced and cancelled in `finally`.

**Idiomatic shape:**
```python
async def run_all(items, call, limit=32, timeout_s=60):
    sem = asyncio.Semaphore(limit)

    async def one(item):
        async with sem:
            try:
                return await asyncio.wait_for(call(item), timeout_s)
            except Exception as exc:  # one failure stays one failure
                return exc

    return await asyncio.gather(*(one(item) for item in items))  # input order
```
