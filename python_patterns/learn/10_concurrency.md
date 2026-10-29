# Concurrency: learning nuggets

Threads and asyncio overlap waiting; processes spread CPU work across cores. Every concurrent
design has to answer the same questions: which result belongs to which input, what happens when
one item fails, how much runs at once, and who waits for or cancels each task.

The examples are written for these notes, not taken from the drills, and the outputs in comments
come from running them on Python 3.14. After the drills, the
[cheat sheet](../answers/10_concurrency/CHECKLIST.md) is the full checklist.

[Concurrency, I/O, retries and caching](../../pr_mock/learn/5_concurrency_io.md) covers these
with examples, so they aren't repeated here: results paired by completion order
(`as_completed`), `pool.map` losing the results after an exception, processes not sharing
globals, blocking calls inside `async def`, and retries.

## 1. Threads don't speed up pure-Python CPU work

```python
def cpu(n):
    s = 0
    for i in range(n):
        s += i * i
    return s

# four calls with n = 10_000_000, on a 12-core machine:
# sequential 0.94 s | 4 threads 0.93 s | 4 processes 0.34 s
```

On the standard (GIL) build, only one thread runs Python bytecode at a time. Threads help when
the work waits (network, disk, subprocesses) or runs in C code that releases the GIL, as much of
NumPy does. **Fix:** processes for pure-Python CPU work, or vectorise it.

## 2. Exceptions in discarded futures vanish

```python
with ThreadPoolExecutor() as ex:
    for p in ["a.cif", "b.bad", "c.cif"]:
        ex.submit(convert, p)                # convert("b.bad") raises ValueError
# the block finishes, and nothing reports the error
```

**Fix:** keep the futures and call `.result()` on each, which is where the exception surfaces,
and handle failures per item.

## 3. Producer–consumer queues: backpressure, `task_done` and sentinels

```python
q = queue.Queue()                            # no maxsize: a fast producer can fill memory

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        process(item)                        # raises on a bad item: the thread dies here...
        q.task_done()                        # ...and task_done is never called

for x in [1, 2, 3]:                          # item 2 is bad
    q.put(x)
q.join()                # hangs: still waiting after 0.5 s, with item 3 left in the queue
```

With three workers and a single `None` sentinel, two workers stayed blocked in `q.get()` for
good. **Fix:** `queue.Queue(maxsize=...)`; a per-item `try/except` with `task_done()` in a
`finally`; one sentinel per worker; and a shutdown path for when the producer itself fails.

## 4. A lock held across slow I/O serialises the workers

```python
def fetch(url):
    with lock:
        data = download(url)                 # 0.2 s, while holding the lock
        cache[url] = data
```

Four threads took 0.82 s; with the download moved outside the lock, 0.21 s. **Fix:** hold the
lock only for the shared-state update.

## 5. Unbounded fan-out in asyncio

```python
await asyncio.gather(*(predict(x) for x in candidates))    # every request in flight at once
```

With 500 candidates, all 500 requests were in flight at the same moment, which hits rate limits
and runs out of sockets. Behind a semaphore, never more than 16 were:

```python
sem = asyncio.Semaphore(16)

async def bounded(x):
    async with sem:
        return await predict(x)

await asyncio.gather(*(bounded(x) for x in candidates))
```

## 6. One failure in `gather`

```python
async def score(x):
    if x == 2:
        raise ValueError("bad input 2")
    return x * 10

await asyncio.gather(*(score(x) for x in range(4)))
# ValueError: bad input 2, and the other results are lost
await asyncio.gather(*(score(x) for x in range(4)), return_exceptions=True)
# [0, 10, ValueError('bad input 2'), 30]
```

**Fix:** `return_exceptions=True` or a per-task `try/except`, then handle the failures
explicitly. Don't turn them into fallback values like `0.0` or `None` that look like results.

## 7. Timeouts, tasks and `asyncio.run`

```python
await asyncio.wait_for(slow_call(), timeout=0.1)   # raises TimeoutError after 0.1 s
```

Every network call needs a timeout; without one, a hung server hangs the client. Keep a
reference to each task you start with `asyncio.create_task`, because unreferenced tasks can be
garbage-collected mid-run. And don't call `asyncio.run` from library code: inside a running loop
(Jupyter, a web server) it fails with
`RuntimeError: asyncio.run() cannot be called from a running event loop`.

## Usually fine (don't flag it)

- `list.append` or `d[k] = v` from several threads in CPython.
- Threads for I/O-bound work, such as network calls and subprocess waits.
- `queue.Queue` without extra locks: it does its own locking.
- Synchronous code between two `await`s: nothing else runs on that loop in the meantime.
