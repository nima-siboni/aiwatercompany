# Generators and lazy pipelines: review checklist

**What it's for:** producing values one at a time, so a pipeline over 50M records runs in constant
memory and stages compose without materialising anything. The price: an iterator is single-use,
does its work later than the code suggests, and is finished after its first unhandled exception.

**When you see it in a PR, check:**
- **Is the input consumed twice?** A comprehension, `len(list(...))`, `in`, or a debug `print(list(x))` before the real loop → the real loop gets nothing. Passes tests with lists, silently empty with generators.
- **A class whose `__iter__` returns `self`** (an iterator, not an iterable) → every pass after the first is empty. `__iter__` should build a fresh generator per call.
- **Hand-written batching or buffering** → is the last partial batch emitted? (`batched`, shuffle buffers, `send()`-based writers that drop their buffer on `close()`.)
- **`zip` over things that must match** → silent truncation and misalignment. Use `strict=True` (3.10+).
- **`itertools.groupby` on unsorted input** → one key split into several groups, wrong aggregates. Sort first, or aggregate in a dict or in SQL.
- **A generator returned from inside `with open(...)`** (`return (l for l in f)`, `return map(f, fh)`) → `ValueError: I/O operation on closed file`. Make the function itself a generator.
- **A bare `next(it)` inside a generator** (reading a header, peeking) → `RuntimeError: generator raised StopIteration` on empty input (PEP 479). Use `next(it, default)`.
- **Errors raised inside a generator and caught by its consumer** → the generator is finished, so the rest of its data is silently gone. Handle per-item errors inside the generator.
- **`except Exception: continue` in a stage** → silent data loss. Catch the specific error, count it, log `path:line`.
- **`tee` with one branch consumed before the other, or an unbounded `seen` set** → O(N) memory in a pipeline meant to be O(1).
- **Laziness and side effects:** validation that fails mid-stream after earlier batches were committed. Validate up front what you can.
- **Argument checks in a generator function** run at the first `next()`, not at the call.
- **`__len__` on a streaming dataset** → rescans the data on each call, or disagrees with what iteration yields.

**Usually fine (don't block on it):**
- A generator expression passed straight to `sum()`, `any()`, `max()` or `"".join()`: one pass, as intended.
- `yield` inside `with open(...)`: the file closes when the generator finishes or is closed.
- `yield batch` then `batch = []` (rebinding, not `clear()`), and buffered shuffles for data that doesn't fit in memory.

**Idiomatic shape:**
```python
class ShardDataset:                          # an iterable, not an iterator
    def __init__(self, paths):
        self.paths = list(paths)
    def __iter__(self):                      # a fresh pipeline on every call
        for path in self.paths:
            yield from read_records(path)

def read_records(path):
    with open(path) as f:                    # the generator owns the file
        for lineno, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:     # per-item errors handled inside
                log.warning("%s:%d: skipped", path, lineno); continue
            yield rec
```
