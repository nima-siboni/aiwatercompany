# Generators and lazy pipelines: learning nuggets

Generators produce values one at a time, so a pipeline over millions of records runs in constant
memory. The price: an iterator can be used only once, does its work later than the code
suggests, and is finished after its first unhandled exception.

The examples are written for these notes, not taken from the drills, and the outputs in comments
come from running them on Python 3.14. After the drills, the
[cheat sheet](../answers/04_generators/CHECKLIST.md) is the full checklist.

## 1. An iterator can be consumed only once

```python
energies = (float(x) for x in ["-1.5", "-2.5", "-3.0"])
n = sum(1 for _ in energies)               # 3, and the generator is now exhausted
total = sum(energies)                      # 0
```

A debug `print(list(gen))`, a `len(list(gen))` or an `x in gen` does the same. Such code passes
tests that pass lists, then silently gets nothing from a generator. **Fix:** do the work in one
pass, or materialise once with `list(...)` when the data fits in memory.

## 2. `__iter__` returning `self` makes a single-use dataset

```python
class Trajectory:
    def __init__(self, frames):
        self.frames, self.i = frames, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == len(self.frames):
            raise StopIteration
        self.i += 1
        return self.frames[self.i - 1]

traj = Trajectory(["f0", "f1", "f2"])
[f for f in traj], [f for f in traj]       # (['f0', 'f1', 'f2'], []): epoch 2 is empty
```

**Fix:** make it an iterable that builds a fresh generator on every call:

```python
    def __iter__(self):
        yield from self.frames
```

## 3. Hand-written batching: the last batch, and `clear()`

```python
def batches(items, size):
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) == size:
            yield batch
            batch = []

list(batches(range(5), 2))                 # [[0, 1], [2, 3]]: item 4 is never seen
```

**Fix:** `if batch: yield batch` after the loop, or `itertools.batched(items, size)` (Python
3.12+), which gives `[(0, 1), (2, 3), (4,)]`. A second trap: replacing `batch = []` with
`batch.clear()` empties the list the consumer is still holding. That version gave `[[], []]` for
`range(4)`.

## 4. `groupby` only groups neighbours

```python
rows = [("MOF-1", 2.0), ("MOF-2", 3.0), ("MOF-1", 2.2)]
[(k, len(list(g))) for k, g in itertools.groupby(rows, key=lambda r: r[0])]
# [('MOF-1', 1), ('MOF-2', 1), ('MOF-1', 1)]: MOF-1 shows up as two groups
```

**Fix:** sort by the same key first (`sorted(rows)` gave `[('MOF-1', 2), ('MOF-2', 1)]`), or
aggregate into a dict, or in SQL.

## 5. Returning a generator from inside `with open(...)`

```python
def read_ids(path):
    with open(path) as f:
        return (line.strip() for line in f)   # the file closes as the function returns

list(read_ids(path))                       # ValueError: I/O operation on closed file.
```

**Fix:** make the function itself a generator, so it owns the file until it's finished:

```python
def read_ids(path):
    with open(path) as f:
        for line in f:
            yield line.strip()
```

## 6. A bare `next()` inside a generator

```python
def records(lines):
    it = iter(lines)
    header = next(it).split(",")           # on empty input: StopIteration inside a generator
    for line in it:
        yield dict(zip(header, line.split(",")))

list(records([]))                          # RuntimeError: generator raised StopIteration
```

**Fix:** `first = next(it, None)`, and `return` if it's `None`.

## 7. An exception finishes the generator

```python
def parse_all(lines):
    for line in lines:
        yield float(line)

gen = parse_all(["1.0", "oops", "3.0"])
out = []
while True:
    try:
        out.append(next(gen))
    except ValueError:
        continue                           # "skip the bad line"
    except StopIteration:
        break
out                                        # [1.0]: the generator died at "oops", so 3.0 is lost too
```

The consumer can't skip a bad item: once an exception escapes, the generator is closed.
**Fix:** handle per-item errors inside the generator, and count and log what you skip
(`path:line`) rather than writing `except Exception: continue`.

## 8. Argument checks in a generator function run late

```python
def windows(xs, size):
    if size <= 0:
        raise ValueError("size must be positive")
    for i in range(len(xs) - size + 1):
        yield xs[i:i + size]

w = windows([1, 2, 3], 0)                  # no error yet: the body hasn't started
next(w)                                    # ValueError: size must be positive
```

**Fix:** validate in a plain function that then returns the generator, so bad arguments fail at
the call.

## Usually fine (don't flag it)

- A generator expression passed straight to `sum()`, `any()`, `max()` or `"".join()`: one pass,
  as intended.
- `yield` inside `with open(...)`: the file closes when the generator finishes or is closed.
- `yield batch` followed by `batch = []`: rebinding, not `clear()`.
