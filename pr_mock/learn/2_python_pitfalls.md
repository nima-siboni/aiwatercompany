# 2. Python pitfalls that turn up in any PR

The classics first, as a table, then the subtle ones with examples. The outputs in comments
come from running each snippet on Python 3.14.

Pattern-specific pitfalls (decorators, caching, context managers, generators, registries,
dataclasses, observers, singletons, concurrency) have their own learning nuggets, one file per
pattern, in `../../python_patterns/learn/`.

## Quick table

| You see | What goes wrong | Fix |
|---|---|---|
| `def f(x, acc=[])` | one list shared by every call | `acc=None`, and create the list inside |
| `except:` or `except Exception: pass` | bugs disappear; a bare `except` also catches Ctrl-C | catch the errors you expect; log; re-raise |
| `return` inside `finally` | the exception is discarded (3.14 warns, PEP 765) | never return from `finally` |
| `json.load(open(p))` | relies on garbage collection to close the file | `with open(p) as f:` |
| `grid = [[0] * n] * m` | m references to the same row | `[[0] * n for _ in range(m)]` |
| `for x in xs: xs.remove(...)` | elements get skipped (a dict raises RuntimeError) | build a new list |
| `zip(ids, preds)` | silently stops at the shorter input | `zip(ids, preds, strict=True)` |
| `list(set(xs))` used as an ordering | the order changes between processes | `sorted(set(xs))` |
| `cache[0.1 + 0.2]` after storing `cache[0.3]` | KeyError: float keys after arithmetic | round to a tolerance, or use integer keys |
| a dict with int keys saved as JSON | keys come back as strings, so lookups miss | string keys, or convert on load |
| `assert` to validate inputs | removed under `python -O` | `if ...: raise ValueError(...)` |
| file reads, clients or env lookups at module level | imports fail or are slow; tests can't swap them | load lazily; pass dependencies in |

## Examples

### Defaults are evaluated once

```python
def make_job(name, tags=[], created=datetime.now()):
    tags.append(name)
    return {"tags": tags, "created": created}

a = make_job("a")
b = make_job("b")
b["tags"]                        # ['a', 'b']: the same list object as a["tags"]
a["created"] == b["created"]     # True: both carry the time the module was imported
```

**Fix:** `tags=None, created=None`, then build both inside the function. **Not a bug when**
the default is never mutated (a tuple, or a list that is only read). Then it's a nit.

### Truthiness treats 0 as "missing"

```python
def filter_stable(rows, max_e_above_hull=None):        # eV/atom
    max_e = max_e_above_hull or 0.1
    return [r["id"] for r in rows if r["e_hull"] <= max_e]

rows = [{"id": "A", "e_hull": 0.0}, {"id": "B", "e_hull": 0.05}]
filter_stable(rows, max_e_above_hull=0.0)              # ['A', 'B']: 0.0 became 0.1
```

A request for stable phases only silently lets metastable ones through. **Fix:**
`if max_e_above_hull is None:`. Look for the same thing in `if not energy:` and
`timeout = timeout or 30`.

### Copies are shallow

```python
DEFAULTS = {"model": {"lr": 1e-3, "layers": [64, 64]}}
cfg = DEFAULTS.copy()
cfg["model"]["lr"] = 1e-2
DEFAULTS["model"]["lr"]          # 0.01: the defaults changed for everyone
```

**Fix:** `copy.deepcopy`, or immutable configs (frozen dataclasses). NumPy views and
`model.state_dict()` belong to the same family (chapter 3).

### Closures capture variables, not values

```python
callbacks = [lambda: f"epoch {i}" for i in range(3)]
[cb() for cb in callbacks]       # ['epoch 2', 'epoch 2', 'epoch 2']

callbacks = [lambda i=i: f"epoch {i}" for i in range(3)]
[cb() for cb in callbacks]       # ['epoch 0', 'epoch 1', 'epoch 2']
```

This bites wherever handlers or callbacks are registered in a loop.

### `zip` truncates, and one dropped row shifts everything

```python
ids = ["MOF-1", "MOF-2", "MOF-3"]
preds = [0.1, 0.2]                       # a row was dropped upstream
dict(zip(ids, preds))                    # {'MOF-1': 0.1, 'MOF-2': 0.2}
dict(zip(ids, preds, strict=True))       # ValueError: zip() argument 2 is shorter than argument 1
```

If the dropped row was the first, every prediction is attached to the wrong ID, and nothing
errors. **Fix:** `strict=True`, or carry the IDs through the pipeline alongside the data.

### Set order changes between processes

```python
vocab = list({"Zn", "Cu", "O", "C", "H"})     # used as the one-hot column order
# PYTHONHASHSEED=1  ['Zn', 'O', 'C', 'H', 'Cu']
# PYTHONHASHSEED=2  ['O', 'Zn', 'Cu', 'C', 'H']
```

String hashing is randomised per process. If training and serving each build the vocabulary
this way, the served model sees its feature columns permuted and predicts garbage without an
error. **Fix:** `sorted(set(...))`, and save the vocabulary with the model. For the same reason,
never persist `hash(s)` (as a cache key or shard number); use `hashlib`.

### Swallowed exceptions

```python
def read_energy(path):
    try:
        with open(path) as f:
            return float(f.read().split()[-1])
    except Exception:
        return 0.0          # a missing file or a crashed run now reads as 0.0 eV
```

A plausible number is the worst kind of error value, because it gets averaged into the
results. `float()` also accepts `"nan"` and `"inf"` without complaint. **Fix:** catch only the
errors you expect, raise (or return `None` explicitly), and check `math.isfinite`.

```python
def f():
    try:
        raise ValueError("boom")
    finally:
        return "ok"         # f() returns 'ok' and the ValueError is gone
# Python 3.14: SyntaxWarning: 'return' in a 'finally' block
```

### Parsing numbers out of text

```python
re.search(r"\d", "Score: 10/10").group()     # '1'
"Energy = -3.2 eV".split()[-1]               # 'eV'
int(0.29 * 100)                              # 28, because 0.29 * 100 == 28.999999999999996
int(9 * 0.1)                                 # 0: a "10% validation split" of 9 rows is empty
```

**Fix:** have the producer write JSON (for LLM output, use structured outputs; see chapter
6). Otherwise match the whole number with an anchored regex, use `round()` rather than `int()`,
and check that every split has at least one row.

### `with sqlite3.connect(...)` doesn't close the connection

```python
with sqlite3.connect("results.db") as conn:
    conn.execute("insert into t values (1)")
# the transaction is committed here, but the connection is still open:
conn.execute("select count(*) from t").fetchone()     # (1,)
```

The context manager handles the transaction, not the connection. **Fix:**
`with contextlib.closing(sqlite3.connect(path)) as conn:`, with `with conn:` inside it for each
transaction. In a review, the commit logic is right and the leak is a Minor, unless a new
connection is opened per request.
