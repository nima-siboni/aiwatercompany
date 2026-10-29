# Singletons, global state and dependency injection: learning nuggets

A singleton, or any module-level object, gives the process one shared instance of something
expensive: a model, a connection, the settings. Dependency injection means code receives those
objects instead of fetching them itself, so tests and other callers can pass their own. Most bugs
are about when the shared object is created, and who else sees it.

The examples are written for these notes, not taken from the drills, and the outputs in comments
come from running them on Python 3.14. After the drills, the
[cheat sheet](../answers/09_singleton_di/CHECKLIST.md) is the full checklist.

## 1. A `__new__` singleton that also defines `__init__`

```python
class Registry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.items = {}                      # runs again on every Registry() call

r = Registry()
r.items["fire"] = "FIRE optimiser"
Registry() is r, Registry().items            # (True, {}): the same object, with its state wiped
```

**Fix:** initialise in `__new__` (or behind a flag), or drop the trick altogether: create one
instance at start-up and pass it around.

## 2. Settings read at import time are frozen

```python
# screen_settings.py
DRY_RUN = os.environ.get("DRY_RUN") == "1"
```

```python
import screen_settings
os.environ["DRY_RUN"] = "1"                  # a test, or a CLI flag applied after import
screen_settings.DRY_RUN                      # False
```

**Fix:** build a `Settings` object in `main()`, from the environment and flags, and pass it down.

## 3. A singleton factory with an argument ignores it after the first call

```python
_client = None

def get_client(base_url="https://api.internal"):
    global _client
    if _client is None:
        _client = Client(base_url)
    return _client

get_client()
get_client("https://staging.internal").base_url    # 'https://api.internal'
```

**Fix:** key the cache on the argument (or reject a different one), or remove the argument.

## 4. Aliased module defaults

```python
DEFAULTS = {"cutoff": 6.0}

def make_opts(**overrides):
    opts = DEFAULTS                          # an alias, not a copy
    opts.update(overrides)
    return opts

make_opts(cutoff=8.0)
make_opts()["cutoff"]                        # 8.0, and DEFAULTS is now {'cutoff': 8.0}
```

**Fix:** `opts = {**DEFAULTS, **overrides}`.

## 5. Request data in a global

```python
CURRENT_USER = None

def handle(request):
    global CURRENT_USER
    CURRENT_USER = request.user
    return run_query()                       # reads CURRENT_USER
```

With two overlapping requests on two threads, alice's request read "bob" by the time it reached
`run_query()`. (The demo forced that interleaving with events.) **Fix:** pass the user as an
argument, or use a `contextvars.ContextVar` for request-scoped values.

## 6. Lazy initialisation races

```python
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model_from_disk()      # slow
    return _model
```

Four threads calling this at start-up built the model four times. `functools.cache` doesn't
prevent it either: its body also ran four times under concurrent first calls. That's harmless
for a read-only model (just wasted work), but not for anything that must exist exactly once.
**Fix:** build it at start-up, or use a lock with a second check inside it; the locked version
built it once.

## 7. One shared connection means one shared transaction

```python
conn = sqlite3.connect("results.db")         # shared by the whole process

def save_uptake(mof_id, value):              # caller A: commits later, in a batch
    conn.execute("insert into results values (?, ?)", (mof_id, value))

def save_failed(mof_id):                     # caller B: hits an error and rolls back
    conn.execute("insert into results values (?, ?)", (mof_id, None))
    conn.rollback()                          # ...which also rolls back A's insert

save_uptake("MOF-5", 4.1)
save_failed("ZIF-8")
conn.commit()
conn.execute("select count(*) from results").fetchone()    # (0,): A's row is gone
```

**Fix:** one connection, or at least one transaction, per unit of work, passed in by whoever owns
that work.

## 8. Code that fetches its own dependencies

```python
def screen(candidates):
    store = get_store()                      # hidden dependency: tests must monkeypatch it
    ...

def screen(candidates, store):               # injected: a test passes a fake
    ...
```

Build the real objects once, in `main()` (the "composition root"), and pass them down. Also
watch for per-process counters used as IDs (`itertools.count()`): they repeat after a restart and
across workers. Use UUIDs or a database sequence.

## Usually fine (don't flag it)

- One read-only object per process (a model, a lookup table), loaded once and shared by all
  requests.
- `@functools.cache` on a factory with no arguments, when building it twice would be harmless.
- Immutable module constants, module-level locks, `atexit` registration.
