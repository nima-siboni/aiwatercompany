# Factory and registry: learning nuggets

A factory turns a name, usually from config, into an object. A registry is the table of names it
draws from, often filled by decorators or by importing plugins. Both move errors from import time
to run time, so the review question is what fails loudly and what fails silently.

The examples are written for these notes, not taken from the drills, and the outputs in comments
come from running them on Python 3.14. After the drills, the
[cheat sheet](../answers/05_factory_registry/CHECKLIST.md) is the full checklist.

## 1. Store classes, not instances

```python
OPTIMISERS = {"fire": Fire(steps=100), "bfgs": BFGS(steps=100)}   # all built at import

def make(name):
    return OPTIMISERS[name]

quick = make("fire")
quick.steps = 5                            # one caller's tweak...
make("fire").steps                         # 5: ...is everyone's; it's the same object
```

Building everything up front is also slow, has side effects, and crashes on settings meant for
another class. **Fix:** store the classes, `{"fire": Fire, "bfgs": BFGS}`, and build per call
with `cls(**params)`.

## 2. Read the config; don't mutate it

```python
def build(cfg):
    steps = cfg.pop("steps", 100)
    return Fire(steps=steps)

cfg = {"steps": 500}
build(cfg).steps, build(cfg).steps         # (500, 100): the second call silently got the default
```

`pop` empties the caller's dict, so the next fold or sweep point runs with defaults. **Fix:**
`cfg.get(...)`, or work on a copy.

## 3. Unknown names must fail loudly, and say what's valid

```python
def make(name):
    if name == "fire":
        return Fire()
    elif name == "bfgs":
        return BFGS()

opt = make("BFGS")       # None: it falls through
opt.run()                # AttributeError: 'NoneType' object has no attribute 'run', far from the cause
```

**Fix:** `raise ValueError(f"unknown optimiser {name!r}; choose from {sorted(REGISTRY)}")`. Make
the config key required too: `cfg.get("kind", "default")` turns a typo into the default.

## 4. A registering decorator must return the class

```python
def register(name):
    def decorator(cls):
        REGISTRY[name] = cls                # no `return cls`
    return decorator

@register("fire")
class Fire: ...

Fire                     # None
Fire()                   # TypeError: 'NoneType' object is not callable
```

Direct use, `isinstance`, subclassing and pickling all break, but lookups through the registry
still work, so tests that only go through the registry pass.

## 5. Duplicates and snapshots

```python
@register("fire")
class Fire: ...

@register("fire")        # copy-pasted
class FireV2: ...

REGISTRY["fire"]         # FireV2: the first class was silently replaced, and import order decides

NAMES = frozenset(REGISTRY)    # a snapshot taken at import

@register("lbfgs")
class LBFGS: ...

sorted(NAMES)            # ['fire']: the snapshot never sees later registrations
```

**Fix:** raise on duplicate names, and read the registry live. Also make sure the modules that
register get imported in one known place; otherwise the error is "unknown name", far from the
cause.

## 6. A catch-all `**kwargs` swallows typos

```python
class Fire:
    def __init__(self, steps=100, **kwargs):
        self.steps = steps

Fire(step=500).steps     # 100: the typo was accepted and ignored
```

Without `**kwargs`, Python 3.14 says `TypeError: Fire.__init__() got an unexpected keyword
argument 'step'. Did you mean 'steps'?`. Forwarding `**kwargs` to `super().__init__()` is fine;
swallowing them at the root of the chain isn't.

## 7. Importing code by name from a request

`importlib.import_module(request["plugin"])` runs whatever module the caller names. Only import
names from trusted config, or check them against an allowlist.

## Usually fine (don't flag it)

- A module-level registry dict filled at import time, without a lock.
- A dict of classes or an `if/elif` chain for a handful of options, as long as unknown names
  raise.
- `importlib.import_module` on every call: after the first import it's a `sys.modules` lookup.
