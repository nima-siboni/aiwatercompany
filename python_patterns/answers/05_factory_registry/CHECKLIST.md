# Factory and registry: review checklist

A factory turns a name, usually from config, into an object; a registry is the table of names it draws from, often filled by decorators or plugin imports. They decouple callers from concrete classes but move errors from import time to run time, so the review is about what fails loudly and what fails silently.

## When you see it in a PR, check:
- **Classes, not instances, in the table.** `{"a": A(**kw), "b": B(**kw)}[name]` builds everything on every call: slow, side effects, and a crash on kwargs meant for another class.
- **Config is read, not mutated.** `config.pop(...)` empties the caller's dict, so the next call (next fold, next sweep point) silently gets the defaults.
- **Unknown names fail loudly and helpfully.** No `if/elif` chain that falls through to `None`; the error lists the valid names.
- **Missing or misspelled keys don't fall back to defaults.** `config.get("kind", "default")` hides typos; make the key required and reject unknown keys.
- **No catch-all `**kwargs` at the root of the constructor chain.** It swallows typo'd keys, which then run with defaults. Forwarding `**kwargs` to `super()` is fine; swallowing them isn't.
- **Registering decorators return what they decorate.** Without `return cls`, the module-level name becomes `None`: direct use, `isinstance`, subclassing and pickling all break.
- **Duplicate names are rejected.** A copy-pasted `@register("x")` silently replaces the first class, and import order decides which one wins.
- **Names are explicit.** Names derived from `cls.__name__` collide across modules and change when someone renames a class.
- **Abstract and helper base classes aren't registered.** `__init_subclass__` auto-registration also registers intermediates, which then appear as options and fail at instantiation.
- **The registry is read live.** `NAMES = frozenset(REGISTRY)` at import is a snapshot that never sees later registrations.
- **Something guarantees that registering modules get imported.** Import-side-effect registration fails with "unknown name" far from the cause; import them in one known place, or use entry points.
- **Modules are imported by name only from trusted config.** `importlib.import_module(request["module"])` runs code the caller chose.
- **Cached instances are keyed on everything that configures them, and hold no per-request state.** A name-only cache hands the first caller's params to everyone; shared mutable instances leak state across requests and threads.
- **The factory isn't called in a hot loop.** Build once, then reuse.

## Usually fine (don't block on it)
- A module-level registry dict filled at import time, without a lock.
- Subclasses forwarding `**kwargs` to `super().__init__`, as long as the root doesn't swallow them.
- `importlib.import_module` on every call: after the first import it's a `sys.modules` lookup.
- A dict of classes or an `if/elif` chain for a handful of options: either is fine if unknown names raise.

## Idiomatic shape
```python
REGISTRY: dict[str, type] = {}

def register(name: str):
    def decorator(cls):
        if name in REGISTRY:
            raise ValueError(f"{name!r} is already registered by {REGISTRY[name].__qualname__}")
        REGISTRY[name] = cls
        return cls
    return decorator

def build(config: dict):
    name = config["name"]
    if name not in REGISTRY:
        raise ValueError(f"unknown {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**config.get("params", {}))
```
