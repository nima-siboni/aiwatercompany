# Checklist: dataclasses and value objects

Dataclasses generate `__init__`, `__repr__`, `__eq__` and optionally ordering and hashing from
field declarations. Value objects (conditions, structures, configs) use them to get equality by
value, safe dict keys, and one validated place where data enters.

**When you see it in a PR, check:**
- `order=True`: compares in declaration order, so `sorted(objs)` sorts by the first field. Use `key=`.
- `unsafe_hash=True` or a custom `__hash__` on a mutable class: mutation loses it from sets and dicts.
- `frozen=True` is shallow: in-place edits to arrays, lists or dicts corrupt "immutable" objects.
- Caller's arrays stored by reference, or written in `__post_init__`: aliasing in both directions.
- ndarray fields with the generated `__eq__`: raises `ValueError` or returns an array, so dedupe breaks.
- Float fields in `__eq__`/`__hash__`: the same value from two code paths doesn't match. Canonicalise.
- No `__post_init__` validation: wrong units and impossible values travel until a job fails.
- Type hints aren't enforced: JSON lists for tuples, env strings for everything (`"false"` is truthy).
- Subclass `__post_init__` without `super().__post_init__()`: the base checks silently vanish.
- `default_factory=lambda: SOME_CONSTANT`: every instance shares that one object.
- `object.__setattr__` outside `__post_init__`: bypasses frozen and validation. Use `replace`.
- `from_dict` as `cls(**d)`: nested dataclasses stay dicts, so the round trip breaks.
- `asdict()` for hashing: it ignores `compare=False` and deep-copies everything.
- `replace()` is shallow: the copy shares mutable fields with the original.

**Usually fine (don't block on it):**
- `field(default_factory=list)`. A bare `= []` default fails at class creation anyway.
- A frozen dataclass instance as a default value: sharing it is safe as long as its fields are immutable too (a dict field is still shared and mutable).
- `dataclasses.replace` to "modify" a frozen object: it re-runs `__post_init__`.
- `functools.cached_property` on a frozen dataclass (but not with `slots=True`).
- `object.__setattr__` *inside* a frozen class's `__post_init__`: the standard idiom.

**Idiomatic shape:**
```python
@dataclass(frozen=True)
class Condition:
    temperature_k: float
    pressure_bar: float
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if self.temperature_k <= 0 or self.pressure_bar <= 0:
            raise ValueError(f"invalid condition: {self}")
        object.__setattr__(self, "tags", tuple(self.tags))  # normalise once, here

hotter = dataclasses.replace(cond, temperature_k=313.0)  # copy and re-validate; never mutate
```
