# Dataclasses and value objects: learning nuggets

Dataclasses generate `__init__`, `__repr__`, `__eq__` and, optionally, ordering and hashing from
the field list. Value objects (conditions, structures, configs) use them to get equality by
value, safe dict keys, and one validated place where data enters. Most bugs come from assuming
the generated methods do more than they do.

The examples are written for these notes, not taken from the drills, and the outputs in comments
come from running them on Python 3.14. After the drills, the
[cheat sheet](../answers/07_dataclasses/CHECKLIST.md) is the full checklist.

## 1. `order=True` compares fields in declaration order

```python
@dataclass(order=True)
class Candidate:
    name: str
    uptake: float

cands = [Candidate("UiO-66", 2.5), Candidate("HKUST-1", 4.1), Candidate("MOF-5", 1.3)]
max(cands).name                            # 'UiO-66': the alphabetically last name
max(cands, key=lambda c: c.uptake).name    # 'HKUST-1'
```

**Fix:** sort with an explicit `key=`. Use `order=True` only when the field order really is the
ordering you want.

## 2. `frozen=True` is shallow

```python
@dataclass(frozen=True)
class Cell:
    lengths: list

c = Cell([10.0, 10.0, 10.0])
c.lengths = [1.0]        # FrozenInstanceError: cannot assign to field 'lengths'
c.lengths[0] = 1.0       # allowed: Cell(lengths=[1.0, 10.0, 10.0])
hash(c)                  # TypeError: unhashable type: 'list'
```

**Fix:** use immutable field types (tuples, frozensets). For arrays, store a copy of the caller's
array, not the array itself, and mark it read-only; otherwise edits leak in both directions.

## 3. Array fields and the generated `__eq__`

```python
@dataclass
class Frame:
    positions: np.ndarray

Frame(np.zeros(3)) == Frame(np.zeros(3))   # array([ True,  True,  True]), not True
if Frame(np.zeros(3)) == Frame(np.zeros(3)): ...
# ValueError: The truth value of an array with more than one element is ambiguous. ...
```

So `in`, `list.index`, `list.remove` and deduplication all raise. On Python 3.13 and later the
`==` returns the array; on 3.12 it raises at the `==` itself. **Fix:** `field(compare=False)` on
the array plus an explicit comparison (`np.array_equal` or `np.allclose`), or store tuples.

## 4. Type hints are not enforced

```python
@dataclass
class RunSettings:
    n_steps: int
    dry_run: bool = False

s = RunSettings(n_steps="1000", dry_run=os.environ.get("DRY_RUN", "false"))
s                        # RunSettings(n_steps='1000', dry_run='false')
bool(s.dry_run)          # True: the string "false" is truthy
```

**Fix:** convert and validate in `__post_init__`: `int(...)`, range checks, and explicit boolean
parsing (`value.lower() in {"1", "true", "yes"}`).

## 5. Floats in `__eq__` and `__hash__`

```python
@dataclass(frozen=True)
class Pressure:
    bar: float

Pressure(0.1 + 0.2) == Pressure(0.3)                   # False
{Pressure(0.3): "cached"}.get(Pressure(0.1 + 0.2))     # None
```

The same value reached by two code paths doesn't match, so caches miss and deduplication fails.
**Fix:** canonicalise once in `__post_init__`: round to a documented tolerance, or store integer
units such as millibar.

## 6. Shared defaults and shallow copies

```python
DEFAULT_TAGS = ["screening"]

@dataclass
class Job:
    name: str
    tags: list = field(default_factory=lambda: DEFAULT_TAGS)   # the same list every time

a, b = Job("a"), Job("b")
a.tags.append("urgent")
b.tags                   # ['screening', 'urgent'], and DEFAULT_TAGS changed too
```

**Fix:** `default_factory=lambda: list(DEFAULT_TAGS)`, or a tuple. `dataclasses.replace` is
shallow in the same way: after `j2 = replace(j1, name="relax-retry")`, appending to `j2.tags`
changed `j1.tags` too.

## 7. A subclass `__post_init__` must call the base one

```python
@dataclass
class Grid:
    n: int

    def __post_init__(self):
        if self.n <= 0:
            raise ValueError("n must be positive")

@dataclass
class LabelledGrid(Grid):
    label: str = ""

    def __post_init__(self):
        self.label = self.label.upper()      # no super().__post_init__()

LabelledGrid(n=-5, label="k-points")         # accepted: LabelledGrid(n=-5, label='K-POINTS')
```

**Fix:** call `super().__post_init__()` first.

## 8. `cls(**d)` doesn't rebuild nested dataclasses

```python
@dataclass
class KGrid:
    n: int

@dataclass
class Config:
    kgrid: KGrid

cfg = Config(**{"kgrid": {"n": 4}})          # e.g. straight from json.load
cfg.kgrid.n              # AttributeError: 'dict' object has no attribute 'n'
```

**Fix:** a `from_dict` classmethod that builds the nested objects (`KGrid(**d["kgrid"])`), with
validation in each `__post_init__`.

## Usually fine (don't flag it)

- `field(default_factory=list)`. A bare `= []` default fails when the class is created anyway.
- `dataclasses.replace` to "modify" a frozen object: it re-runs `__post_init__`.
- `object.__setattr__` inside a frozen class's `__post_init__`: the standard way to normalise a
  field.
- A frozen instance as a default value, as long as its own fields are immutable too.
