# Answer key: L1_gcmc_conditions.py — Dataclasses, level 1 (10 min)

**Pattern focus:** what the `@dataclass` flags actually generate. `order=True` compares
fields in declaration order, `unsafe_hash=True` hashes mutable state, and nothing validates
fields unless you write `__post_init__`. A value object should be frozen.

**Good clarifying questions:** Is a higher `priority` number more urgent? (The docstring says
so.) Are `Condition`s used as keys in the results cache? Do callers keep using the `base`
condition after building an isotherm? Which temperature range do the force fields support?

## Blockers (silently wrong results)

1. **`schedule` sorts by MOF id, not by priority.** `order=True` makes `JobRequest` compare
   like the tuple `(mof_id, condition, priority, tags)`, so `sorted(jobs, reverse=True)`
   sorts reverse-alphabetically by `mof_id`. Priority only breaks ties between jobs for the
   same MOF and condition. With `ZIF-8` (priority 0), `HKUST-1` (10) and `MOF-5` (5), the
   scheduler submits ZIF-8, MOF-5, HKUST-1: the urgent job goes last. *Fix:* drop
   `order=True` and use `sorted(jobs, key=lambda j: j.priority, reverse=True)`. Python's sort
   is stable even with `reverse=True`, so equal priorities keep their submission order, as
   the docstring promises.

2. **`pressure_sweep` returns the same object N times.** It sets `base.pressure = p` and
   appends `base` on every iteration, so `isotherm_jobs("ZIF-8", Condition(298.0, 1.0),
   [0.1, 1.0, 10.0])` returns three jobs that all run at 10 bar. `unique_conditions` then
   reports one condition, and the caller's `base` now says 10 bar as well. *Fix:* make
   `Condition` frozen and build new objects with
   `[dataclasses.replace(base, pressure=p) for p in pressures]`.

## Major

3. **`unsafe_hash=True` on a mutable class.** The hash is computed from the current field
   values, so a `Condition` that is mutated after it's been used as a dict key or set member
   is lost. With `cache = {c: 2.1}` followed by `c.pressure = 5.0`, neither
   `Condition(298.0, 1.0)` nor `Condition(298.0, 5.0)` finds the entry. That is exactly what
   `pressure_sweep` does to `base`. The usual story is that someone hit
   `TypeError: unhashable type` and reached for `unsafe_hash`. *Fix:* `frozen=True` (and no
   `unsafe_hash`). The generated `__hash__` is then safe, and mutation raises
   `FrozenInstanceError`.

4. **No validation.** `Condition(-5.0, 1.0)` and `Condition(298.0, -1.0)` are accepted, and so
   is `Condition(25.0, 1.0)` written by someone thinking in Celsius. The job then fails hours
   later on the cluster or returns a nonsense uptake. *Fix:* a `__post_init__` that raises
   `ValueError` for T ≤ 0 K or P ≤ 0 bar. Ask what temperature range the force fields
   support; a range check would also catch Celsius passed by mistake.

## Minor

5. **Float fields in equality and hashing.** `Condition.from_celsius(-196.15, 1.0, "N2")` (77 K,
   the standard nitrogen isotherm) is `76.99999999999997` K, so it isn't equal to
   `Condition(77.0, 1.0, "N2")`. Both survive `unique_conditions`, and a cache keyed on
   conditions misses, which means a duplicate simulation. *Fix:* canonicalise once in
   `__post_init__`, e.g. round the temperature to 1e-6 K with
   `object.__setattr__(self, "temperature", ...)` (the standard idiom inside a frozen class's
   `__post_init__`). A tolerance in `__eq__` doesn't work, because equal objects must have
   equal hashes.

## Nit
- Put units in the names (`temperature_k`, `pressure_bar`) rather than in comments.

## Red herring (say why it's fine)
- **`tags: list[str] = field(default_factory=list)`** is the correct way to give each job its
  own list; appending to one job's tags doesn't touch another's. (A plain `= []` default would
  be rejected at class creation with `ValueError`.) `order=True` on
  `Condition` is also fine: sorting conditions by temperature, then pressure, is what you want
  for an isotherm table.

## Model 1-minute summary
"Two things are wrong without any error. First, `order=True` on `JobRequest` compares fields
in declaration order, so `schedule` sorts reverse-alphabetically by MOF id and priority is
only a tie-breaker. I'd drop `order=True` and sort with an explicit key. Second,
`pressure_sweep` mutates and re-appends the same `Condition`, so an isotherm comes back as N
copies of the last pressure, and the caller's base condition changes too. The root cause is
that `Condition` is mutable but hashable through `unsafe_hash`, so any later mutation also
loses it from dicts and sets. Make it `frozen=True` and use `dataclasses.replace`. Then add
`__post_init__` validation for non-positive temperature and pressure, and canonicalise the
temperature so 77 K converted from Celsius matches 77 K typed in. `default_factory=list` is
fine. I'd add tests for the schedule order and for the pressures of an isotherm."

## Reference fix
`L1_gcmc_conditions_fixed.py`:
- `Condition` is `frozen=True` without `unsafe_hash`, so it is immutable and properly hashable.
- `__post_init__` rejects T ≤ 0 and P ≤ 0, and rounds the temperature to 1e-6 K.
- `pressure_sweep` builds new objects with `dataclasses.replace`.
- `JobRequest` drops `order=True`, and `schedule` sorts by an explicit priority key.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_07_dataclasses.py -k L1 -v`
