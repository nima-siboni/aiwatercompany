# Answer key: L2_filter_registry.py — Factory and registry, level 2 (20 min)

**Pattern focus:** Decorator registries fail quietly. A name gets registered twice, a decorator forgets to return the class, a snapshot of the registry goes stale, or a catch-all `**kwargs` swallows the config it was given. This drill trains you to check what the registry guarantees (unique names, live contents, loud errors), and how construction cost and the timing of validation interact with it.

**Good clarifying questions:** Who imports `genfilters.py`, and when, relative to the campaign API that validates configs? How many candidates go through `screen` per campaign: thousands or millions? Are campaign configs written by hand (so typos are likely)? Can a descriptor be missing for some candidates, for example when Zeo++ fails on a structure? If so, `record["pore_diameter"]` raises `KeyError`, or compares against `None`.

## Blockers (silently wrong results / data loss)
1. **`SurfaceAreaFilter` is registered as `"density"` and silently replaces the density filter.** The decorator line was copied from `DensityFilter` and the name wasn't changed, and `register_filter` just overwrites `FILTERS[name]`. Every campaign that asks for `{"name": "density", "max_density": 1.0}` now gets a surface-area filter with the default `min_area=500`, so dense frameworks pass. Because of #2, the unexpected `max_density` key doesn't even raise. Meanwhile there's no name under which you can request the surface-area filter. *Fix:* register it as `"surface_area"`, and make `register_filter` raise `ValueError` when a name is already taken, so a copy-paste fails at import.

2. **The base class swallows unknown config keys.** `Filter.__init__(self, **kwargs): pass` accepts anything, so `{"name": "pore_size", "min_diamter": 4.0}` silently runs with `min_diameter=3.3` and lets through pores the campaign meant to exclude. It's also what hides #1. The stated reason, allowing a `"comment"` key, doesn't need a catch-all. *Fix:* declare the extra keys you allow (`def __init__(self, comment: str = "")`) and nothing else. Then a typo raises `TypeError: Filter.__init__() got an unexpected keyword argument 'min_diamter'`.

## Major
3. **`register_filter`'s decorator returns `None`, so every decorated class name is `None`.** `decorator` stores the class but has no `return cls`, so after import `PoreSizeFilter`, `DensityFilter` and the rest are all `None`. `make_filter` still works because it goes through `FILTERS`, which is probably why nobody noticed. But `PoreSizeFilter(min_diameter=4.0)` raises `TypeError: 'NoneType' object is not callable`, `isinstance(f, PoreSizeFilter)` raises `TypeError`, subclassing fails at import, and pickling a filter (e.g. to send it to worker processes) raises `PicklingError: ... it's not the same object as ...PoreSizeFilter`. *Fix:* `return cls`.

4. **`screen` rebuilds every filter for every candidate.** `make_filter(entry)` sits inside the comprehension, so 1M candidates × 3 filters means 3M constructions, lookups and dict copies instead of 3. It also means a broken config is only noticed when the first candidate arrives: `screen([], [{"name": "nope"}])` returns `[]` without complaint. *Fix:* build `filters = [make_filter(e) for e in config]` once, then filter. That validates the config up front and leaves only `f.passes(c)` in the loop.

5. **`FILTER_NAMES` is a snapshot taken at import, so filters registered later are rejected.** `FILTER_NAMES = frozenset(FILTERS)` runs when this module finishes executing. `genfilters.py` has to import this module before it can register anything, so its filters are never in the snapshot. `make_filter({"name": "novelty"})` raises `ValueError: unknown filter 'novelty'` even though `FILTERS["novelty"]` exists, and the campaign API rejects those configs too. *Fix:* check the live registry (`name in FILTERS`), or make `FILTER_NAMES` a live view (`FILTERS.keys()`) or a function.

6. **Registration only happens if someone imports `genfilters.py`.** Even with #5 fixed, if the API process never imports it, `"novelty"` is unknown. The error appears far from the cause, and only in some processes, depending on what else they happened to import. *Fix:* import every module that registers filters in one known place (the package's `__init__.py`), use a name → `"module:Class"` table imported lazily, or use entry points for external plugins. Ask which one the team already uses.

## Minor
7. **`MetalFilter(allowed="Zn")` silently rejects everything.** A string is iterable, so `set("Zn")` is `{"Z", "n"}` and no framework passes. Writing `"allowed": "Zn"` instead of `["Zn"]` in JSON is an easy mistake. *Fix:* reject a `str` explicitly (or wrap it in a list).

8. **The errors don't help whoever wrote the config.** An unknown name gives `unknown filter 'pore'` without listing the registered names, and an entry without `"name"` gives a bare `KeyError: 'name'`. *Fix:* check for `"name"`, and list `sorted(FILTERS)` in the message.

## Red herrings (say why they're fine)
- **The `**kwargs` in each subclass's `__init__`.** They're forwarded to `super().__init__(**kwargs)`, which is the normal cooperative pattern. Once the base stops swallowing them, unknown keys still reach the base and fail there. The bug is in the base class, not in the forwarding.
- **A module-level dict mutated without a lock.** Registration happens once, at import, and single dict assignments and lookups are atomic in CPython. Nothing here iterates the registry while another thread changes it.

## Model 1-minute summary
"Two issues silently give wrong screening results. The surface-area filter was registered under `"density"`, so every density filter in every campaign is actually a surface-area cut. And because the base class swallows unknown keys, nothing complains that `max_density` was ignored. The same swallowing means any typo in a config just runs with the defaults. So: enforce unique names in `register_filter`, and no catch-all `**kwargs` in the base. Then the decorator returning `None`, which breaks direct use and pickling of every filter class. Then `screen`, which rebuilds every filter for every candidate and validates nothing when the batch is empty. And `FILTER_NAMES` is frozen at import, so the genfilters filters can never be used; I'd also ask who imports that module at all. Tests I'd want: a duplicate name, a typo'd key, and a filter registered after import."

## Reference fix
`L2_filter_registry_fixed.py`:
- `register_filter` returns the class and raises on a duplicate name. The surface-area filter is registered as `"surface_area"`.
- `Filter.__init__` accepts only `comment`, so unknown keys raise `TypeError`. Subclasses still forward `**kwargs`.
- `FILTER_NAMES` is a live view of `FILTERS`. `make_filter` checks for `"name"` and lists the registered names in its error.
- `screen` builds the filters once, before it looks at any candidate. `MetalFilter` rejects a bare string.
- #6 can't be fixed inside this file: the package has to import `genfilters.py` (noted in the fix's docstring).

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_05_factory_registry.py -k L2 -v`
