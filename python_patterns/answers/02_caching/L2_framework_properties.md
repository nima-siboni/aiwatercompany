# Answer key: L2_framework_properties.py — Caching, level 2 (20 min)

**Pattern focus:** caching on objects that can change. Check whether every cached value is
invalidated by every mutator, what `lru_cache` on a method really keys on (`self`, through
`__eq__` and `__hash__`), what it keeps alive, and whether "normalising" a key quietly
changes the question being asked.

**Good clarifying questions:** Which conditions do callers actually use: a fixed grid (298.15 K;
0.15 and 1.0 bar) or arbitrary values? Is a framework ever changed after it has been asked
for its properties? Do generated candidates really all have `mof_id=None` until they're
registered? How many frameworks does a screen hold, and is the 50 ms model call batchable?

## Blockers (silently wrong results)

1. **Cached volume and density go stale when the framework changes.** `cached_property`
   stores the value in the instance's `__dict__` on first access and never recomputes it.
   `remove_guests` and `strain` change the atoms and the cell but don't clear it, so after
   activation `density` still reports the solvated value (0.809 instead of 0.740 g/cm³ in
   the test structure), and density filters and gravimetric conversions use the wrong
   number. `activated()` has the same bug: `copy.copy` copies `__dict__`, cached values
   included, so if anything read `density` before activation, the new object starts with the
   solvated density. *Fix:* volume and density cost microseconds, so make them plain
   `@property`. If something really is expensive, clear it in every mutator, or make the
   class immutable and have mutators return new objects.

2. **The uptake cache is keyed by `mof_id`, so different structures share results.**
   `lru_cache` on a method includes `self` in the key, and it finds entries through
   `__hash__` and `__eq__`, which both use `mof_id`. Every generated candidate has
   `mof_id=None`, so they're all "equal", and every candidate gets the uptake of whichever
   was scored first at those conditions. The activated copy keeps the original's `mof_id`,
   so it gets the solvated framework's uptake. An in-place `remove_guests` or `strain` is
   never seen either, because the key doesn't change. *Fix:* a per-object cache dict
   (`self._uptake_cache`) that every mutator clears, or a key built from the actual content
   (a hash of species, coordinates and lattice).

3. **"Normalising" the conditions changes what the model is asked.** `uptake` passes
   `round(temperature)` and `round(pressure, 1)` to the model, not only to the cache key.
   `round(0.15, 1)` is `0.1` (0.15 is stored as 0.1499...), so the flue-gas condition is
   predicted at 0.1 bar, which is about a third too low in the low-pressure region where
   uptake is roughly proportional to pressure. `round(0.04, 1)` is `0.0`, so the model is
   asked for uptake at zero pressure. Rounding 298.15 K to 298 K is physically small, but
   it's still not what the caller asked for. *Fix:* key on the exact floats (callers use a
   fixed grid, so they hit anyway). If keys must be quantised, quantise only the key, to a
   documented tolerance well below the physical one, and always compute at the exact
   condition.

## Major

4. **`lru_cache` on a method keeps every framework alive.** The cache holds a strong
   reference to `self` in each key, and with `maxsize=None` it never evicts. Each framework
   the screen touches stays in memory with its arrays: about 17 kB for a 500-atom
   structure, so a 50k screen pins nearly a gigabyte (measured). `del fw` frees nothing, and
   `Framework._uptake.cache_clear()` is the only way out. *Fix:* the per-object cache from
   issue 2, which is freed with the object.

5. **Every unregistered candidate compares equal to every other.** `__eq__` compares
   `mof_id`, which is `None` for all generated candidates, so `set(candidates)` or a dict
   keyed by framework silently keeps only one of them. That's the root cause of issue 2,
   but it bites anywhere frameworks are deduplicated. *Fix:* fall back to identity when
   `mof_id` is None, with `__hash__` kept consistent (`id(self)`), and return
   `NotImplemented` for non-`Framework` operands.

## Minor

6. **There's no way to invalidate one framework.** `lru_cache` on a method is one cache
   for the whole class, so the only reset is `Framework._uptake.cache_clear()`, which
   throws away every framework's entries. That's another reason for a per-object cache.
7. **Memoisation won't make the screen fast.** It only helps the second time a framework
   is asked. The first pass still makes one 50 ms model call per framework and condition:
   50k frameworks at 3 conditions is about 2 hours. *Fix:* batch the descriptors and call
   the model once per batch; cache on top of that if repeats really are common.

## Nit
- `ATOMIC_MASS` is hand-written and partial (no Cd, In, Ca, ...), so `density` raises
  `KeyError` for those elements. That failure is loud, but a shared periodic-table source
  would be better.

## Red herrings (say why they're fine)
- **The `1e-24` in `density` is correct:** 1 Å = 1e-8 cm, so 1 Å³ = 1e-24 cm³, and
  `sum(masses) / AVOGADRO` gives grams. The formula is right; it just goes stale (issue 1).
- **"`cached_property` isn't thread-safe."** True since Python 3.12, which removed its
  lock, so two threads can both compute a value on first access. For a pure, cheap getter
  like `volume`, that's harmless duplicate work, not a bug.

## Model 1-minute summary
"Three blockers, all silently wrong numbers. First, `volume` and `density` are
`cached_property` on an object that `remove_guests` and `strain` change, so after activation
the density is still the solvated one. `activated()` even copies the stale value across with
`copy.copy`. They're cheap, so they should just be properties. Second, `lru_cache` on
`_uptake` keys on `self` through `__eq__`, which compares `mof_id`, so every unregistered
candidate, all `mof_id=None`, shares one cached uptake, and the activated copy gets the
solvated one. Third, `uptake` rounds the conditions before calling the model, so 0.15 bar is
predicted at 0.1 bar, and 0.04 bar at zero. After that, the method cache pins every framework
in memory, and the equality makes `set(candidates)` collapse to one. I'd use a per-object
uptake cache cleared by every mutator, exact keys, and batched model calls for throughput."

## Reference fix
`L2_framework_properties_fixed.py`:
- `volume` and `density` are plain properties.
- `uptake` uses a per-object dict keyed by the exact `(temperature, pressure)`; `remove_guests`
  and `strain` clear it.
- `activated()` builds a new `Framework` instead of `copy.copy`, so it can't share the cache.
- `__eq__`/`__hash__` fall back to identity when `mof_id` is None, and `__eq__` returns
  `NotImplemented` for other types.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_02_caching.py -k L2 -v`
