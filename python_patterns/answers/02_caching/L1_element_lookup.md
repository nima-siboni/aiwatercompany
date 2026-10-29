# Answer key: L1_element_lookup.py — Caching, level 1 (10 min)

**Pattern focus:** the canonical `lru_cache` pitfalls: a cached mutable value that a caller
changes, arguments that can't be hashed, and an unbounded cache over a key space that never
repeats. Also learn to recognise the case where an unbounded cache is exactly right.

**Good clarifying questions:** Can the elements table change while a job is running? Do
the same distances ever repeat across structures? Who calls `mean_property`, and what do
they pass? How many structures, and how many atoms per structure, does one job featurise?

## Blockers (silently wrong results)

1. **`probe_energy` writes into the cached dicts, so later results are wrong.**
   `element_properties` returns the same dict object on every call, and `probe_energy`
   overwrites its `"epsilon"` and `"sigma"` with the mixed values. The next time that
   element comes up, the already-mixed values are mixed again, drifting towards the probe's.
   This already happens inside one call: the second `"O"` in `["Zn", "O", "O", "C", "H"]` is
   mixed twice, so the first result is −0.2469 kcal/mol instead of −0.2421. Calling it
   again gives 0.449, then 1.222, then 1.783. The sign flips, and the descriptor depends on
   how many structures the process has already featurised. Anything else that reads
   epsilon or sigma from the cache is wrong too. *Fix:* never mutate a cached value. Compute
   the mixed parameters in locals (or new arrays), and make the cached value immutable (a
   `MappingProxyType`, a `NamedTuple` or a frozen dataclass), so any future attempt to
   change it fails loudly.

## Major

2. **Caching `pair_energy` over float distances only costs memory, and time.** Distances
   are continuous, so nearly every call is a miss, and `maxsize=None` keeps every entry for
   the life of the process. At about 140 bytes per entry, featurising 40k MOFs of 500 atoms
   each (20M distances) needs about 2.8 GB, and it keeps growing. It's also slower: a miss
   costs about 150 ns, against about 90 ns to just evaluate the formula (measured). *Fix:*
   drop the cache and vectorise over atoms with numpy. The function body already works
   elementwise on arrays.

3. **`mean_property` raises `TypeError: unhashable type: 'list'`** on exactly the input
   its type hint asks for. `lru_cache` hashes its arguments, and lists (and numpy arrays)
   can't be hashed. *Fix:* drop the cache, since the function is cheap once
   `element_properties` is cached, or convert to a tuple before calling a cached helper.

## Minor

4. **An unknown element gives a confusing error.** `fetchone()` returns None, so
   `element_properties("Xx")` raises `TypeError: cannot unpack non-iterable NoneType object`,
   which doesn't say which symbol it was (a typo, or an unusual label from a CIF). *Fix:*
   check for None and raise `KeyError(f"unknown element {symbol!r}")`. `lru_cache` doesn't
   cache exceptions, so a later fix to the table is picked up.

## Nit

5. **`with sqlite3.connect(...)` doesn't close the connection.** As a context manager, the
   connection only commits or rolls back. CPython closes it when it's garbage-collected, and
   Python 3.13 emits a `ResourceWarning` when that happens. *Fix:*
   `contextlib.closing(sqlite3.connect(...))`. It runs once per element, so this is cosmetic.
6. **`zip(symbols, distances)` stops at the shorter input without complaint**, so a length
   mismatch from an upstream bug gives a plausible-looking partial sum. *Fix:* check the
   lengths (or `zip(..., strict=True)`, Python 3.10+).

## Red herring (say why it's fine)
- **`@lru_cache(maxsize=None)` on `element_properties` is fine.** There are about a hundred
  element symbols and the table doesn't change during a run, so the cache holds at most a
  hundred small entries, and it turns one query per atom into one query per element, which
  is the speed-up the PR is after. The problem with this function is what it returns
  (blocker 1), not the size of its cache.

## Model 1-minute summary
"One blocker: `probe_energy` writes the mixed Lennard-Jones parameters back into the dict
that `element_properties` caches. Every later lookup of that element gets values that are
already mixed, and mixes them again. Results are wrong from the second atom of an element
onwards and drift with every structure; in my example the energy went from −0.25 to +1.8
over four calls. The fix is to compute the mixed values in locals and return a read-only
mapping or a frozen dataclass. Then two misuses of `lru_cache`. `pair_energy` is cached over
continuous distances, so it never hits, grows by gigabytes and is slower than recomputing;
I'd remove the cache and vectorise. `mean_property` can't be cached with a list argument at
all; it raises TypeError. The unbounded cache on element symbols is fine. I'd add a test
that calls `probe_energy` twice and checks both results match a hand calculation."

## Reference fix
`L1_element_lookup_fixed.py`:
- `element_properties` returns a `MappingProxyType`, raises `KeyError` for unknown symbols,
  and its lookup closes the connection.
- `probe_energy` checks that the lengths match, computes the mixed parameters into new
  arrays and sums a vectorised `pair_energy`.
- No cache on `pair_energy` or `mean_property`.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_02_caching.py -k L1 -v`
