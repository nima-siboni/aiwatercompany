# Caching and memoisation: learning nuggets

A cache trades memory for time by remembering results. It's correct only if the key captures
everything the result depends on, nobody can change a value once it's stored, and failures
aren't stored as if they were results.

The examples are written for these notes, not taken from the drills, and the outputs in comments
come from running them on Python 3.14. After the drills, the
[cheat sheet](../answers/02_caching/CHECKLIST.md) is the full checklist.

## 1. `lru_cache` needs hashable arguments

```python
@functools.lru_cache(maxsize=None)
def formula_mass(elements): ...

formula_mass(["Zn", "O"])                  # TypeError: unhashable type: 'list'
formula_mass(np.array([65.38, 16.0]))      # TypeError: unhashable type: 'numpy.ndarray'
formula_mass(("Zn", "O"))                  # works
```

**Fix:** pass tuples or frozensets. For arrays, key on a hash of the bytes (plus dtype and shape)
rather than on the array.

## 2. Callers can edit cached values

```python
@functools.lru_cache(maxsize=None)
def default_params(code):
    return {"cutoff": 12.0, "cycles": 5000}

p = default_params("gcmc")
p["cycles"] = 10                           # "just for a quick test"
default_params("gcmc")                     # {'cutoff': 12.0, 'cycles': 10}, for every later caller
```

The cache hands out the same object every time. **Fix:** return immutable values (a tuple, a
frozen dataclass, `types.MappingProxyType`), return copies, or mark cached arrays read-only with
`arr.setflags(write=False)`.

## 3. `lru_cache` on a method keeps every instance alive

```python
class Potential:
    def __init__(self, name):
        self.name = name

    @functools.lru_cache(maxsize=None)
    def energy(self, config_id): ...

pot = Potential("uff")
pot.energy(1)
del pot          # a weak reference to pot was still alive after this and gc.collect()
```

`self` is part of the cache key, so the cache holds every instance forever. The cache is also
shared by all instances (one `maxsize` for all of them), and instances that compare equal share
entries. **Fix:** cache per instance (a dict on `self`, or `functools.cached_property` for a value
without arguments), or use a module-level function keyed on an explicit model version.

## 4. `cached_property` goes stale when the object changes

```python
class Cell:
    def __init__(self, lengths):
        self.lengths = list(lengths)

    @functools.cached_property
    def volume(self):
        return math.prod(self.lengths)

    def scale(self, factor):
        self.lengths = [x * factor for x in self.lengths]

c = Cell([10.0, 10.0, 10.0])
c.volume             # 1000.0
c.scale(2.0)
c.volume             # 1000.0; the true volume is now 8000.0
```

**Fix:** clear it in every method that changes the object (`self.__dict__.pop("volume", None)`),
make the object immutable, or don't cache something this cheap.

## 5. Don't store failures, and don't mistake a cached falsy value for a miss

```python
def uptake(mof_id):
    if mof_id in _cache:
        return _cache[mof_id]
    try:
        value = fetch(mof_id)
    except TimeoutError:
        value = None                       # a transient failure...
    _cache[mof_id] = value                 # ...now stored as "no data", for good
    return value
```

The database was slow once, and every later `uptake("MOF-5")` returned `None` without asking
again. The opposite mistake:

```python
cached = _cache.get(mof_id)
if cached:            # a cached 0.0 (a metallic MOF's band gap) counts as a miss
    return cached
```

That function recomputed its value on every call. **Fix:** cache only successes, and use a
sentinel: `MISSING = object()`, then `value = _cache.get(key, MISSING)` and
`if value is not MISSING:`.

## 6. The key must include everything the result depends on

```python
CUTOFF = 6.0

@functools.lru_cache(maxsize=None)
def neighbour_shell_volume(mof_id):
    return 4 / 3 * math.pi * CUTOFF**3     # uses CUTOFF, which isn't part of the key

neighbour_shell_volume("MOF-5")            # 904.78
CUTOFF = 8.0
neighbour_shell_volume("MOF-5")            # 904.78 again; it should be 2144.66
```

Globals, settings, model or featuriser versions and the content of input files all count.
Rounding a key goes wrong the other way: inputs that differ by less than the rounding step share
one result. **Fix:** pass everything the result depends on as an argument, and include versions in
the key.

## 7. An LRU must refresh an entry on every hit

```python
class LRU:
    def __init__(self, capacity):
        self.capacity, self.data = capacity, OrderedDict()

    def get(self, key):
        return self.data.get(key)          # doesn't mark the key as recently used

    def put(self, key, value):
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)  # evicts the oldest insertion

c = LRU(2)
c.put("a", 1); c.put("b", 2)
c.get("a")                                 # "a" is the hot entry...
c.put("c", 3)
list(c.data)                               # ['b', 'c']: ...and it was evicted
```

Without `self.data.move_to_end(key)` on each hit, this is a FIFO queue. Under threads,
`if key in c: return c[key]` can also race with eviction; hold one lock per operation.

## Usually fine (don't flag it)

- An unbounded `lru_cache` over a small, fixed set of keys (element symbols, config names).
- `cached_property` computing twice under threads, when the getter is pure.
- Content-addressed keys (a hash of the input) instead of IDs.
- `pickle` for a cache file that only this service writes and reads.
