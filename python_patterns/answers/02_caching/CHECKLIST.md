# Caching: review checklist

A cache trades memory for time by remembering the results of expensive, deterministic calls.
It's only correct if the key captures everything the result depends on and nobody can
change a cached value after it's stored.

## When you see it in a PR, check:
- **Key completeness:** does the key include everything the result depends on (model or featuriser version, conditions, config)? If not, results go stale after a reload or a change.
- **Key precision:** rounding a key merges different inputs (0.15 bar becomes 0.1), and rounding before computing changes the answer. Quantise only to a documented tolerance, and compute at the exact value.
- **Key stability:** built-in `hash()` of `str`/`bytes` is salted per process, so never persist or share it. Storing only a hash hides collisions. Use SHA-256 of a canonical encoding (dtype, shape, bytes).
- **Mutable values:** dicts, lists and arrays handed out by reference get edited by callers, which poisons the cache. Return immutable values or copies, or make arrays read-only.
- **Hashability:** `lru_cache` hashes its arguments, so lists and arrays raise TypeError.
- **Key space:** `maxsize=None` over continuous floats or per-request inputs means unbounded memory, almost no hits, and calls slower than recomputing.
- **Methods:** `lru_cache` on a method keys on `self`. It pins every instance, shares one `maxsize` across them, and matches through `__eq__`/`__hash__`, so value-equal objects share entries.
- **Invalidation:** `cached_property` on a mutable object goes stale when a mutator runs, and `copy.copy` copies the cached values. Clear in every mutator, or don't cache cheap things.
- **Failures:** never cache exceptions or `None` from a transient error, and use a sentinel to tell "cached None" from a miss.
- **Eviction:** an LRU must `move_to_end` on every hit; otherwise it's FIFO.
- **Threads:** `if k in c: return c[k]` races with eviction, and counters lose updates. Hold one lock per operation and do a single `get` with a sentinel.
- **Side tables:** per-key stats must be evicted with their keys.
- **Persistence:** write atomically (temp file plus `os.replace`), survive an unreadable file at start-up, and make sure keys mean the same thing in the next process.
- **Batching:** per-item memoisation doesn't fix a per-item model call. Batch the misses.

## Usually fine (don't block on it)
- An unbounded `lru_cache` over a small fixed domain (element symbols, config names).
- `cached_property` computing twice under threads (no lock since 3.12) when the getter is pure.
- `pickle` for a file that only this service writes.
- Content-addressed keys (hash of the input) instead of IDs.

## Idiomatic shape
```python
@functools.lru_cache(maxsize=None)           # small, fixed key space
def element(symbol: str) -> ElementProps:    # frozen dataclass: immutable
    ...

def cache_key(x: np.ndarray, model_version: str) -> str:
    x = np.ascontiguousarray(x, dtype=np.float64)
    h = hashlib.sha256(f"{model_version}|{x.shape}|".encode())
    h.update(x.tobytes())
    return h.hexdigest()
```
