# Decorators: review checklist

A decorator wraps a function to add behaviour around every call (logging, timing, retries,
rate limits, validation, registration) without touching its body. The wrapper replaces the
function everywhere, so one mistake in it affects every call site at once.

## When you see it in a PR, check:
- **Return value:** does the wrapper `return func(...)`? If not, every call returns None (and `dtype=float` turns that into NaN).
- **Arguments:** does it take and forward `*args, **kwargs`? If not, the first keyword call raises TypeError.
- **`functools.wraps`:** without it `__name__`, `__qualname__` and `__doc__` are the wrapper's, so registries, metrics and logs keyed by name collide silently.
- **Exceptions:** is the bookkeeping (timing, counters, releasing a slot) in `try/finally`? If not, it's skipped exactly when the call fails.
- **Factory used bare:** `@retry` instead of `@retry()` binds the name to the inner decorator, so calls return a function. Keyword-only factory arguments make it fail at import.
- **Shared state:** `nonlocal` variables and attributes on a decorator object are shared by every call, every thread and, on methods, every instance: backoff that never resets, merged metrics, one rate limit for all.
- **Retry policy:** retries transient errors only, re-raises after the last attempt, doesn't sleep after it, uses jitter and a cap, and never wraps a non-idempotent call (or a whole workflow) without an idempotency key.
- **Async:** a sync wrapper on `async def` just returns the coroutine, so try/except, timing and limits do nothing, and `time.sleep` blocks the event loop.
- **Methods:** a class-based decorator needs `__get__` to bind `self`. Decorators in a class body run once, so their state is per class, not per instance.
- **Stacking order:** they apply bottom-up and run top-down. Put retry above the rate limiter so every attempt is limited, and decide whether timing goes inside or outside retry.
- **Logging arguments:** `args` includes `self`, and a dataclass repr prints secrets. Use lazy `%s` formatting and log IDs and shapes, not payloads.
- **Rewriting arguments:** a "validation" wrapper that edits arrays in place (`nan_to_num(copy=False)`) silently changes the caller's data.
- **Work per call:** `inspect.signature`, regex compiles and config reads belong at decoration time.
- **Clocks:** use `perf_counter()` for durations and `monotonic()` for limits and deadlines, never `time.time()`.

## Usually fine (don't block on it)
- `@register` above `@timed`: it's applied last, so the registry holds the timed function.
- Retrying idempotent GETs on 429/5xx.
- A plain (non-factory) decorator used without parentheses.
- Registries filled at import time, if names are checked for duplicates.

## Idiomatic shape
```python
def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            record(func.__qualname__, time.perf_counter() - start)
    return wrapper
```
