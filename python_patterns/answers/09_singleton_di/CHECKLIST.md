# Checklist: singletons, module-level state and dependency injection

A singleton (or any module-level object) gives the whole process one shared instance of something
expensive or process-wide: a model, a connection, the settings. Dependency injection means code
receives those objects instead of reaching for them, so tests and other callers can pass their own.

**When you see it in a PR, check:**
- `__new__` singleton that also defines `__init__` → `__init__` re-runs on every call and resets state.
- Environment or config read at import → later changes (tests, CLI flags) are silently ignored.
- Request- or user-scoped data in a global (`CURRENT_USER`) → stale across requests, shared across threads: data leaks.
- Module-level defaults aliased and mutated (`opts = DEFAULTS; opts.update(...)`) → one call changes every later call.
- Lazy init outside a lock, or published before it's fully built → duplicate objects, half-initialised ones.
- Singleton factory with an argument (`get_db(path)`) → the argument is ignored after the first call.
- One shared connection → one shared transaction: one caller's rollback, or a close without commit, loses everyone's writes.
- Close or reconfigure that leaves references behind → "closed database" errors, caches built from the old dependency.
- Connections or sockets opened before workers `fork()` → several processes share one socket or file handle.
- Per-process counters used as IDs → the IDs repeat after a restart and across workers.
- `functools.cache` keyed by request data → unbounded memory, entries that never refresh.
- `+=` on shared counters from threads → lost updates.
- Code that fetches the global itself → tests must monkeypatch; `reset_for_tests()` ends up in production.

**Usually fine (don't block on it):**
- One read-only object per process (a model, a lookup table), loaded once and shared by all requests.
- `@functools.cache` on a factory with no (or a few fixed) arguments: lazy, per process, `cache_clear()` in tests.
- Immutable module constants, module-level locks, `atexit` registration.
- Double-checked locking in CPython, as long as the object is published only once it's complete.

**Idiomatic shape:**
```python
@dataclass(frozen=True)
class Settings:
    results_db: str
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(os.environ.get("AIWC_RESULTS_DB", "results.sqlite"),
                   os.environ.get("AIWC_DRY_RUN", "").lower() in {"1", "true"})

def main() -> None:  # composition root: build once, pass down
    settings = Settings.from_env()
    store = ResultsStore(sqlite3.connect(settings.results_db))
    run_screen(settings, store)  # tests call run_screen with fakes
```
