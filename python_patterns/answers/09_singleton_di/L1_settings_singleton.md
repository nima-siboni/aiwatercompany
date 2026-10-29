# Answer key: L1_settings_singleton.py — Singleton and dependency injection, level 1 (10 min)

**Pattern focus:** A `__new__`-based singleton shares one object, but Python still runs
`__init__` on every call. Check when the state is created, when the environment is read, and
whether code that uses the singleton can be tested without mutating process-wide state.

**Good clarifying questions:** Is `/shared/aiwc/results.sqlite` the production results store (AIWaterCompany)?
Who calls `override`: tests, notebooks, CLI flags? Does anything set environment variables after
import (a CLI that maps `--db` to `AIWC_RESULTS_DB`, pytest's `monkeypatch.setenv`)? Where do
the settings get logged?

## Blockers (silently wrong results / data loss)

1. **Every `Settings()` call wipes the overrides.** `__new__` returns the cached instance, but
   when `__new__` returns an instance of the class, Python calls `__init__` on it, every time.
   So `self.results_db = RESULTS_DB` and the environment reads run again on each `Settings()`.
   A test that does `Settings().override(results_db="scratch.sqlite")` and then calls code
   that reads `Settings().results_db` writes into `/shared/aiwc/results.sqlite`: test rows
   land in the shared results store. A notebook's `override(dry_run=True)` is undone by the
   next `Settings()` call inside the runner, so real jobs get submitted. *Fix:* initialise once.
   Load the values in `__new__` when the instance is created and drop `__init__` (or guard it
   with an `_initialised` flag). Better still, see #6.

## Major

2. **`AIWC_RESULTS_DB` is read at import, the other variables at call time.** `RESULTS_DB` is
   captured when the module is first imported. `monkeypatch.setenv("AIWC_RESULTS_DB", tmp)` in
   a test, or a CLI that sets the variable from a `--db` flag after the import, has no effect,
   while `AIWC_GCMC_CYCLES` set at the same moment does. With a default that points at the
   shared store, the failure is again "tests write into the shared DB". *Fix:* read every
   variable in one place when the settings are built, and ask whether anything should default
   to the shared store at all.

3. **`dry_run` is a string.** `os.environ.get("AIWC_DRY_RUN", False)` returns the raw string
   when the variable is set, and every non-empty string is truthy: `AIWC_DRY_RUN=0` or
   `AIWC_DRY_RUN=false` turns dry-run *on*, and the runner silently submits nothing. The type
   also changes: `bool` when unset, `str` when set. *Fix:* parse it explicitly, e.g.
   `value.strip().lower() in {"1", "true", "yes", "on"}`.

4. **`__repr__` prints the API token.** `f"Settings({self.__dict__})"` includes `api_token`, so
   `logger.info("starting with %s", Settings())`, an error reporter that captures local
   variables, or a notebook cell output leaks it. *Fix:* keep secrets out of `repr` (mask them,
   or use a dataclass field with `repr=False`).

## Minor

5. **`override` accepts any name.** `setattr` with a typo, `override(gcmc_cylces=500)`, creates
   a new attribute and leaves `gcmc_cycles` at 5000 without a word. *Fix:* reject names that
   aren't existing settings. With a dataclass, `dataclasses.replace` raises `TypeError` on an
   unknown field.

6. **The singleton is a hidden global dependency.** Any function that calls `Settings()`
   internally can only be tested by mutating process-wide state. Once #1 is fixed, an override
   made in one test persists into every later test in the same process, so tests become
   order-dependent. *Fix:* build the settings once at the entry point (`Settings.from_env()`) and
   pass them to the code that needs them. If a global accessor has to stay, give the test suite
   a fixture that restores it.

## Red herring (say why it's fine)
- **`@cache def element_table(path)`** looks like another hidden singleton, but it's the
  idiomatic lazy, per-process one. It's built on first use rather than at import, keyed by its
  argument, and reset in tests with `element_table.cache_clear()`. Two caveats are worth one
  sentence. Every caller gets the same dict, so nobody may mutate it (return a
  `MappingProxyType` to enforce that). And two threads making the first call at the same moment
  can both parse the file (the `functools` docs say so), which is harmless here.
  `DEFAULT_TEMPERATURE_K` is an immutable module constant, which is fine too.

## Model 1-minute summary
"The blocker is the singleton itself. `__new__` hands back the cached object, but Python still
runs `__init__` on every `Settings()` call, so any override is wiped the next time anyone asks for
the settings. With a default that points at the shared results store, a test that overrides
`results_db` ends up writing into production data. Next is the environment handling:
`AIWC_RESULTS_DB` is frozen at import while the other variables are re-read, and `AIWC_DRY_RUN=0`
counts as true because it's a non-empty string. `__repr__` puts the API token into any log line
that prints the settings. Smaller: `override` silently accepts typos. Longer term I'd build the
settings once at the entry point and pass them in, rather than have code reach for a global. The
cached `element_table` is fine as it is."

## Reference fix
`L1_settings_singleton_fixed.py`:
- Loads from the environment once, in `__new__`, and has no `__init__`, so later `Settings()`
  calls can't reset anything.
- Reads every variable when the instance is built (no import-time `RESULTS_DB`) and parses
  `AIWC_DRY_RUN` into a real `bool`.
- `override` raises `AttributeError` on unknown names; `repr` masks the token.
- Keeps the singleton API so the same checks run on both files. #6 (passing the settings in) is
  the design change I'd ask for in a follow-up PR.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_09_singleton_di.py -k L1 -v`
