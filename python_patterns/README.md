# Python pattern drills

Thirty short PR reviews, three per Python design pattern, in the same format as the eight full
mocks in `../pr_mock`. Most review findings are instances of a handful of failure modes per
pattern: a decorator that drops the return value, a cache keyed on the wrong thing, a generator
consumed twice. The drills train you to spot those fast, so that in the interview your time goes
on prioritising and explaining rather than on finding.

## Learn first

`learn/` has one file of learning nuggets per pattern, 7–8 short examples each. Every nugget
shows a failure mode, the output it really produces and the fix, and ends with a list of things
that look wrong but are fine. The examples are written for the nuggets, not taken from the
drills, so read a pattern's file before its drills. Afterwards, the pattern's `CHECKLIST.md` in
`answers/` is the full recap; it gives some drill answers away, so save it for after.

## Levels

| Level | Size (lines of code) | Planted issues | Time box | What it trains |
|---|---|---|---|---|
| L1 warm-up | 25–40 | 5–6 | 10 min | the classic pitfalls of the pattern |
| L2 intermediate | 45–75 | 7–9 | 20 min | subtler pitfalls, and how they hit performance and robustness |
| L3 interview | 80–120 | 10–13 | 35 min + 5 min summary | the real format: mixed issues, clarifying questions, prioritisation |

Every drill also contains at least one red herring: code that looks wrong but is fine.

## Drills

| # | Pattern | Learn first | L1 (10 min) | L2 (20 min) | L3 (35 + 5 min) | Cheat sheet |
|---|---|---|---|---|---|---|
| 01 | Decorators | [nuggets](learn/01_decorators.md) | [Time each featuriser and register them in one place](exercises/01_decorators/L1_featuriser_timing.py) | [Retry transient failures when talking to sim-api](exercises/01_decorators/L2_retry_backoff.py) | [Client for the surrogate-model service](exercises/01_decorators/L3_surrogate_client.py) | [checklist](answers/01_decorators/CHECKLIST.md) |
| 02 | Caching and memoisation | [nuggets](learn/02_caching.md) | [Cache element lookups and pair energies in the featuriser](exercises/02_caching/L1_element_lookup.py) | [Cache derived properties on Framework](exercises/02_caching/L2_framework_properties.py) | [Prediction cache for the surrogate service](exercises/02_caching/L3_prediction_cache.py) | [checklist](answers/02_caching/CHECKLIST.md) |
| 03 | Context managers | [nuggets](learn/03_context_managers.md) | [Timing and working-directory helpers for DFT post-processing](exercises/03_context_managers/L1_dft_helpers.py) | [Transactional writes and atomic snapshots for the results store](exercises/03_context_managers/L2_results_store.py) | [SimulationWorkspace context manager for DFT/MD/GCMC jobs](exercises/03_context_managers/L3_simulation_workspace.py) | [checklist](answers/03_context_managers/CHECKLIST.md) |
| 04 | Generators and lazy pipelines | [nuggets](learn/04_generators.md) | [Batched inference for the uptake surrogate model](exercises/04_generators/L1_batched_inference.py) | [Streaming dataset for the DFT relaxation data](exercises/04_generators/L2_streaming_dataset.py) | [Streaming ingest of GCMC results into the results database](exercises/04_generators/L3_gcmc_ingest.py) | [checklist](answers/04_generators/CHECKLIST.md) |
| 05 | Factory and registry | [nuggets](learn/05_factory_registry.md) | [Choose the featuriser from the experiment config](exercises/05_factory_registry/L1_featuriser_factory.py) | [Config-driven screening filters](exercises/05_factory_registry/L2_filter_registry.py) | [Plugin engines for the simulation service](exercises/05_factory_registry/L3_engine_plugins.py) | [checklist](answers/05_factory_registry/CHECKLIST.md) |
| 06 | Strategy and template method | [nuggets](learn/06_strategy_template.md) | [Pluggable acquisition functions for the active-learning loop](exercises/06_strategy_template/L1_acquisition.py) | [Common base class for structure relaxations](exercises/06_strategy_template/L2_relaxer_template.py) | [Formation energies with a DFT -> ML -> surrogate fallback](exercises/06_strategy_template/L3_energy_fallback.py) | [checklist](answers/06_strategy_template/CHECKLIST.md) |
| 07 | Dataclasses and value objects | [nuggets](learn/07_dataclasses.md) | [Value objects for GCMC conditions and job requests](exercises/07_dataclasses/L1_gcmc_conditions.py) | [Frozen Structure value object for generated crystals](exercises/07_dataclasses/L2_frozen_structure.py) | [Typed, frozen configs for active-learning campaigns](exercises/07_dataclasses/L3_campaign_config.py) | [checklist](answers/07_dataclasses/CHECKLIST.md) |
| 08 | Observer and callbacks | [nuggets](learn/08_observer.md) | [In-process event bus for simulation-job events](exercises/08_observer/L1_event_bus.py) | [Callback hooks for the surrogate-model trainer](exercises/08_observer/L2_training_callbacks.py) | [Job-status pub/sub for the orchestration service](exercises/08_observer/L3_job_events.py) | [checklist](answers/08_observer/CHECKLIST.md) |
| 09 | Singleton, global state, DI | [nuggets](learn/09_singleton_di.md) | [One shared Settings object for the screening services](exercises/09_singleton_di/L1_settings_singleton.py) | [Keep one SQLite connection for the GCMC results store](exercises/09_singleton_di/L2_results_db.py) | [Request handlers for the uptake-prediction service](exercises/09_singleton_di/L3_prediction_service.py) | [checklist](answers/09_singleton_di/CHECKLIST.md) |
| 10 | Concurrency | [nuggets](learn/10_concurrency.md) | [Featurise candidate structures in a thread pool](exercises/10_concurrency/L1_parallel_featurise.py) | [Stream generated candidates through a pool of DFT workers](exercises/10_concurrency/L2_dft_queue.py) | [Async client for the property-prediction API](exercises/10_concurrency/L3_async_predict_client.py) | [checklist](answers/10_concurrency/CHECKLIST.md) |

## How to practise

1. Read the pattern's nuggets in `learn/` (5–10 minutes).
2. Open only the file in `exercises/`. Everything in `answers/` and `checks/` is a spoiler.
3. Practise without a linter. CoderPad won't underline unused names or type errors for you, but
   VS Code will. Paste the drill into CoderPad's free candidate sandbox, or switch off Pylance
   for this folder.
4. Start the timer and talk out loud. Write each comment as
   `[severity] function: problem → concrete failure → fix`, using the tiers in `../README.md`.
5. Keep the last few minutes for the summary: the top three must-fix items, the tests you'd ask
   for, and what you'd let go.
6. Compare with the answer key. Score yourself on four things: blockers found, whether your top
   three match the key, whether each comment names a concrete failure, and whether you called
   the red herrings correctly.
7. Read the reference fix (`answers/<pattern>/*_fixed.py`) and the pattern's `CHECKLIST.md`.

## Suggested plans

- **Short on time (about 3 hours):** read the nuggets for concurrency, caching, context managers
  and generators (about 20 min), then do those four L3 drills under interview conditions.
- **Thorough (about 10 hours):** one pattern per sitting: nuggets, then L1 → L2 → L3. Then the
  eight full mocks in `../pr_mock` as dress rehearsals.

## Checks

Every planted bug that can be demonstrated in code has a check in `checks/`. Each check runs
twice: against the PR code (`[pr]`, which must fail and is reported as XFAIL) and against the
reference fix (`[fixed]`, which must pass). Red-herring checks pass on both. Run from `aiwc/`:

```
uv run --with pytest --with numpy pytest python_patterns/checks -v                    # everything
uv run --with pytest --with numpy pytest python_patterns/checks/test_02_caching.py -k L2 -v
uv run --with pytest --with numpy pytest python_patterns/checks -k L2 --runxfail      # show how each bug fails
```

Performance and design issues have no checks; the answer keys explain those.
