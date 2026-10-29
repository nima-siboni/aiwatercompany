# Answer key: L3_energy_fallback.py — Strategy and template method, level 3 (35 min + 5 min summary)

**Pattern focus:** A family of interchangeable calculators with a shared base class (caching), cooperative mixins (retry, timing) and an orchestrator that falls back through them. This drill trains you to check that every implementation keeps the same contract (units!), that the mixin chain actually initialises, that a cache in the base class is keyed on everything the result depends on, and that the fallback only catches the failures it means to catch.

**Good clarifying questions:**
- Are these energies used as training labels for the ML potential? Then a wrong value from the fallback feeds straight back into the model.
- What does `run_dft` raise when the SCF doesn't converge or the job times out, as opposed to a lost node?
- Can a structure's geometry change under the same `id`, for example when it's relaxed in place? The cache would return the old energy.
- Is `update_model` called on a different thread from `compute_energy`?
- How many structures go through `compute_energy` per round? It's serial, and `run_dft` blocks for the whole cluster job.

## Blockers (silently wrong results / data loss)
1. **The surrogate returns eV per atom; everything else returns eV per cell.** `Calculator.energy` documents "eV per unit cell", and DFT and the ML potential (which sums per-atom contributions) follow it. `CompositionSurrogate` was "fitted to our DFT formation energies per atom" and returns that per-atom value. For an 8-atom cell at −1.75 eV/atom the chain returns −1.75 instead of −14.0; for a 200-atom MOF it's off by a factor of 200. Whenever DFT and ML both fail, the stored value is in different units under the same field. *Fix:* multiply by the number of atoms in `_compute` (or have every calculator return per-atom values and convert in one place). Add a test that runs every calculator on the same structure and compares magnitudes.

2. **`RetryMixin.__init__` doesn't call `super().__init__()`, so DFT never runs.** The MRO of `DFTCalculator` is `DFTCalculator → RetryMixin → TimingMixin → Calculator`. `RetryMixin.__init__` sets `retries` and stops, so `TimingMixin.__init__` and `Calculator.__init__` never run: there's no `_cache` and no `timings`. Every `dft.energy(...)` raises `AttributeError: 'DFTCalculator' object has no attribute 'timings'` (from `TimingMixin`'s `finally`). The `except Exception` in `compute_energy` (#4) logs it as a warning and falls back to ML, so every structure is labelled by the ML potential and nothing fails. The result's `source` says `"ml"`, so it's visible if someone looks, but the point of the PR, DFT labels, silently isn't happening. DFT only runs on the cluster, which is probably why nobody noticed. *Fix:* `super().__init__(**kwargs)` in `RetryMixin.__init__`. Every `__init__` in a cooperative mixin chain must call it.

3. **The base cache is keyed on `structure.id` only, so `update_model` doesn't change any energy that was already computed.** `Calculator.energy` caches in `self._cache[structure.id]`. `MLPotential.update_model` swaps the model and bumps `self.version`, but nothing reads `version`. After retraining, every structure seen before returns the old model's energy. The active-learning loop then evaluates and selects with stale predictions, and "the new model didn't change anything" looks like a modelling result. The same key also returns the old energy if a structure is relaxed in place under the same `id`. *Fix:* key the base cache on `(structure.id, self.version)`, and have subclasses bump `version` whenever their results change (or clear the cache on update). If structures can change under the same id, add a hash of the geometry to the key.

## Major
4. **`except Exception` turns bugs into silent fallbacks.** The chain is meant to fall back when a calculation fails (SCF not converging, lost node), but it also catches `AttributeError`, `TypeError` and `KeyError`, which are bugs in our own code. That's exactly what hid #2. *Fix:* a `CalculationError` base class for expected failures (with `TransientError` as a subclass), raised by the calculators and caught by `compute_energy`. Everything else propagates. Agree with the owner of `run_dft` which of its exceptions mean "DFT failed on this structure", and translate those.

5. **Non-finite energies are cached and returned.** ML potentials can return NaN or huge values for structures far from their training data (overlapping atoms, unusual metals). `float(np.sum(...))` passes NaN through, the base class caches it, and `compute_energy` returns it with `source="ml"` instead of falling back to the surrogate. *Fix:* in `Calculator.energy`, raise `CalculationError` for a non-finite result, before caching it.

6. **The surrogate gives unseen elements a coefficient of zero.** With `self.coef.get(el, 0.0)`, a framework containing an element the surrogate never saw (Pu, say) gets an energy computed as if that element weren't there. And the structures that reach the last resort are the unusual ones that DFT and ML already failed on, so this is more likely than it looks. *Fix:* raise `CalculationError` naming the unseen elements.

## Minor
7. **The orchestrator dispatches on `isinstance(calc, DFTCalculator)`.** The DFT atom limit lives in `compute_energy`, not in the calculator. A new DFT-like calculator that doesn't subclass `DFTCalculator` (a hybrid-functional wrapper, another code) doesn't get the guard, and a DFT subclass can't change its limit. *Fix:* each calculator answers `supports(structure)` (by default everything; for DFT, up to `max_atoms`), and `compute_energy` asks it.

8. **When everything fails, the error doesn't say why.** `RuntimeError(f"all calculators failed for {structure.id}")` drops the individual errors, which only exist in the warning logs. *Fix:* collect them and include them in the message (or raise an `ExceptionGroup`).

9. **`timings` grows without bound.** A long-running service appends one float per call, forever. Cache hits are timed too (microseconds), which drags down the dashboard's average DFT time. *Fix:* a bounded `collections.deque(maxlen=...)`, and record hits separately if the dashboard needs real compute times.

## Nit
10. `retries=2` means three attempts, with no pause between them, and a lost node is often still lost a second later. Document that it counts extra attempts (or call it `max_attempts`), and consider a backoff.

## Red herrings (say why they're fine)
- **`RetryMixin.energy` looks as if it can fall off the end of the loop and return `None`.** It can't. On the last attempt, `if attempt == self.retries: raise` re-raises the `TransientError`, and every other path returns. (Compare the unbound variable after retries in the simulation-runner mock.)
- **`super().energy()` in `TimingMixin` and `RetryMixin`, which have no base class.** `super()` follows the MRO of the *instance's* class, not of the class the method is written in, so in `MLPotential` it reaches `Calculator.energy`. That's how cooperative mixins work.
- **`isinstance(calc, EnergyModel)` with a `@runtime_checkable` Protocol.** It's fine as a smoke check, as long as nobody mistakes it for contract checking: it only checks that `name` and `energy` exist. I checked: it passes for the per-atom surrogate, and for an object whose `energy` takes three arguments. Running it for every structure inside the loop is wasted work; check once. (`issubclass` against this protocol would raise `TypeError`, because `name` is a data member.)

## Model 1-minute summary
"There are three silent blockers. The surrogate returns energy per atom while the others return per cell, so any fallback to it stores a number that's off by the number of atoms. `RetryMixin` doesn't call `super().__init__`, so DFT's cache and timings are never created, every DFT call raises `AttributeError`, and the catch-all `except` quietly falls back to ML: we'd be labelling everything with the ML potential. And the base cache is keyed on structure id only, so after `update_model` we keep serving the old model's energies. I'd fix the per-atom conversion and the `super` call, and put the model version in the cache key. Then narrow the `except` to a `CalculationError` so bugs surface, reject NaN energies and unseen elements instead of returning them, and move the DFT size check onto the calculator. Tests: run every calculator on one structure and compare units, and check that the chain uses DFT when DFT succeeds."

## Reference fix
`L3_energy_fallback_fixed.py`:
- `RetryMixin.__init__` calls `super().__init__(**kwargs)`.
- `CompositionSurrogate` returns energy per cell and raises `CalculationError` for unseen elements.
- `Calculator.energy` keys its cache on `(structure.id, self.version)`, which `update_model` bumps, and raises `CalculationError` for non-finite results.
- `compute_energy` checks the protocol once, asks each calculator `supports(structure)`, catches only `CalculationError` (`TransientError` is a subclass), and lists every failure in its final error.
- `timings` is a bounded deque, and the `RetryMixin` docstring says what `retries` counts.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_06_strategy_template.py -k L3 -v`
