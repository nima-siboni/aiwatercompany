# Strategy and template method: review checklist

Strategy swaps one algorithm for another behind a shared interface (acquisition functions, optimisers, calculators); template method fixes the skeleton of an algorithm in a base class and lets subclasses fill in the steps. Both only work if every implementation keeps the same contract, so the review is mostly about that contract and about who owns state.

## When you see it in a PR, check:
- **The base can't be left unimplemented.** `return NotImplemented` or a non-abstract base lets a misspelt override run silently; use `ABC` and `@abstractmethod`.
- **Every strategy has the same signature.** An override with an extra required argument forces `isinstance` checks on callers and crashes everyone else.
- **Every strategy means the same thing by its return value.** Direction (higher is better), units (eV vs kJ/mol), per atom vs per cell: a mismatch gives silently wrong results, not errors.
- **Overrides keep the base contract (LSP).** A subclass that redefines `converged()` or returns a different quantity breaks every caller written against the base.
- **The caller doesn't dispatch with `isinstance` on concrete strategies.** Put the decision (`supports()`, limits, costs) on the strategy.
- **Mutable state lives on the instance, and is reset per run if the object is reused.** `history = []` in the class body is shared by every instance.
- **Hooks aren't called from `__init__`.** They run before the subclass's `__init__` body, and only once instead of once per run.
- **The template owns the invariants:** copying inputs, resetting state, iteration limits, and teardown in a `finally`.
- **Hooks don't mutate their inputs.** `x += ...` on the caller's array overwrites the caller's data.
- **Every `__init__` in a mixin chain calls `super().__init__(**kwargs)`.** One that doesn't silently skips every class after it in the MRO.
- **A cache in the base class is keyed on everything the result depends on,** including state that subclasses change (model version, settings).
- **Fallback chains catch only expected failures.** `except Exception` turns bugs into silent fallbacks.
- **Results are validated (finite, in range) before they're cached or returned.**

## Usually fine (don't block on it)
- `super().method()` inside a mixin with no base class: it follows the instance's MRO.
- A retry loop whose last attempt re-raises: it can't fall through and return `None`.
- A `@runtime_checkable` Protocol as a smoke check, as long as nobody treats it as contract checking (it only checks that the names exist).
- Plain functions instead of strategy classes when there's no state or configuration.

## Idiomatic shape
```python
class Relaxer(ABC):
    def run(self, positions):                     # the template: invariants live here
        positions = np.array(positions, dtype=float)
        self.setup()
        try:
            for _ in range(self.max_steps):
                ...                               # calls self.step(...), the hook
        finally:
            self.teardown()

    @abstractmethod
    def step(self, positions, forces): ...        # the variation point
```
