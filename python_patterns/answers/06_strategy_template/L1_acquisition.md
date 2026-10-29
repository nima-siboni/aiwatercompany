# Answer key: L1_acquisition.py — Strategy and template method, level 1 (10 min)

**Pattern focus:** Strategies are only interchangeable if they share one contract: the same signature, the same meaning for the return value (here, higher is better), and a base class that can't be left unimplemented. This drill trains you to check that contract, not each strategy on its own.

**Good clarifying questions:** Can `batch_size` reach 0 in the last round, or does the loop stop first? Can the surrogate return NaN (a failed featurisation, or a slightly negative variance under a square root)? Can scores be negative, for example with standardised targets? Dividing a negative UCB score by a larger cost makes it rank *higher*, so `CostAwareUCB` assumes positive scores.

## Blockers (silently wrong results / data loss)
1. **`TargetUptake` returns a distance, so `select` picks the candidates furthest from the target.** Every other strategy means "higher is better", and `select` takes the highest scores with `np.argsort(scores)[-batch_size:]`. But `np.abs(mean - self.target)` is lower-is-better. With `target=5.0`, predicted means `[1, 5, 9, 4]` and `batch_size=1`, it picks one of the two MOFs 4 mmol/g away from the target (1 or 9) and never the one predicted at 5. The campaign spends its GCMC budget on the worst candidates, and nothing errors. *Fix:* return `-np.abs(mean - self.target)`, and write the direction into the base class docstring.

## Major
2. **The base `score()` returns `NotImplemented` instead of failing.** `Acquisition` isn't abstract, so a subclass that misspells `score`, or a config that builds the base class itself, runs without error. `np.argsort(NotImplemented)` returns `array([0])` (I checked), so `select` returns `[0]` whatever the batch size: one candidate, always the first, every round. `NotImplemented` is meant for binary operators such as `__eq__`, not for "subclasses must override this". *Fix:* make `Acquisition` an `ABC` with `@abstractmethod` on `score`. Then the misspelt subclass fails at construction.

3. **`batch_size=0` selects every candidate.** The docstring says the loop passes `min(96, remaining budget)`. When the budget is exactly used up, `np.argsort(scores)[-0:]` is the same as `[0:]`: the whole pool. The last round submits every unlabelled MOF to GCMC. *Fix:* sort in descending order and slice from the front, `np.argsort(-scores)[:batch_size]`, which returns nothing for 0 and puts the best first.

## Minor
4. **`CostAwareUCB.score` needs an extra argument, so `select` has to type-check.** It overrides `score(mean, std)` with `score(mean, std, cost)`, so a `CostAwareUCB` can't be used where an `Acquisition` is expected, and `select` special-cases it with `isinstance`. Any other caller that does `acq.score(mean, std)` crashes. The next strategy that needs costs won't get them until someone edits `select`: it's called without `cost` and fails with `TypeError: unsupported operand type(s) for /: 'float' and 'NoneType'`. *Fix:* one signature for every strategy, `score(mean, std, cost=None)`. `select` passes everything it has, and each strategy ignores what it doesn't need.

5. **NaN scores are treated as the best.** `np.argsort` sorts NaN to the end, and `[-batch_size:]` takes from the end, so a candidate with a NaN prediction is selected first. *Fix:* replace non-finite scores with `-np.inf` before sorting, or reject them if NaN means a bug upstream.

## Nit
6. The returned indices are in ascending score order, so the best candidate in the batch comes last. If anything downstream submits them in order and stops early, it drops the best ones. Return them best first (the fix for #3 does this).

## Red herrings (say why they're fine)
- **UCB uses `mean + self.beta * std`, not `sqrt(beta) * std`.** GP-UCB papers write the bonus as √β·σ, but that's just another parametrisation of the same exploration weight. Here `beta` is the multiplier itself (often called κ), so `beta=2` means "two standard deviations". It's worth a docstring, not a change.

## Model 1-minute summary
"The blocker is the direction contract. Every strategy is supposed to return higher-is-better, but `TargetUptake` returns a distance, so `select` silently spends the budget on the candidates furthest from the target. I'd fix the sign and write the contract into the base class. Close behind: the base `score` returns `NotImplemented`, so a strategy with a misspelt method doesn't fail. `argsort` of `NotImplemented` is `[0]`, so we'd simulate the first candidate every round; making the base an ABC fixes that. Then `[-batch_size:]`: when the budget hits zero, `[-0:]` selects the entire pool. Sorting in descending order and slicing from the front fixes that and also puts the best first. Lastly, the `isinstance` special case for cost-aware UCB: give every strategy the same signature. Tests: target selection on a toy array, `batch_size=0`, and a NaN prediction."

## Reference fix
`L1_acquisition_fixed.py`:
- `Acquisition` is an ABC with an abstract `score(mean, std, cost=None)`. Every strategy has that signature, so `select` has no `isinstance` checks.
- `TargetUptake` returns `-abs(mean - target)`, and the base docstring says higher is better.
- `select` maps non-finite scores to `-inf`, sorts in descending order and slices `[:batch_size]`, so 0 selects nothing and the best come first.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_06_strategy_template.py -k L1 -v`
