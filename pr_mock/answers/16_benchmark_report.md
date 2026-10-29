# Answer key: 16_benchmark_report.py

**Good clarifying questions:** What will screening feed the model, relaxed or unrelaxed
structures? Which kind of generalisation matters: new compositions, new chemical systems or new
prototypes? How were the E_hull labels computed (with which corrections)? How is the threshold
used downstream? What fraction of the test set is stable?

## Blockers

1. **The ensemble is one model, five times** (`train_ensemble`). `torch.manual_seed(0)` before
   every member gives the same initialisation and the same batch order, so in the check the
   members' predictions were identical. The ensemble's uncertainty is then only each member's
   own σ: the disagreement term, which is why you build an ensemble, is zero. σ is therefore
   overconfident exactly on unfamiliar inputs. *Fix:* seed member k with `base + k`, which is
   still reproducible.

2. **Compositions leak despite GroupKFold** (`split`). The groups are raw formula strings, so
   "Fe2O3", "Fe4O6" and "O3Fe2", one composition from different sources, are three groups. In
   the check they landed in folds 2, 1 and 2, and LiCoO2/CoLiO2 and NaCl/ClNa were split too.
   Test compositions are seen in training, so the MAE and F1 are optimistic. *Fix:* group by
   `Composition(f).reduced_formula`, or by chemical system or prototype, depending on the
   generalisation you need.

3. **The model is given DFT-relaxed structures** (`featurize(df["relaxed_structure"])`). In
   screening, the relaxed structure is the output of the expensive calculation we want to
   skip, so candidates arrive unrelaxed. Matbench Discovery calls relaxed inputs circular and
   requires unrelaxed ones. This benchmark measures an easier task than production. *Fix:*
   featurise the initial structures, or structures relaxed with our MLIP: whatever screening
   will actually see.

4. **Coverage uses a one-sided quantile** (`coverage`). `norm.ppf(p)` is the one-sided
   quantile; a central interval at level p needs `norm.ppf(0.5 + p / 2)`. A *perfectly*
   calibrated model scored 0.0 at 0.5, 0.60 at 0.8, 0.80 at 0.9 and 0.90 at 0.95. It looks
   overconfident, so someone will "fix" it by inflating σ. *Fix:* `norm.ppf(0.5 + p / 2)`, and
   a unit test on synthetic calibrated data.

5. **The bootstrap interval has zero width** (`bootstrap_ci`). The generator is recreated with
   the same seed inside the loop, so every resample is identical. On top of that,
   `replace=False` makes each resample a permutation of the whole test set. The MAE is the same
   1,000 times, and the interval came out [0.0395, 0.0395]. *Fix:* one generator outside the
   loop, and draw with replacement (`rng.integers(0, n, n)`).

## Major

6. **Member σs are averaged instead of combined** (`predict`). The ensemble's predictive
   variance is mean(σ_m²) + var(μ_m), and averaging σ_m ignores the members' disagreement. Five
   members with σ = 0.1 and means from −0.2 to 0.4: the PR reports 0.10; the mixture gives
   0.24. *Fix:* `sigma = sqrt(mean(σ²) + mean(μ²) − mean(μ)²)`.
7. **`r2_score(mu, y_test)` has its arguments swapped.** R² isn't symmetric: on synthetic data
   the right order gave 0.612 and the swapped order 0.716; for a shrunk prediction, 0.576 against
   −0.929. *Fix:* `r2_score(y_test, mu)`.
8. **The F1 threshold is picked on the test set.** Choosing the best of 61 thresholds on the
   data you report on is optimistic. With 300 test points, the reported F1 averaged 0.034 above
   the F1 on new data (higher in 80% of repeats); with 2,000 points, 0.006 above. *Fix:* choose
   the threshold on training folds or a validation split, then report test F1 at that fixed
   threshold.
9. **"Do we beat production?" from two separate intervals.** Once the bootstrap works,
   checking whether two intervals overlap ignores that both models are scored on the same test
   points. In a synthetic check the separate intervals overlapped ([0.0484, 0.0522] and
   [0.0505, 0.0544]), yet the paired interval of the difference was [0.0015, 0.0029], a clear
   improvement. *Fix:* bootstrap the difference on the same resampled indices.

## Minor

10. **One fold and one ensemble run** (`split(df)[0]`), so there's no estimate of variability
    across folds or seeds.
11. **F1 alone for a rare positive class.** Also report precision, recall and the discovery
    acceleration factor (precision divided by the stable fraction), as Matbench Discovery does.
12. **The threshold grid (−0.1 to 0.2 eV/atom) is hard-coded.**

## Red herrings (say why they're fine)

- **`y <= 0` for "stable"** is right. Compounds on the hull have E_hull exactly 0, so `< 0`
  would give no positives.
- **`@torch.no_grad()` in `predict`** is right: nothing here needs gradients.
- **GroupKFold** is the right tool; the problem is the grouping key.
- **The 2.5th and 97.5th percentiles** are the standard 95% percentile bootstrap.
- **MAE as the main regression metric** is fine.

## Model 1-minute summary
"Every number in this report is either optimistic or not what it claims. The ensemble members
all use seed 0, so it's one model five times, with no disagreement term in its uncertainty. The
grouping uses raw formula strings, so Fe2O3 written three ways lands in different folds and
compositions leak into the test set. The model gets DFT-relaxed structures, which screening
won't have. Coverage uses the one-sided quantile, so a perfectly calibrated model looks
overconfident, and the bootstrap reseeds inside the loop without replacement, giving an
interval of width zero. After those: σs averaged instead of combined, R² with swapped
arguments, a threshold tuned on the test set, and overlapping intervals used instead of a
paired comparison. I'd ask for a synthetic sanity suite: a calibrated toy for coverage, known
duplicates for the split, and a paired bootstrap."
