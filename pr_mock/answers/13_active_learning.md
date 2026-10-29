# Answer key: 13_active_learning.py

**Good clarifying questions:** Were the seed results computed under the same conditions and
force field (0.15 bar, 298 K)? How noisy is one GCMC result? How big is the library, and does
it contain near-duplicates? Can the cluster run all 16 simulations at once, and what should
happen when one fails? Is `benchmark()` purely retrospective, with every label known? What is
the unit of the minimisation objective?

## Blockers

1. **The seed uptakes are multiplied instead of divided** (`load_seed`). One mmol of gas at STP
   occupies 22.414 cm³, so cm³(STP)/g → mmol/g divides by 22.414. The seeds come out about
   500× too large: a mean of 922 "mmol/g" in the check, against 1.84 really. The GP is then
   fitted on old values 500× the new ones, `best` is the largest seed, and no new simulation can
   ever beat it, so the curve stays flat at the seed maximum. *Fix:* divide, and name the
   constant `CM3_STP_PER_MMOL`. Also confirm which STP the seed data used: 22.414 L/mol at
   0 °C and 1 atm, or 22.711 at 1 bar.

2. **The GP is fitted on raw descriptors with one length scale** (`fit_gp`). VSA spans about
   2,800 m²/cm³, LCD about 20 Å and the void fraction 0.6, so distances are almost entirely VSA
   differences. In a synthetic library where uptake depends only on pore size, the PR's GP
   predictions correlated 0.45 with the truth and its 16 picks averaged 1.9 mmol/g. With
   standardised inputs, one length scale per feature and `normalize_y=True`, the correlation
   was 1.00 and the picks averaged 3.7, out of a best possible 3.8. `normalize_y` also defaults
   to False, so far from the data the GP falls back to a prior mean of 0. *Fix:*
   `StandardScaler` on the features (fitting it on the whole library is fine, since no labels
   are involved), `RBF(np.ones(n_features))`, `normalize_y=True`, and a `WhiteKernel` or an
   `alpha` that matches the GCMC noise.

3. **Picks are removed from the pool by label, but they are positions** (`run_campaign`).
   `propose` returns positions. `iloc` uses them correctly, but `drop` removes by label. Round 1
   works because `reset_index` made labels equal positions. From round 2 on, the index has gaps,
   so `drop` does one of two things:
   - It silently removes other MOFs. In the check, the picked hMOF-36, 57 and 93 stayed in the
     pool, while 20, 41 and 77 were removed. The simulated MOFs can then be picked again.
   - It raises KeyError when a position matches a label that's already gone. In the full run
     this happened in round 2, after that round's simulations had finished. Nothing had been
     saved (see 8), so every result since the start was lost.
   *Fix:* work with stable IDs, e.g. `pool = pool[~pool.mof_id.isin(batch.mof_id)]`, and
   `assert not set(pool.mof_id) & set(labelled.mof_id)`.

4. **Minimisation is flipped twice.** `run_campaign` negates y so that the GP always maximises,
   then calls EI with `maximize=False`, which flips it again. In the check, the "minimise"
   picks had a mean true uptake of 5.24, against a pool mean of 1.81 and a maximum of 5.55. It
   chased the maximum. `curve` also records `.max()` for both objectives. *Fix:* pick one
   convention: negate and always maximise, or keep y and pass the flag. Test it on a 1-D toy
   with a known minimum.

5. **The paper figure compares unlike things** (`benchmark`). The BO curve is
   `labelled.max()`, which includes the 300 seeds, so it starts at the best seed (5.54 in the
   check). The random baseline only counts its own picks: 4.17 after round 1 and 4.99 after
   round 10. BO "wins" from round 0 even if it picks nothing useful. *Fix:* give both arms the
   same starting set and budget, plot the improvement over the seed best, and repeat over
   several seeds with the spread shown.

## Major

6. **The top 16 by EI from one posterior cluster together.** Their mean pairwise distance in
   standardised descriptor space was 1.32, against 2.72 for a random batch: 16 near-identical
   MOFs, mostly redundant simulations. *Fix:* a batch-aware acquisition (q-EI, kriging
   believer, local penalisation) or a diversity constraint.
7. **The GCMC runs are sequential.** `gcmc_oracle` loops over 16 runs of 2–6 hours each, so a
   round can take 4 days instead of about 6 hours. The whole point of a batch is to run it in
   parallel. *Fix:* submit all 16 (a job array, or a pool of futures), wait for them, and handle
   each failure separately.
8. **Nothing is saved until the end.** An exception from `run_gcmc`, or the KeyError in 3,
   loses every result so far, i.e. days of cluster time. *Fix:* write `labelled` to disk after
   each round (atomically), resume from it, and record failed runs explicitly. Don't record
   them as 0.0 uptake, which the GP would learn as a bad MOF.

## Minor

9. **The random baseline draws with replacement** and can redraw across rounds: 5 repeats in
   160 draws here. It's small, but it handicaps the baseline. Use `replace=False`, and remove
   the drawn MOFs.
10. **`load_seed(...).merge(pool, on="mof_id")` is an inner join**, so seeds outside the
    filtered library vanish silently. Log how many.
11. **`xi=0.01` is in the target's units.** It means 0.01 mmol/g here, and something different
    for parasitic energy. Scale it with the target's spread.
12. **Conditions are hard-coded** in the oracle and the docstring, and nothing checks that the
    seeds match them.
13. **No `random_state` for the GP restarts, and the fitted kernel isn't logged.** Logging
    `gp.kernel_` would have exposed bug 2.

## Red herrings (say why they're fine)

- **`pld > 3.3`:** 3.3 Å is the kinetic diameter of CO2, so it's a sensible filter.
- **The EI formula for maximisation is correct.** The bug is the double flip around it.
- **`np.maximum(sd, 1e-9)`** guards the division, as it should.
- **`np.argsort(-ei)[:BATCH]` and `pool.iloc[picks].copy()`:** positions are what `iloc` wants,
  and the `.copy()` makes the column assignment safe. Only the `drop` is wrong.
- **Removing the seeds from the pool with `isin`** is correct.

## Model 1-minute summary
"Three things make the loop pick the wrong MOFs. The seed uptakes are multiplied by 22.414
instead of divided, so the old data is 500 times too large and nothing new can ever beat it.
The GP sees raw descriptors with one length scale, so VSA swamps pore size: on a synthetic test
its picks were half as good as with scaled inputs. The minimisation path is flipped twice and
chases the maximum. The bookkeeping is broken too: picks are dropped by label though they are
positions, so from round 2 it either removes the wrong MOFs or crashes, and nothing is saved, so
a crash loses days of GCMC. The paper figure is biased, because BO counts the seeds and random
selection doesn't. Then the batch clusters, and the 16 runs are sequential. I'd ask for a 1-D
toy test with a known optimum, a test that the pool and the labelled set stay disjoint, and a
benchmark with matched starting sets."
