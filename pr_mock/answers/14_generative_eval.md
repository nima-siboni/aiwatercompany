# Answer key: 14_generative_eval.py

**Good clarifying questions:** Which published numbers will these be compared with, and with
which definitions, tolerances and sample sizes? Were the reference energies computed with the
same method as the samples (MLIP or DFT, same functional, same corrections)? How many samples
and training structures are there, and how long may the evaluation take? Do the generators
ever emit empty strings or partly occupied sites?

## Blockers

1. **Validity ignores periodic images** (`structure_valid`). `pdist` on Cartesian coordinates
   only sees distances inside one copy of the cell. Two atoms at x = 0.01 and 0.99 of a 5 Å cell
   are 0.10 Å apart across the boundary, but `pdist` reports 4.90 Å, so the structure passes.
   Generated crystals have atoms near cell faces all the time. *Fix:* use the periodic
   `structure.distance_matrix` with the diagonal masked, as CDVAE does.

2. **Energies per atom are passed where `PDEntry` expects the total energy** (`e_above_hull`).
   `PDEntry(composition, energy)` takes the energy of the whole composition. A Li2O sample
   exactly as stable as the reference came out 1.233 eV/atom above the hull, instead of 0.000.
   Every stability number is wrong. *Fix:* `x["e_per_atom"] * composition.num_atoms`. Also make
   the samples and the references come from the same energy method (see 9).

3. **`ehull < 0.0` can never be true.** The samples are added to the phase diagram, so a sample
   on or below the old hull becomes part of the hull and gets exactly 0. A sample 0.15 eV/atom
   below the reference hull got 0.000 and wasn't counted, so the S.U.N. rate is always zero.
   *Fix:* build the diagram from the references only and call
   `get_e_above_hull(entry, allow_negative=True)`, which gave −0.15 for that sample. Then use a
   stated threshold: ≤ 0, or 0.1 eV/atom for "metastable".

4. **The linker metrics work on raw strings** (`linker_metrics`). Three problems:
   - **Empty strings count as valid.** `Chem.MolFromSmiles("")` returns a molecule with 0 atoms,
     not None.
   - **Uniqueness counts spellings, not molecules.** "OCC", "C(O)C" and "CCO" are the same
     molecule. In the check it reported 0.90 where the true value was 0.50 (4 distinct molecules
     among 8 valid ones).
   - **Novelty compares canonical SMILES with raw training strings,** so a training molecule
     written differently looks new: 0.80 reported, 0.25 true.
   *Fix:* reject None and 0-atom molecules, and canonicalise both sides with the same RDKit
   version (or compare InChIKeys).

## Major

5. **The denominators don't match the table they go into.** The crystal metrics divide
   everything by the number of samples. The linker metrics divide uniqueness by all samples and
   novelty by the canonical set. The MOSES table divides uniqueness by the valid molecules and
   novelty by the unique ones, so the PR's numbers aren't comparable with the baselines next to
   them. *Fix:* use the benchmark's definitions, and report the counts as well.
6. **FCD on 1,000 samples, compared with baselines computed at another size.** The Fréchet
   distance has an upward bias that shrinks with sample size. Two samples from the *same*
   512-dimensional distribution scored 81 at n = 1,000 and 8.1 at n = 10,000. *Fix:* compute it
   at the same n as the numbers you compare with, as large as you can afford.
7. **All-pairs StructureMatcher is far too slow.** One `fit` on 8-atom cells took 6.8 ms here.
   At that rate, novelty for 10,000 samples against 27,000 MP-20 training structures takes
   about 500 hours, and all-pairs uniqueness about 95 hours. *Fix:* bucket by reduced formula
   (and space group) and match only within buckets, or use `matcher.group_structures`.
8. **Default matcher tolerances.** `StructureMatcher()` uses ltol 0.2, stol 0.3 and angle_tol 5,
   while the crystal-generation papers use 0.3, 0.5 and 10. Uniqueness and novelty therefore
   aren't comparable with the papers. *Fix:* set the tolerances explicitly to the ones in the
   comparison, and state them.

## Minor

9. **MLIP energies are mixed with MP DFT references in one hull.** A systematic offset between
   the two moves every sample up or down relative to the hull. Relax and score the references
   with the same MLIP, or check the "stable" candidates with DFT.
10. **"Validity" here is only the distance check.** CDVAE's validity also includes charge
    balance, so this "validity" is weaker than the one in the papers. Say so, or add the check.
11. **Uniqueness depends on the sample order.** `StructureMatcher.fit` isn't transitive, so
    which sample is kept, and how many, depends on the order. Fix the order, or cluster.
12. **No empty-input guards.** `len(canonical) == 0` divides by zero, and `pdist` on a one-atom
    structure fails on `.min()` of an empty array.

## Red herrings (say why they're fine)

- **`Chem.MolToSmiles(m)` without `canonical=True`:** it's canonical by default ("OCC" becomes
  "CCO").
- **StructureMatcher matching a cell to its supercell** is intended (`primitive_cell=True`).
- **The 0.5 Å threshold** is the CDVAE convention.
- **A set of canonical SMILES for uniqueness** is right, once the strings really are canonical.

## Model 1-minute summary
"The headline numbers are inflated or impossible. Validity uses `pdist`, which ignores periodic
images, so atoms overlapping across a cell face pass. The S.U.N. rate is always zero, for two
separate reasons: energies per atom go into `PDEntry`, which expects totals, so a sample as
stable as the reference sits 1.2 eV/atom above the hull; and with the samples inside the
diagram, `< 0` can never be true. On the linker side, empty strings count as valid, uniqueness
counts spellings rather than molecules, and novelty compares canonical against raw training
SMILES: 0.90 and 0.80 where the true values are 0.50 and 0.25. Then comparability: different
denominators from the MOSES table, FCD at a different sample size, default matcher tolerances,
and an all-pairs matcher that would take weeks. I'd ask for unit tests on known duplicates and
known hull compounds, and a check against the published numbers of one baseline."
