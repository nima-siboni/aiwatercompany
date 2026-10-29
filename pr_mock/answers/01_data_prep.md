# Answer key: 01_data_prep.py

**Good clarifying questions:** Is `co2_uptake` ever 0 or negative? Can one `mof_id` appear
more than once (e.g. measured at different conditions, or duplicates from different sources)?
Will these features be used at inference time on new MOFs? What composition formats show up
(parentheses, fractional occupancies)?

## Blockers (silently wrong results)

1. **Normalisation is fitted before the split, which leaks test data** (`prepare` → `normalise`).
   The mean and std come from all rows, including val and test, so evaluation is optimistic.
   The stats also aren't returned, so new data at inference can't be normalised the same way.
   *Fix:* split first, fit mu/sigma on train only, apply to val and test, and persist them
   (or use a fitted sklearn `StandardScaler`).

2. **Mutable default `extra_columns=[]` gets mutated** (`load_dataset`). The first call appends
   `"co2_uptake"`. The second call in the same process (a notebook, a CV loop, a test suite)
   appends it again, so `df` ends up with a duplicated `co2_uptake` column. `df["co2_uptake"]`
   then returns a DataFrame, `y` becomes shape (N, 2), and downstream broadcasting can hide it.
   *Fix:* `extra_columns=None` → `cols = [...] + list(extra_columns or [])`, and don't mutate
   the caller's list.

3. **`parse_composition` overwrites repeated elements** (`out[el] = ...`). Formulas such as
   `"CH3COOH"` or linker-style `"C8H4O4Zn2O"` count only the last occurrence of an element.
   It also can't handle parentheses (`"Zn4O(BDC)3"`, `"Cu(OH)2"`) or fractional occupancy
   (`"Fe0.5"`). Unknown characters get turned into nonsense "elements".
   *Fix:* `out[el] = out.get(el, 0) + n`, or better, use `pymatgen.Composition`. Add tests
   with repeated elements and brackets.

4. **Random row split ignores groups** (`split`). `drop_duplicates()` only drops rows that are
   identical in every column, so the same `mof_id` with slightly different values, or
   near-identical polymorphs, lands in both train and test. That inflates the test score.
   *Fix:* split by `mof_id` (`GroupShuffleSplit`) and possibly by structure/scaffold cluster,
   depending on what generalisation the model is meant to show.

5. **`np.log(y)` with zero or negative uptakes** gives `-inf`/`NaN` targets that train
   silently, and many screened MOFs have ~0 uptake. *Fix:* validate inputs, use `log1p`, or
   filter with an explicit, logged reason. Also document that the target is in log space,
   so anyone consuming predictions knows to invert it.

## Major

6. **Divide by zero in `normalise`** when a feature is constant (e.g. a filtered subset
   where every density is the same, or a single-row dataset). That produces NaN columns.
   *Fix:* `sigma[sigma == 0] = 1` or add an epsilon.

7. **The import-time side effect uses a hard-coded absolute path** (`ELEMENT_PROPS`,
   `/home/alex/...`). Importing the module fails on every other machine and in CI.
   *Fix:* load lazily, take the path from config, or use a packaged resource.

8. **`KeyError` on an unknown element** in `ELEMENT_PROPS.loc[el, ...]`, and nothing tells
   you which MOF caused it. *Fix:* validate and report the offending `mof_id`.

9. **No seed for `random.shuffle`**, so the splits differ on every run and results can't be
   compared or reproduced. *Fix:* take a `seed` argument and use
   `np.random.default_rng(seed).permutation`.

## Minor

10. **Performance:** `iterrows` plus two `.loc` lookups per element per row is slow for
    100k+ MOFs. Parse each composition once, cache results for repeated compositions, and
    vectorise the property lookups (e.g. a dict built from `ELEMENT_PROPS`).
11. **Small datasets:** `int(len * frac)` can be 0, which gives an empty val or test set
    with no warning.
12. **API shape:** `split` takes `df` only for its length; the nested tuples are hard to
    read. Prefer returning a small dataclass or dict. Add type hints and a docstring for
    the feature order.

## Nit
- The featurisation is a pretty weak descriptor (an atom-fraction-weighted mean of two
  properties). Ask whether that's intentional for a baseline. Raise it as a question, not a blocker.

## Red herring
- `df.dropna()` without `reset_index()` **looks** like an index-alignment bug, but everything
  afterwards uses positional NumPy arrays (`np.array(feats)`, `.values`), so it's fine here.
  Say so, and suggest `reset_index(drop=True)` as cheap defensive hygiene.

## Model 1-minute summary
"Three things must change before merging: the scaler is fitted on the whole dataset, so
there's test leakage; the mutable default breaks on the second call; and the composition parser
miscounts repeated elements. Close behind: a group split by `mof_id` and handling zero uptake
before taking the log. I'd want unit tests for the parser and an assertion that no `mof_id`
appears in more than one split."
