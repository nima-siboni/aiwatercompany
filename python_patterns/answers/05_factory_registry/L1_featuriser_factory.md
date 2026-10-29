# Answer key: L1_featuriser_factory.py — Factory and registry, level 1 (10 min)

**Pattern focus:** A factory should map a name to a *class*, build only what was asked for, and leave the caller's config alone. This drill trains you to spot eager construction, mutation of the caller's config, and silent defaults in config-driven construction.

**Good clarifying questions:** Is the config dict reused after this call (CV folds, sweeps, saved with the run)? How big is the Zeo++ file, and does every machine that trains have it (laptops, CI)? Its default path is relative to the working directory, so where is train.py run from? Will `"zeopp"` be used on newly generated structures? They have no precomputed descriptors, so `featurise` raises a bare `KeyError` for them.

## Blockers (silently wrong results / data loss)
1. **`config.pop` empties the caller's config, so later folds silently get the default featuriser.** `make_featuriser` does `config.pop("kind", "composition")` and `config.pop("params", {})` on the dict that train.py passes in (`cfg["featuriser"]`). With `{"kind": "rdf"}`, fold 1 gets an `RDFFeaturiser`. By fold 2 the dict is `{}`, so `kind` falls back to `"composition"` and every remaining fold trains on composition features. Nothing errors, and the CV score is an average over two different models. If the config is saved with the run afterwards, it records `"featuriser": {}`, so the run can't be reproduced either. *Fix:* read, don't pop: `kind = config["kind"]`, `params = config.get("params", {})`. If you really need to strip keys, copy first.

## Major
2. **Every featuriser is built with the same `params`, so any featuriser-specific parameter crashes.** The dict literal calls `CompositionFeaturiser(**params)`, `RDFFeaturiser(**params)` and `ZeoppFeaturiser(**params)` before looking anything up. The PR's own example, `{"kind": "rdf", "params": {"n_bins": 64}}`, raises `TypeError: CompositionFeaturiser.__init__() got an unexpected keyword argument 'n_bins'`. The factory only works with empty params, which is presumably the only case that was tried. *Fix:* map names to classes and instantiate only the chosen one: `FEATURISERS = {"rdf": RDFFeaturiser, ...}`, then `FEATURISERS[kind](**params)`.

3. **Every call loads the 1.5 GB Zeo++ file, whatever was asked for.** It's the same root cause: building the dict runs `ZeoppFeaturiser.__init__`, which reads the whole `.npz`. With 5-fold CV that's five full loads, just to get a composition featuriser. On a machine without `data/zeopp_descriptors.npz` (a laptop, CI), even `{"kind": "composition"}` fails with `FileNotFoundError`. *Fix:* the class map from #2; only the requested class is constructed.

4. **Missing or misspelled keys silently fall back to defaults.** `{"type": "rdf"}` gives a `CompositionFeaturiser`, and `{"kind": "rdf", "param": {"n_bins": 64}}` gives an RDF with the default 32 bins. Both run without complaint. *Fix:* make `kind` required and reject unknown top-level keys with a `ValueError` that names them.

## Minor
5. **The unknown-name error doesn't help.** `{"kind": "soap"}` raises `KeyError: 'soap'` from the dict lookup, with no list of valid names. *Fix:* `raise ValueError(f"unknown featuriser {kind!r}; expected one of {sorted(FEATURISERS)}")`.

## Red herrings (say why they're fine)
- **`self.rows` is used after the `with np.load(...)` block has closed the file.** Loading an `.npz` is lazy, but `data["mof_id"]` and `data["descriptors"]` each read a whole array into memory inside the block. The dict holds rows of those in-memory arrays, so nothing touches the file after it's closed.

## Model 1-minute summary
"The one that worries me is `config.pop`. It empties the dict that train.py passes in, so from the second CV fold on we silently train on the default composition featuriser, and the saved config no longer says what we ran. Next, the dict literal builds every featuriser on every call. Any featuriser-specific parameter crashes, including the PR's own `n_bins` example, and every call loads the 1.5 GB Zeo++ file, which also fails outright on machines that don't have it. I'd map names to classes and build only the one requested, from a config we read but don't modify. I'd also make `kind` required and reject unknown keys, so a typo fails instead of quietly giving us composition features, and list the valid names in the error. Tests: call the factory twice with the same dict, and build `rdf` with custom params."

## Reference fix
`L1_featuriser_factory_fixed.py`:
- `FEATURISERS` maps names to classes, and `make_featuriser` instantiates only the requested one.
- It reads `config["kind"]` and `config.get("params", {})` instead of popping, so the caller's dict is unchanged.
- Unknown keys, a missing `kind` and unknown names raise a `ValueError` that lists the options.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_05_factory_registry.py -k L1 -v`
