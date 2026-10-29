# Answer key: L3_campaign_config.py — Dataclasses, level 3 (35 min + 5 min summary)

**Pattern focus:** nested config dataclasses with inheritance and a JSON round trip. Check
where values enter (JSON, environment) and whether they reach the declared types and the
`__post_init__` checks, whether "frozen" objects are really immutable, and whether
`to_dict`, `from_dict`, equality and the run hash agree on what a config is.

**Good clarifying questions:** Does one process build several configs (a launcher, a
notebook, the test suite)? That decides whether #2 and #3 leak between campaigns. Which
fields identify a run: is `name` part of it, and should `run_id` survive adding a new field?
Do we reload saved configs to resume? Does anything else on the cluster set `AL_*`
variables? Where does UCB's uncertainty come from: the ensemble spread?

## Blockers (silently wrong settings / wrong labels)

1. **Environment overrides are never converted to the field's type.** `_set_path` stores the
   raw string. `AL_USE_CACHED_LABELS=false` gives `use_cached_labels == "false"`, which is
   truthy, so the campaign keeps reusing cached uptakes in exactly the situation where
   someone turned caching off to force fresh simulations (after a simulator fix, say).
   Numbers stay strings too: `AL_BUDGET=200` makes `budget == "200"`, which either crashes at
   the first comparison with an int or slips through string formatting into a command line,
   and it changes `run_id`. `from_dict` doesn't convert JSON values either. *Fix:* build the
   config through a `from_dict` that converts every value to its declared type, with explicit
   boolean parsing (`"false"`, `"0"`, `"no"`).

2. **`default_factory=lambda: DEFAULT_FORCEFIELD` hands every config the same dict.** The
   factory returns the module constant itself, not a copy, so every `SimulationConfig` shares
   one dict, and `frozen=True` doesn't protect its contents.
   `AL_SIMULATION__FORCEFIELD__CUTOFF=14` (or a notebook doing
   `cfg.simulation.forcefield["cutoff"] = 14`) changes `DEFAULT_FORCEFIELD` and the force
   field of every config in the process that uses the default, including ones created earlier. A launcher that
   loads several campaigns runs all of them with the wrong cutoff. *Fix:*
   `default_factory=lambda: dict(DEFAULT_FORCEFIELD)`, and stop mutating configs (#3). A
   small frozen `ForceFieldConfig` dataclass would remove the problem for good.

3. **`object.__setattr__` outside `__post_init__` defeats `frozen=True`.** It causes two
   failures. First, nothing is validated: `__post_init__` doesn't run again, so
   `AL_SIMULATION__PRESSURE=101325` (pascals, not bar) is accepted. Second, it mutates shared
   objects. When the JSON has no `simulation` section, `cfg.simulation` is the class-level
   default `GCMCConfig()` instance that every `CampaignConfig` shares, so
   `AL_SIMULATION__TEMPERATURE=273` changes the temperature of every other config in the
   process that uses the default simulation section (to the string `"273"`). *Fix:* treat configs as immutable. Merge overrides into
   the raw dict and build the config once, so conversion and validation run. To change a
   built config, use `dataclasses.replace`, which re-runs `__post_init__`.

4. **`GCMCConfig.__post_init__` doesn't call `super().__post_init__()`.** The generated
   `__init__` calls only the most-derived `__post_init__`, so the temperature and pressure
   checks never run for GCMC configs, which are the only simulation configs in use.
   `GCMCConfig(pressure=101325.0)` is accepted, the simulations run at 101325 bar where every
   MOF is saturated, and the surrogate trains on the wrong quantity. *Fix:* call
   `super().__post_init__()` first.

## Major

5. **`from_dict` doesn't rebuild nested configs.** `cls(**d)` leaves `simulation` and
   `surrogate` as plain dicts whenever the JSON has those sections, and `save()` always writes
   them. So a saved config never loads back properly: `loaded == cfg` is False, and
   `summary()` raises `AttributeError: 'dict' object has no attribute 'temperature'`. A partial
   section such as `"simulation": {"temperature": 273}` also gets no defaults and no
   validation. It passes a unit test that only uses top-level keys. *Fix:* build recursively
   from `fields()` and the field types, and add the round-trip test
   `from_dict(cfg.to_dict()) == cfg`.

6. **Misspelt override keys are silently accepted.** `object.__setattr__` happily creates a
   new attribute, so `AL_BUGDET=50` sets `cfg.bugdet` and the budget stays at 500, which is ten
   times the intended GCMC spend, with no error. Any unrelated `AL_*` variable gets attached
   too. *Fix:* reject keys that aren't fields (the rebuilt `from_dict` does this), and log
   which overrides were applied.

7. **`run_id` disagrees with equality.** `notes` is marked `compare=False`, so it isn't meant
   to be part of identity, but `asdict` ignores `compare` and the hash includes it. Editing
   the notes, toggling `use_cached_labels` or moving `output_dir` all give a new `run_id`, so
   `find_previous_run` misses the earlier run and the campaign starts again from scratch.
   *Fix:* hash only `compare=True` fields, and mark the run options (`use_cached_labels`,
   `output_dir`, `notes`) `compare=False`, or better, move them out of the settings object.
   Related: `temperature=300` and `300.0` compare equal but serialise differently; converting
   types on load (#1) covers the JSON and environment paths.

## Minor

8. **`run_id` is recomputed from mutable state on every access.** Each access runs `asdict`
   (a deep copy), `json.dumps` and SHA-256. That costs only about 6 µs for this config, so
   speed isn't the point. The point is that the id is recomputed from state that can change
   (#2, #3): `run_dir` can move mid-run when another campaign's override edits the shared
   force-field dict. Once configs are really immutable, compute it once. `cached_property`
   works on a frozen dataclass, because it writes straight to the instance `__dict__`.

9. **`name` goes into a path unchecked.** `run_dir` is
   `os.path.join(output_dir, f"{name}-{run_id}")`, so `name="../../shared/co2"` writes outside
   `output_dir`, and a `/` in the name creates nested directories. *Fix:* validate `name` in
   `__post_init__` (e.g. `[A-Za-z0-9][\w.-]*`).

10. **No cross-field check between acquisition and ensemble size.** `SurrogateConfig` allows
    `ensemble_size=1`, but UCB and EI get their uncertainty from the ensemble spread. With
    one member the spread is zero, so UCB silently becomes greedy (and EI nearly so).
    *Fix:* in `CampaignConfig.__post_init__`, require `ensemble_size >= 2` unless
    `acquisition == "greedy"`, after confirming where the uncertainty comes from.

## Nit
- `run_id` changes whenever a field is added, even one whose default reproduces the old
  behaviour, so earlier runs stop being found after any schema change. Decide whether that's
  wanted; a schema version in the saved config makes it explicit.

## Red herrings (say why they're fine)
- **`simulation: GCMCConfig = GCMCConfig()`** shares one default instance across all
  configs. For a frozen object that's fine and idiomatic. Python accepts it because a frozen
  dataclass's class defines `__hash__` (since 3.11, a default whose class is unhashable, such
  as a dict or a non-frozen dataclass, is rejected with `ValueError`). The check only looks at
  the class: actually hashing this instance would raise `TypeError` because of the
  `forcefield` dict. It only goes wrong because #3 mutates it.
- **`hexdigest()[:12]`** keeps 48 bits. The collision probability for n runs is about
  n²/2⁴⁹, roughly 2 × 10⁻⁷ for 10,000 runs.
- **`environ=os.environ`** is evaluated once at import, but it's the live mapping that
  `os.environ[...] = ...` and `monkeypatch.setenv` modify in place. The parameter also makes
  `load_config` testable without touching the real environment.

## Model 1-minute summary
"Four things can silently give a campaign the wrong settings. Environment overrides are
stored as raw strings, so `AL_USE_CACHED_LABELS=false` is truthy and numbers become strings.
The force-field default factory returns the module-level dict itself, so one override changes
every config in the process. Overrides go through `object.__setattr__`, which skips
validation and mutates the shared `GCMCConfig()` default. And `GCMCConfig.__post_init__`
doesn't call super, so a pressure in pascals sails through. Most of these have one fix: apply
overrides to the raw dict and build the config once, through a `from_dict` that recurses into
nested sections, converts types and rejects unknown keys. That also fixes the broken save/load
round trip and silently ignored typos like `AL_BUGDET`. Then `run_id` should hash only the
`compare=True` settings. The shared frozen default and the 12-character hash are fine."

## Reference fix
`L3_campaign_config_fixed.py`:
- `load_config` merges overrides into the raw JSON dict (parsing each value as JSON where it
  can) and builds the config once.
- `from_dict` uses `_build`, which recurses into nested dataclasses, converts values to the
  declared types (with explicit boolean parsing), merges a partial `forcefield` section over
  the defaults, and rejects unknown keys.
- The `forcefield` factory copies `DEFAULT_FORCEFIELD`, and `GCMCConfig.__post_init__` calls
  `super()`.
- Run options are `compare=False`; `run_id` hashes only `compare=True` fields and is a
  `cached_property`.
- `CampaignConfig.__post_init__` validates `name` and requires an ensemble of at least 2 for
  UCB and EI.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_07_dataclasses.py -k L3 -v`
