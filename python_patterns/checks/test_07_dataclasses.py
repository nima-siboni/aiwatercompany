"""Checks for pattern 07: dataclasses and value objects."""
import dataclasses
import json

import numpy as np
from harness import bug, fine

L1 = "07_dataclasses/L1_gcmc_conditions"
L2 = "07_dataclasses/L2_frozen_structure"
L3 = "07_dataclasses/L3_campaign_config"


def rejects(fn, exc=ValueError) -> bool:
    """True if calling fn raises exc."""
    try:
        fn()
    except exc:
        return True
    return False


# --- L1: GCMC conditions ---------------------------------------------------------


@bug(L1)
def test_L1_schedule_by_priority(m):
    """Blocker 1: order=True sorts by mof_id (first field), not by priority."""
    c = m.Condition(298.0, 1.0)
    jobs = [m.JobRequest("ZIF-8", c, 0), m.JobRequest("HKUST-1", c, 10), m.JobRequest("MOF-5", c, 5)]
    assert [j.priority for j in m.schedule(jobs)] == [10, 5, 0]


@bug(L1)
def test_L1_sweep_aliasing(m):
    """Blocker 2: pressure_sweep returns one mutated object, so every job gets the last pressure."""
    base = m.Condition(298.0, 1.0)
    jobs = m.isotherm_jobs("ZIF-8", base, [0.1, 1.0, 10.0])
    assert [j.condition.pressure for j in jobs] == [0.1, 1.0, 10.0]
    assert base.pressure == 1.0


@bug(L1)
def test_L1_mutated_key_lost(m):
    """Major 3: unsafe_hash on a mutable Condition loses dict entries after mutation."""
    c = m.Condition(298.0, 1.0)
    uptake_cache = {c: 2.1}
    try:
        c.pressure = 5.0
    except AttributeError:  # FrozenInstanceError subclasses AttributeError
        pass
    assert uptake_cache.get(m.Condition(298.0, 1.0)) == 2.1


@bug(L1)
def test_L1_validation(m):
    """Major 4: no __post_init__ validation, so negative temperature or pressure is accepted."""
    assert rejects(lambda: m.Condition(-5.0, 1.0))
    assert rejects(lambda: m.Condition(298.0, -1.0))


@bug(L1)
def test_L1_float_key(m):
    """Minor 5: 77 K via from_celsius is 76.99999999999997 K, so it doesn't dedupe with 77 K."""
    a = m.Condition.from_celsius(-196.15, 1.0, "N2")
    b = m.Condition(77.0, 1.0, "N2")
    assert a == b and len({a, b}) == 1


@fine(L1)
def test_L1_default_factory_tags(m):
    """Red herring: field(default_factory=list) gives every JobRequest its own list."""
    c = m.Condition(298.0, 1.0)
    a, b = m.JobRequest("ZIF-8", c), m.JobRequest("MOF-5", c)
    a.tags.append("urgent")
    assert b.tags == []


# --- L2: frozen Structure with ndarray fields ----------------------------------------

COORDS = [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]]


def _structure(m, coords=COORDS, **kw):
    return m.Structure(np.eye(3) * 10.0, ("Zn", "O"), np.array(coords, dtype=float), **kw)


@bug(L2)
def test_L2_rattle_mutates_original(m):
    """Blocker 1: rattle adds noise in place, so the original changes and every copy shares one array."""
    s = _structure(m)
    before = s.frac_coords.copy()
    copies = m.augment([s], n_copies=3, sigma=0.01)
    assert np.array_equal(s.frac_coords, before)
    assert not np.array_equal(copies[0].frac_coords, copies[1].frac_coords)


@bug(L2)
def test_L2_post_init_aliases_caller_array(m):
    """Blocker 2: __post_init__ wraps the caller's array in place and keeps a reference to it."""
    traj = np.array([[[1.2, -0.3, 0.5], [0.6, 0.7, 0.8]]] * 2)  # unwrapped MD frames
    m.Structure(np.eye(3) * 10.0, ("Zn", "O"), traj[0])
    assert traj[0, 0, 0] == 1.2
    coords = np.array(COORDS)
    s = m.Structure(np.eye(3) * 10.0, ("Zn", "O"), coords)
    coords[0, 0] = 0.9  # the caller reuses its buffer
    assert s.frac_coords[0, 0] == 0.1


@bug(L2, raises=ValueError)
def test_L2_generated_eq_with_arrays(m):
    """Major 3: the generated __eq__ can't compare arrays, so deduplicating a real repeat raises."""
    a, b = _structure(m), _structure(m)
    assert m.deduplicate([a, b]) == [a]
    assert a == b


@bug(L2, raises=TypeError)
def test_L2_species_list_unhashable(m):
    """Major 4: from_dict keeps species as the JSON list, which is unhashable."""
    d = {"lattice": (np.eye(3) * 10).tolist(), "species": ["Zn", "O"], "frac_coords": COORDS}
    s = m.Structure.from_dict(json.loads(json.dumps(d)))
    assert m.deduplicate([s]) == [s]


@bug(L2)
def test_L2_replace_shares_metadata(m):
    """Major 5: replace() is shallow, so tagging the rattled copy also tags the original."""
    s = _structure(m, metadata={"source": "CoRE-MOF"})
    m.rattle(s, 0.01, np.random.default_rng(0))
    assert s.metadata == {"source": "CoRE-MOF"}


@bug(L2)
def test_L2_int_lattice_hash(m):
    """Minor 6: an integer lattice from JSON hashes differently from the same float lattice."""
    a = m.Structure.from_dict({"lattice": [[10, 0, 0], [0, 10, 0], [0, 0, 10]], "species": ("Zn", "O"), "frac_coords": COORDS})
    b = m.Structure.from_dict({"lattice": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], "species": ("Zn", "O"), "frac_coords": COORDS})
    assert len(m.deduplicate([a, b])) == 1


@bug(L2, raises=AttributeError)
def test_L2_list_inputs(m):
    """Minor 7: validation assumes ndarrays, so nested lists raise AttributeError."""
    s = m.Structure([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], ("Zn",), [[0.1, 0.2, 0.3]])
    assert np.isclose(s.volume, 1000.0)


@fine(L2)
def test_L2_with_energy_replace(m):
    """Red herring: with_energy via dataclasses.replace returns a new object and leaves the original alone."""
    s = _structure(m)
    s2 = s.with_energy(-1.5)
    assert s2.energy == -1.5 and s.energy is None
    assert np.array_equal(s2.frac_coords, s.frac_coords)


@fine(L2)
def test_L2_cached_property_on_frozen(m):
    """Red herring: cached_property works on a frozen dataclass (it writes to __dict__ directly)."""
    s = _structure(m)
    assert np.allclose(s.cart_coords, s.frac_coords @ s.lattice)
    assert s.cart_coords is s.cart_coords


# --- L3: campaign config ------------------------------------------------------------


def _config_file(tmp_path, d=None):
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(d or {"name": "co2"}))
    return str(path)


@bug(L3)
def test_L3_env_values_not_converted(m, tmp_path):
    """Blocker 1: env overrides stay strings, so AL_USE_CACHED_LABELS=false is truthy."""
    cfg = m.load_config(_config_file(tmp_path), environ={"AL_USE_CACHED_LABELS": "false", "AL_BUDGET": "200"})
    assert cfg.use_cached_labels is False
    assert cfg.budget == 200


@bug(L3)
def test_L3_forcefield_default_shared(m, tmp_path):
    """Blocker 2: default_factory returns DEFAULT_FORCEFIELD itself, so one edit changes every config."""
    a = m.SimulationConfig()
    a.forcefield["cutoff"] = 14.0
    assert m.SimulationConfig().forcefield["cutoff"] == 12.8
    a.forcefield["cutoff"] = 12.8
    cfg = m.load_config(_config_file(tmp_path), environ={"AL_SIMULATION__FORCEFIELD__CUTOFF": "14.0"})
    assert m.DEFAULT_FORCEFIELD["cutoff"] == 12.8
    assert cfg.simulation.forcefield["cutoff"] == 14.0


@bug(L3)
def test_L3_override_mutates_shared_default(m, tmp_path):
    """Blocker 3: object.__setattr__ edits the shared GCMCConfig() default, so other configs change too."""
    m.load_config(_config_file(tmp_path), environ={"AL_SIMULATION__TEMPERATURE": "273"})
    assert m.CampaignConfig(name="other").simulation.temperature == 298.0


@bug(L3)
def test_L3_override_skips_validation(m, tmp_path):
    """Blocker 3: overrides bypass __post_init__, so a pressure given in Pa is accepted."""
    path = _config_file(tmp_path)
    assert rejects(lambda: m.load_config(path, environ={"AL_SIMULATION__PRESSURE": "101325"}))


@bug(L3)
def test_L3_subclass_skips_base_validation(m):
    """Blocker 4: GCMCConfig.__post_init__ doesn't call super(), so the base checks never run."""
    assert rejects(lambda: m.GCMCConfig(pressure=101325.0))
    assert rejects(lambda: m.GCMCConfig(temperature=-5.0))


@bug(L3)
def test_L3_from_dict_nested(m, tmp_path):
    """Major 5: from_dict leaves nested sections as dicts, so a saved config doesn't load back."""
    cfg = m.CampaignConfig(name="co2", simulation=m.GCMCConfig(temperature=273.0), output_dir=str(tmp_path))
    loaded = m.load_config(cfg.save(), environ={})
    assert loaded == cfg
    assert loaded.simulation.temperature == 273.0


@bug(L3)
def test_L3_unknown_override_ignored(m, tmp_path):
    """Major 6: a misspelt override (AL_BUGDET) is silently stored as a new attribute."""
    path = _config_file(tmp_path)
    assert rejects(lambda: m.load_config(path, environ={"AL_BUGDET": "50"}))


@bug(L3)
def test_L3_run_id_ignores_compare_false(m):
    """Major 7: run_id hashes notes (compare=False) and run options, so the same settings get new ids."""
    a = m.CampaignConfig(name="co2")
    b = dataclasses.replace(a, notes="rerun after fixing the CIFs")
    assert a == b
    assert a.run_id == b.run_id
    assert dataclasses.replace(a, use_cached_labels=False).run_id == a.run_id


@bug(L3)
def test_L3_name_escapes_output_dir(m, tmp_path):
    """Minor 9: a name with path separators puts the run directory outside output_dir."""
    assert rejects(lambda: m.CampaignConfig(name="../../shared/co2", output_dir=str(tmp_path)))


@bug(L3)
def test_L3_single_member_ensemble(m):
    """Minor 10: ensemble_size=1 gives zero uncertainty, so UCB silently becomes greedy."""
    assert rejects(lambda: m.CampaignConfig(name="co2", surrogate=m.SurrogateConfig(ensemble_size=1)))


@fine(L3)
def test_L3_shared_frozen_default_is_safe(m):
    """Red herring: the shared GCMCConfig() default is frozen, so ordinary assignment can't change it."""
    a, b = m.CampaignConfig(name="a"), m.CampaignConfig(name="b")
    assert rejects(lambda: setattr(a.simulation, "temperature", 273.0), dataclasses.FrozenInstanceError)
    assert b.simulation.temperature == 298.0


@fine(L3)
def test_L3_run_id_short_and_stable(m):
    """Red herring: a 12-hex-character run_id is stable across instances."""
    a, b = m.CampaignConfig(name="co2"), m.CampaignConfig(name="co2")
    assert a.run_id == b.run_id and len(a.run_id) == 12


@fine(L3)
def test_L3_environ_default_is_live(m, tmp_path, monkeypatch):
    """Red herring: environ=os.environ is bound at import, but it's the live mapping."""
    for key in [k for k in m.os.environ if k.startswith("AL_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("AL_NOTES", "set after import")
    assert m.load_config(_config_file(tmp_path)).notes == "set after import"
