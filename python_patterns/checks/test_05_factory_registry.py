"""Checks for the 05_factory_registry drills: every planted bug fails on the PR code and passes on the fix."""
import sys
from pathlib import Path

import numpy as np
import pytest

from harness import bug, fine


def _raised(exc_type, fn, *args, **kwargs) -> bool:
    """True if fn(*args, **kwargs) raises exc_type (other exceptions propagate)."""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    return False


# --- L1: featuriser factory ----------------------------------------------------------------

L1 = "05_factory_registry/L1_featuriser_factory"


@pytest.fixture
def zeopp_cwd(tmp_path, monkeypatch):
    """Run in a temp dir that holds the Zeo++ file at the factory's default (relative) path."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    np.savez(
        tmp_path / "data" / "zeopp_descriptors.npz",
        mof_id=np.array(["mof-1", "mof-2"]),
        descriptors=np.array([[6.1, 1500.0], [3.2, 400.0]]),
    )
    return tmp_path


@bug(L1)
def test_L1_config_not_mutated(m, zeopp_cwd):
    """Blocker 1: pop() empties the caller's config, so the next CV fold silently gets the default featuriser."""
    cfg = {"kind": "rdf"}
    first = m.make_featuriser(cfg)
    second = m.make_featuriser(cfg)
    assert cfg == {"kind": "rdf"}
    assert type(first) is type(second) is m.RDFFeaturiser


@bug(L1, raises=TypeError)
def test_L1_featuriser_specific_params(m, zeopp_cwd):
    """Major 2: every featuriser is built with the same params, so the documented rdf example crashes."""
    featuriser = m.make_featuriser({"kind": "rdf", "params": {"n_bins": 64}})
    assert len(featuriser.edges) == 65


@bug(L1, raises=FileNotFoundError)
def test_L1_builds_only_the_requested_featuriser(m, tmp_path, monkeypatch):
    """Major 3: every call loads the Zeo++ file, even when another featuriser was asked for."""
    monkeypatch.chdir(tmp_path)  # no data/zeopp_descriptors.npz here, as on a laptop or in CI
    assert isinstance(m.make_featuriser({"kind": "composition"}), m.CompositionFeaturiser)


@bug(L1)
def test_L1_missing_kind_rejected(m, zeopp_cwd):
    """Major 4: a config whose 'kind' key is missing or misspelled silently gets the composition featuriser."""
    assert _raised((KeyError, ValueError), m.make_featuriser, {"type": "rdf"})


@bug(L1)
def test_L1_unknown_kind_lists_options(m, zeopp_cwd):
    """Minor 5: an unknown kind gives a bare KeyError that doesn't say what the valid names are."""
    with pytest.raises((KeyError, ValueError)) as excinfo:
        m.make_featuriser({"kind": "soap"})
    assert "composition" in str(excinfo.value)


@fine(L1)
def test_L1_zeopp_rows_outlive_the_closed_file(m, zeopp_cwd):
    """Red herring: the npz file is closed after __init__, but the rows were read into memory first."""
    featuriser = m.ZeoppFeaturiser()

    class Structure:
        mof_id = "mof-2"

    assert featuriser.featurise(Structure())[0] == pytest.approx(3.2)


# --- L2: decorator registry of screening filters -------------------------------------------

L2 = "05_factory_registry/L2_filter_registry"

CANDIDATES = [
    {"id": "a", "pore_diameter": 6.0, "density": 0.9, "surface_area": 2000.0, "metals": ["Zn"]},
    {"id": "b", "pore_diameter": 2.5, "density": 0.8, "surface_area": 1800.0, "metals": ["Cu"]},
    {"id": "c", "pore_diameter": 7.0, "density": 2.4, "surface_area": 900.0, "metals": ["Zr"]},
]


@bug(L2)
def test_L2_duplicate_name(m):
    """Blocker 1: a second @register_filter("density") silently replaces the density filter."""
    assert type(m.make_filter({"name": "density"})).__name__ == "DensityFilter"
    assert _raised(ValueError, m.register_filter("density"), type("AnotherDensity", (m.Filter,), {}))


@bug(L2)
def test_L2_unknown_keys_rejected(m):
    """Blocker 2: the base class swallows unknown keys, so a typo silently runs with the default."""
    assert _raised(TypeError, m.make_filter, {"name": "pore_size", "min_diamter": 4.0})


@bug(L2)
def test_L2_decorator_returns_the_class(m):
    """Major 3: the decorator returns None, so every decorated class name in the module is None."""
    assert m.PoreSizeFilter is m.FILTERS["pore_size"]


@bug(L2)
def test_L2_filters_built_once(m, monkeypatch):
    """Major 4: screen() rebuilds every filter for every candidate."""
    cls = m.FILTERS["pore_size"]
    original_init = cls.__init__
    built = []

    def counting_init(self, *args, **kwargs):
        built.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(cls, "__init__", counting_init)
    kept = m.screen(CANDIDATES * 100, [{"name": "pore_size", "min_diameter": 4.0}])
    assert len(kept) == 200
    assert len(built) == 1


@bug(L2, raises=ValueError)
def test_L2_late_registration_visible(m):
    """Major 5: FILTER_NAMES is a snapshot, so filters registered later (genfilters.py) are rejected."""

    class NoveltyFilter(m.Filter):
        def passes(self, record):
            return record["id"] != "b"

    m.register_filter("novelty")(NoveltyFilter)
    assert [c["id"] for c in m.screen(CANDIDATES, [{"name": "novelty"}])] == ["a", "c"]


@bug(L2)
def test_L2_string_allowed_rejected(m):
    """Minor 7: allowed="Zn" becomes {"Z", "n"} and silently rejects every framework."""
    assert _raised((TypeError, ValueError), m.make_filter, {"name": "metals", "allowed": "Zn"})


@bug(L2)
def test_L2_unknown_name_lists_filters(m):
    """Minor 8: the unknown-filter error doesn't list the registered names."""
    with pytest.raises(ValueError) as excinfo:
        m.make_filter({"name": "pore"})
    assert "pore_size" in str(excinfo.value)


@fine(L2)
def test_L2_subclass_kwargs_forwarding_is_fine(m):
    """Red herring: subclasses forwarding **kwargs to super().__init__ is the normal cooperative pattern."""
    pore = m.make_filter({"name": "pore_size", "min_diameter": 4.0, "comment": "CO2 is ~3.3 Å"})
    assert pore.min_diameter == 4.0
    assert [c["id"] for c in m.screen(CANDIDATES, [{"name": "metals", "allowed": ["Zn", "Zr"]}])] == ["a", "c"]


# --- L3: plugin engines for the simulation service -----------------------------------------

L3 = "05_factory_registry/L3_engine_plugins"

UFF = "Zn 62.4 2.46\nO 30.2 3.12\nC 52.8 3.43\nH 22.1 2.57\n"


@pytest.fixture
def ff(m, tmp_path, monkeypatch):
    """Point the module at a temp force-field directory with a UFF.def in it."""
    ff_dir = tmp_path / "forcefields"
    ff_dir.mkdir()
    (ff_dir / "UFF.def").write_text(UFF)
    monkeypatch.setattr(m, "FORCEFIELD_DIR", ff_dir)
    return ff_dir


@pytest.fixture
def plugin_dir(tmp_path, monkeypatch):
    """A directory on sys.path for throwaway plugin modules; they are removed from sys.modules afterwards."""
    path = tmp_path / "plugins"
    path.mkdir()
    monkeypatch.syspath_prepend(path)
    before = set(sys.modules)
    yield path
    for name in set(sys.modules) - before:
        if name.startswith("l3_"):
            del sys.modules[name]


GCMC = {"engine": "gcmc", "structure_id": "mof-00017"}


@bug(L3)
def test_L3_plugin_import_is_allowlisted(m, ff, plugin_dir, tmp_path):
    """Blocker 1: any importable module named in a request is imported, and its top-level code runs."""
    marker = tmp_path / "ran"
    (plugin_dir / "l3_not_a_plugin.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    try:
        m.handle_request({**GCMC, "plugin": "l3_not_a_plugin"})
    except Exception:
        pass
    assert not marker.exists()


@bug(L3)
def test_L3_params_of_later_requests_respected(m, ff):
    """Blocker 2: engines are cached by name only, so a later request silently gets the first one's params."""
    m.handle_request({**GCMC, "params": {"cycles": 2000}})
    files = m.handle_request({**GCMC, "params": {"cycles": 50000}})
    assert "NumberOfCycles 50000" in files["simulation.input"]


@bug(L3)
def test_L3_conditions_do_not_leak(m, ff):
    """Blocker 3: prepare() updates the engine's own conditions, so 77 K sticks for later jobs."""
    engine = m.ENGINES["gcmc"]()
    engine.prepare("mof-1", temperature=77)
    files = engine.prepare("mof-2")
    assert "ExternalTemperature 298.0" in files["simulation.input"]


@bug(L3)
def test_L3_name_collision_does_not_replace_engine(m, ff):
    """Blocker 4: any later class called GCMCEngine silently takes over "gcmc" for every request."""
    original = m.ENGINES["gcmc"]

    class GCMCEngine(m.RaspaEngine):  # e.g. a fork in someone's plugin module
        def prepare(self, structure_id, **conditions):
            return {"simulation.input": "SimulationType MonteCarlo\nNumberOfCycles 1\n"}

    assert m.ENGINES["gcmc"] is original
    with pytest.raises(ValueError):  # the fix registers by explicit name and rejects duplicates

        class AnotherGCMC(m.RaspaEngine, name="gcmc"):
            def prepare(self, structure_id, **conditions):
                return {}


@bug(L3)
def test_L3_abstract_base_not_offered(m, ff):
    """Major 5: the shared RaspaEngine base registers as "raspa" and is offered by GET /engines."""
    assert m.available_engines() == ["gcmc", "widom"]


@bug(L3)
def test_L3_forcefield_read_once(m, ff, monkeypatch):
    """Major 7: the mixing_rules property re-reads and re-parses the force-field file for every job."""
    reads = []
    real_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        reads.append(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    for i in range(5):
        m.handle_request({**GCMC, "structure_id": f"mof-{i}"})
    assert len(reads) <= 1


@bug(L3, raises=ValueError)
def test_L3_forcefield_comments_and_blank_lines(m, ff):
    """Minor 8: a comment or blank line in the force-field file crashes the parser."""
    (ff / "UFF.def").write_text("# UFF, epsilon in K, sigma in Angstrom\n\n" + UFF)
    files = m.handle_request(GCMC)
    assert files["force_field_mixing_rules.def"].startswith("Zn lennard-jones 62.4 2.46")


@bug(L3)
def test_L3_unknown_engine_message(m, ff):
    """Minor 9: an unknown engine gives a bare KeyError that doesn't list the available engines."""
    with pytest.raises((KeyError, ValueError)) as excinfo:
        m.handle_request({**GCMC, "engine": "gmc"})
    assert "widom" in str(excinfo.value)


@bug(L3)
def test_L3_structure_id_cannot_inject_directives(m, ff):
    """Minor 10: a newline in structure_id injects extra lines into the RASPA input file."""
    assert _raised(ValueError, m.handle_request, {**GCMC, "structure_id": "mof-1\nNumberOfCycles 999999999"})


@fine(L3)
def test_L3_repeat_import_is_cached(m, ff, plugin_dir, tmp_path, monkeypatch):
    """Red herring: calling import_module on every request runs the plugin's code only once."""
    counter = tmp_path / "imports.txt"
    (plugin_dir / "l3_counting_plugin.py").write_text(f"with open({str(counter)!r}, 'a') as f:\n    f.write('x')\n")
    monkeypatch.setattr(m, "ALLOWED_PLUGINS", frozenset({"l3_counting_plugin"}), raising=False)
    for i in range(3):
        m.handle_request({**GCMC, "plugin": "l3_counting_plugin", "structure_id": f"mof-{i}"})
    assert counter.read_text() == "x"


@fine(L3)
def test_L3_default_conditions_class_dict_untouched(m, ff):
    """Red herring: DEFAULT_CONDITIONS is a mutable class attribute, but __init__ copies it and nothing mutates it."""
    m.ENGINES["gcmc"]().prepare("mof-1", temperature=77)
    m.ENGINES["widom"]().prepare("mof-2", pressure=10.0)
    assert m.RaspaEngine.DEFAULT_CONDITIONS == {"temperature": 298.0, "pressure": 1e5}
