"""Checks for the 02_caching drills: every planted bug fails on the PR code and passes on the fix."""
import contextlib
import gc
import os
import sqlite3
import subprocess
import sys
import warnings
import weakref
from collections import OrderedDict

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


# --- L1: lru_cache on element lookups and pair energies ----------------------------------

L1 = "02_caching/L1_element_lookup"

# mass, electronegativity, epsilon (kcal/mol), sigma (angstrom): UFF-style values
ELEMENTS = {
    "H": (1.008, 2.20, 0.044, 2.571),
    "C": (12.011, 2.55, 0.105, 3.431),
    "O": (15.999, 3.44, 0.060, 3.118),
    "Zn": (65.38, 1.65, 0.124, 2.462),
}


@pytest.fixture
def db_reads(m, monkeypatch):
    """Serve element rows from ELEMENTS instead of SQLite, and record each read."""
    reads = []

    def fake_read(symbol):
        reads.append(symbol)
        return ELEMENTS.get(symbol)

    monkeypatch.setattr(m, "_read_element_row", fake_read)
    return reads


def _lj_reference(symbols, distances, probe="C"):
    total = 0.0
    for symbol, r in zip(symbols, distances):
        eps = (ELEMENTS[symbol][2] * ELEMENTS[probe][2]) ** 0.5
        sigma = (ELEMENTS[symbol][3] + ELEMENTS[probe][3]) / 2
        sr6 = (sigma / r) ** 6
        total += 4 * eps * (sr6 * sr6 - sr6)
    return total


@bug(L1)
def test_L1_cached_dict_mutated(m, db_reads):
    """Blocker 1: probe_energy writes mixed parameters into the cached dicts."""
    symbols = ["Zn", "O", "O", "C", "H"]
    distances = np.array([3.1, 3.4, 3.6, 4.0, 2.9])
    expected = _lj_reference(symbols, distances)
    assert m.probe_energy(symbols, distances) == pytest.approx(expected)
    assert m.probe_energy(symbols, distances) == pytest.approx(expected)
    assert m.element_properties("Zn")["epsilon"] == pytest.approx(0.124)


@bug(L1)
def test_L1_float_cache_grows_without_bound(m):
    """Major 2: pair_energy keeps one cache entry per distinct distance, for ever."""
    for r in np.linspace(2.5, 8.0, 5_000):
        m.pair_energy(float(r), 0.1, 3.4)
    info = getattr(m.pair_energy, "cache_info", None)
    assert info is None or info().currsize < 5_000


@bug(L1, raises=TypeError)
def test_L1_list_argument_unhashable(m, db_reads):
    """Major 3: mean_property is cached, so the list callers pass raises TypeError."""
    assert m.mean_property(["Zn", "O", "O"], "mass") == pytest.approx((65.38 + 2 * 15.999) / 3)


@bug(L1)
def test_L1_unknown_element_message(m, db_reads):
    """Minor 4: an unknown element fails with an unpacking TypeError that doesn't name it."""
    with pytest.raises(Exception) as excinfo:
        m.element_properties("Xx")
    assert "Xx" in str(excinfo.value)


@bug(L1)
def test_L1_connection_not_closed(m, tmp_path, monkeypatch):
    """Nit 5: `with sqlite3.connect(...)` commits but never closes the connection."""
    path = tmp_path / "elements.sqlite"
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE elements (symbol, mass, electronegativity, epsilon, sigma)")
        conn.execute("INSERT INTO elements VALUES ('Zn', 65.38, 1.65, 0.124, 2.462)")
        conn.commit()
    monkeypatch.setattr(m, "ELEMENT_DB", str(path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert m._read_element_row("Zn") == (65.38, 1.65, 0.124, 2.462)
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


@bug(L1)
def test_L1_lengths_not_checked(m, db_reads):
    """Nit 6: zip silently drops atoms when symbols and distances have different lengths."""
    assert _raised(ValueError, m.probe_energy, ["Zn", "O", "O"], np.array([3.1, 3.4]))


@fine(L1)
def test_L1_element_lookup_cached_once(m, db_reads):
    """Red herring: an unbounded cache over element symbols is fine; each element is read once."""
    for _ in range(3):
        for symbol in ("Zn", "O", "C"):
            m.element_properties(symbol)
    assert sorted(db_reads) == ["C", "O", "Zn"]


# --- L2: cached_property and lru_cache on a mutable Framework ----------------------------

L2 = "02_caching/L2_framework_properties"

FRAMEWORK = ["Zn"] * 4 + ["O"] * 13 + ["C"] * 24 + ["H"] * 12  # 53 atoms
WATER = ["O"] * 4 + ["H"] * 8  # 4 guest water molecules left in the pores
GUESTS = list(range(len(FRAMEWORK), len(FRAMEWORK) + len(WATER)))


def _density(species, a):
    mass = sum({"Zn": 65.38, "O": 15.999, "C": 12.011, "H": 1.008}[s] for s in species)
    return mass / 6.02214076e23 / (a**3 * 1e-24)


def _toy_predictor(descriptor, temperature, pressure):
    """Uptake that depends on density and conditions, so stale inputs show up."""
    return 10.0 * descriptor[0] * pressure * 298.15 / temperature


def _solvated(m, a=12.0, mof_id=None, predictor=_toy_predictor):
    species = FRAMEWORK + WATER
    coords = np.random.default_rng(0).random((len(species), 3))
    return m.Framework(np.eye(3) * a, species, coords, predictor, mof_id=mof_id)


@bug(L2)
def test_L2_cached_density_stale_after_mutation(m):
    """Blocker 1: cached volume and density aren't invalidated by remove_guests or strain."""
    fw = _solvated(m)
    assert fw.density == pytest.approx(_density(FRAMEWORK + WATER, 12.0))
    fw.remove_guests(GUESTS)
    assert fw.density == pytest.approx(_density(FRAMEWORK, 12.0))
    fw.strain(1.01)
    assert fw.volume == pytest.approx((12.0 * 1.01) ** 3)


@bug(L2)
def test_L2_activated_copy_inherits_stale_cache(m):
    """Blocker 1: copy.copy in activated() copies the cached density into the new object."""
    fw = _solvated(m)
    fw.density  # e.g. printed by a filter before activation
    activated = fw.activated(GUESTS)
    assert activated.density == pytest.approx(_density(FRAMEWORK, 12.0))


@bug(L2)
def test_L2_unregistered_candidates_share_uptake(m):
    """Blocker 2: _uptake is keyed by mof_id equality, so different mof_id=None candidates share results."""
    small, large = _solvated(m, a=12.0), _solvated(m, a=14.0)
    assert small.uptake(298.15, 1.0) != pytest.approx(large.uptake(298.15, 1.0))


@bug(L2)
def test_L2_activated_copy_reuses_solvated_uptake(m):
    """Blocker 2: the activated copy has the same mof_id, so it gets the solvated framework's uptake."""
    fw = _solvated(m, mof_id="MOF-5")
    solvated_uptake = fw.uptake(298.15, 1.0)
    activated = fw.activated(GUESTS)
    expected = _toy_predictor([_density(FRAMEWORK, 12.0)], 298.15, 1.0)
    assert activated.uptake(298.15, 1.0) == pytest.approx(expected)
    assert expected != pytest.approx(solvated_uptake)


@bug(L2)
def test_L2_conditions_rounded_before_prediction(m):
    """Blocker 3: uptake rounds the conditions, so 0.15 bar is predicted at 0.1 bar."""
    seen = []

    def recording_predictor(descriptor, temperature, pressure):
        seen.append((temperature, pressure))
        return pressure

    fw = _solvated(m, predictor=recording_predictor)
    assert fw.uptake(298.15, 0.15) == pytest.approx(0.15)
    assert seen == [(298.15, 0.15)]


@bug(L2)
def test_L2_method_cache_keeps_frameworks_alive(m):
    """Major 4: lru_cache on a method holds a reference to every framework it has seen."""
    fw = _solvated(m, mof_id="MOF-5")
    fw.uptake(298.15, 1.0)
    ref = weakref.ref(fw)
    del fw
    gc.collect()
    assert ref() is None


@bug(L2)
def test_L2_unregistered_candidates_compare_equal(m):
    """Major 5: every mof_id=None candidate compares equal, so a set keeps only one of them."""
    candidates = [_solvated(m, a=a) for a in (12.0, 13.0, 14.0)]
    assert len(set(candidates)) == 3


@fine(L2)
def test_L2_density_units(m):
    """Red herring: the 1e-24 factor is the right A^3 -> cm^3 conversion."""
    fw = m.Framework(np.eye(3) * 10.0, ["C"] * 10, np.zeros((10, 3)), _toy_predictor)
    assert fw.density == pytest.approx(10 * 12.011 / 6.02214076e23 / 1e-21)


# --- L3: hand-rolled prediction cache in front of a surrogate model ----------------------

L3 = "02_caching/L3_prediction_cache"
T, P = 298.15, 0.15
X1 = np.array([1.0, 2.0])  # mean = sum(features) * pressure = 0.45, std = 0.5


class FakeModel:
    """mean = sum(features) * pressure + offset, std = 0.5; can fail its first calls."""

    def __init__(self, version="run-41", offset=0.0, fail_times=0):
        self.version = version
        self.offset = offset
        self.fail_times = fail_times
        self.calls = 0

    def predict(self, X, temperature, pressure):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("CUDA out of memory")
        X = np.asarray(X, dtype=float)
        mean = X.sum(axis=1) * pressure + self.offset
        return np.stack([mean, np.full(len(X), 0.5)], axis=1)


def _service(m, model=None, **kwargs):
    model = model or FakeModel()
    return m.SurrogateService(lambda path: model, "run-41", **kwargs)


@bug(L3)
def test_L3_reload_serves_old_predictions(m):
    """Blocker 1: the model version isn't in the key, so a reload keeps serving the old model's predictions."""
    svc = m.SurrogateService(lambda path: FakeModel(version=path, offset=10.0 if path == "run-42" else 0.0), "run-41")
    before = svc.predict_one(X1, T, P)[0]
    svc.reload_model("run-42")
    assert svc.predict_one(X1, T, P)[0] == pytest.approx(before + 10.0)


@bug(L3)
def test_L3_ucb_grows_on_every_call(m):
    """Blocker 2: predict_ucb adds beta*std into the cached array, so each call returns a bigger UCB."""
    svc = _service(m)
    expected = 0.45 + 2.0 * 0.5
    assert svc.predict_ucb(X1, T, P) == pytest.approx(expected)
    assert svc.predict_ucb(X1, T, P) == pytest.approx(expected)


@bug(L3)
def test_L3_cached_array_returned_by_reference(m):
    """Blocker 2: callers get the cached array itself, so their edits change what everyone is served."""
    svc = _service(m)
    svc.predict_one(X1, T, P)[0] = -999.0
    assert svc.predict_one(X1, T, P)[0] == pytest.approx(0.45)


@bug(L3)
def test_L3_transient_failure_cached(m):
    """Blocker 3: a failed prediction is cached as None and served from then on."""
    svc = _service(m, FakeModel(fail_times=1))
    assert svc.predict_one(X1, T, P) is None  # the failure itself is reported
    assert svc.predict_one(X1, T, P) is not None


@bug(L3, raises=KeyError)
def test_L3_eviction_between_check_and_read(m):
    """Major 4: `key in cache` then `cache[key]` is check-then-act; an eviction in between raises KeyError."""
    svc = _service(m)
    svc.predict_one(X1, T, P)

    class EvictsAfterCheck(OrderedDict):
        """Simulates another request thread evicting the entry right after the membership check."""

        def __contains__(self, key):
            found = super().__contains__(key)
            if found:
                self.popitem(last=False)
            return found

    svc.cache._data = EvictsAfterCheck(svc.cache._data)
    assert svc.predict_one(X1, T, P) is not None


@bug(L3)
def test_L3_lru_is_fifo(m):
    """Major 5: a hit doesn't refresh recency, so the "LRU" evicts the hottest entries first."""
    cache = m.PredictionCache(max_entries=2)
    cache["a"] = np.array([1.0])
    cache["b"] = np.array([2.0])
    cache["a"]  # "a" is now the most recently used
    cache["c"] = np.array([3.0])
    assert "a" in cache and "b" not in cache


@bug(L3)
def test_L3_key_differs_between_processes(m):
    """Major 6: make_key uses hash(), which is salted per process, so saved keys never match after a restart."""
    code = (
        "import importlib.util, numpy as np\n"
        f"spec = importlib.util.spec_from_file_location('drill', {m.__file__!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "print(mod.make_key(np.arange(4.0), 298.15, 0.15))\n"
    )
    keys = set()
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True, text=True, check=True, timeout=60,
        )
        keys.add(result.stdout.strip())
    assert len(keys) == 1, keys


@bug(L3)
def test_L3_per_key_stats_unbounded(m):
    """Major 7: key_hits keeps a counter for every key ever hit, even after it's evicted."""
    cache = m.PredictionCache(max_entries=10)
    for i in range(100):
        cache[f"k{i}"] = np.array([float(i)])
        cache[f"k{i}"]
    assert len(cache.key_hits) <= cache.max_entries


@bug(L3)
def test_L3_batch_calls_model_per_row(m):
    """Major 8: predict_batch calls the model once per row instead of once for all the misses."""
    model = FakeModel()
    svc = _service(m, model)
    X = np.arange(200.0).reshape(100, 2)
    preds = svc.predict_batch(X, T, P)
    np.testing.assert_allclose(preds[7], [X[7].sum() * P, 0.5])
    assert model.calls == 1


@bug(L3)
def test_L3_interrupted_save_breaks_next_start(m, tmp_path, monkeypatch):
    """Major 9: save truncates the file before writing, so a crash mid-save stops the next start-up."""
    path = str(tmp_path / "warm_start.pkl")
    svc = _service(m, warm_start_path=path)
    svc.predict_one(X1, T, P)
    svc.shutdown()  # a good file on disk

    def dies_mid_write(obj, f, *args, **kwargs):
        f.write(b"\x80\x05\x95")  # a few bytes, then the disk fills up (or the pod is killed)
        raise OSError("No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(m.pickle, "dump", dies_mid_write)
        with contextlib.suppress(OSError):
            svc.shutdown()
    try:
        restarted = _service(m, warm_start_path=path)
    except Exception as exc:
        raise AssertionError(f"the service can't start after an interrupted save: {exc!r}")
    assert len(restarted.cache) == 1


@bug(L3)
def test_L3_key_ignores_shape_and_dtype(m):
    """Minor 10: the key uses the raw bytes only, so arrays of different shape or dtype can collide."""
    assert m.make_key(np.zeros((2, 3)), T, P) != m.make_key(np.zeros((3, 2)), T, P)
    assert m.make_key(np.zeros(2, dtype=np.float32), T, P) != m.make_key(np.zeros(1), T, P)


@bug(L3)
def test_L3_empty_shared_cache_is_replaced(m):
    """Minor 11: `cache or PredictionCache()` throws away an empty cache that was passed in to be shared."""
    shared = m.PredictionCache(max_entries=1_000_000)
    svc = _service(m, cache=shared)
    assert svc.cache is shared


@bug(L3)
def test_L3_load_ignores_size_limit(m, tmp_path):
    """Minor 12: load bypasses max_entries, and one pop per insert never brings the size back down."""
    path = str(tmp_path / "warm_start.pkl")
    big = m.PredictionCache(max_entries=100)
    for i in range(100):
        big[f"k{i}"] = np.array([1.0, 0.5])
    big.save(path)
    small = m.PredictionCache(max_entries=10)
    small.load(path)
    small["new"] = np.array([1.0, 0.5])
    assert len(small) <= small.max_entries


@fine(L3)
def test_L3_eviction_removes_oldest(m):
    """Red herring: popitem(last=False) evicts from the least recently used end, which is right."""
    cache = m.PredictionCache(max_entries=2)
    for key in ("a", "b", "c"):
        cache[key] = np.array([0.0])
    assert "a" not in cache and "b" in cache and "c" in cache


@fine(L3)
def test_L3_warm_start_round_trip(m, tmp_path):
    """Red herring: pickling the cache to a file only this service writes is fine (here, within one process)."""
    path = str(tmp_path / "warm_start.pkl")
    svc = _service(m, warm_start_path=path)
    svc.predict_one(X1, T, P)
    svc.shutdown()
    model = FakeModel()
    restarted = _service(m, model, warm_start_path=path)
    assert restarted.predict_one(X1, T, P) is not None and model.calls == 0
