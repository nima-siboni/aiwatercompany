"""Checks for pattern 09: singletons, module-level state and dependency injection.

Run: uv run --with pytest --with numpy pytest python_patterns/checks/test_09_singleton_di.py -v
"""
import collections
import importlib.util
import json
import os
import sqlite3
import sys
import threading
import types
from contextlib import closing

import pytest

from harness import bug, fine

L1 = "09_singleton_di/L1_settings_singleton"
L2 = "09_singleton_di/L2_results_db"
L3 = "09_singleton_di/L3_prediction_service"


# --------------------------------------------------------------------------- L1: Settings


@bug(L1)
def test_L1_override_survives_next_call(m):
    """Blocker 1: every Settings() call re-runs __init__ and wipes earlier overrides."""
    m.Settings().override(results_db="scratch.sqlite")
    assert m.Settings().results_db == "scratch.sqlite"


@bug(L1)
def test_L1_env_read_at_import(m, monkeypatch, tmp_path):
    """Major 2: AIWC_RESULTS_DB is read at import time, so setting it later has no effect."""
    db = str(tmp_path / "test.sqlite")
    monkeypatch.setenv("AIWC_RESULTS_DB", db)
    assert m.Settings().results_db == db


@bug(L1)
def test_L1_dry_run_flag_parsing(m, monkeypatch):
    """Major 3: AIWC_DRY_RUN=0 gives the string "0", which is truthy."""
    monkeypatch.setenv("AIWC_DRY_RUN", "0")
    assert m.Settings().dry_run is False


@bug(L1)
def test_L1_repr_hides_token(m, monkeypatch):
    """Major 4: repr(Settings()) includes the API token, so logging the settings leaks it."""
    monkeypatch.setenv("AIWC_API_TOKEN", "tok-s3cret")
    assert "tok-s3cret" not in repr(m.Settings())


@bug(L1)
def test_L1_override_rejects_unknown_names(m):
    """Minor 5: a misspelt override is accepted silently and changes nothing."""
    settings = m.Settings()
    try:
        settings.override(gcmc_cylces=500)
    except (AttributeError, TypeError, ValueError):
        return
    raise AssertionError(f"typo accepted; gcmc_cycles is still {settings.gcmc_cycles}")


@fine(L1)
def test_L1_element_table_cache(m, tmp_path):
    """Red herring: the functools.cache factory parses the table once and can be reset."""
    path = tmp_path / "elements.json"
    path.write_text(json.dumps({"Zn": {"electronegativity": 1.65}}))
    first = m.element_table(str(path))
    path.write_text(json.dumps({"Zn": {"electronegativity": 0.0}}))
    assert m.element_table(str(path)) is first
    m.element_table.cache_clear()
    assert m.element_table(str(path))["Zn"]["electronegativity"] == 0.0


# ------------------------------------------------------------------ L2: global SQLite connection

RESULT = ("MOF-5", 298.0, 1.0, 1.2)  # mof_id, temperature_k, pressure_bar, uptake_mmol_g


def committed_jobs(db):
    """Job IDs that another connection (another process) can see, i.e. committed rows."""
    with closing(sqlite3.connect(db)) as conn:
        return sorted(row[0] for row in conn.execute("SELECT job_id FROM results"))


@bug(L2)
def test_L2_second_path_is_ignored(m, tmp_path):
    """Blocker 1: get_db(path) only honours path on the first call."""
    first, second = str(tmp_path / "campaign_41.sqlite"), str(tmp_path / "campaign_42.sqlite")
    m.get_db(first)
    try:
        try:
            conn = m.get_db(second)
        except ValueError:
            return  # refusing loudly is fine
        opened = conn.execute("PRAGMA database_list").fetchone()[2]
        assert os.path.realpath(opened) == os.path.realpath(second)
    finally:
        m.close_db()


@bug(L2)
def test_L2_rollback_discards_batch(m, tmp_path, monkeypatch):
    """Blocker 2: rolling back one failed insert throws away every uncommitted result."""
    db = str(tmp_path / "results.sqlite")
    monkeypatch.setattr(m, "COMMIT_EVERY", 3, raising=False)
    m.get_db(db)
    try:
        m.record_result("job-1", *RESULT)
        m.record_result("job-2", *RESULT)
        with pytest.raises(sqlite3.IntegrityError):
            m.record_result("job-1", *RESULT)  # a retried job reports a second time
        m.record_result("job-3", *RESULT)  # PR: the counter says 3, so this commits
        assert committed_jobs(db) == ["job-1", "job-2", "job-3"]
    finally:
        m.close_db()


@bug(L2)
def test_L2_close_commits_tail(m, tmp_path):
    """Blocker 3: close_db() discards the rows written since the last batch commit."""
    db = str(tmp_path / "results.sqlite")
    m.get_db(db)
    m.record_result("job-1", *RESULT)
    m.record_result("job-2", *RESULT)
    m.close_db()
    assert committed_jobs(db) == ["job-1", "job-2"]


@bug(L2, raises=sqlite3.OperationalError)
def test_L2_open_batch_blocks_other_writers(m, tmp_path):
    """Major 4: the open batch holds SQLite's write lock, so other writers get 'database is locked'."""
    db = str(tmp_path / "results.sqlite")
    m.get_db(db)
    try:
        m.record_result("job-1", *RESULT)
        # A second runner process, or a notebook, writing to the same file.
        with closing(sqlite3.connect(db, timeout=0.05)) as other, other:
            other.execute("INSERT INTO results VALUES ('job-x', 'UiO-66', 298.0, 1.0, 2.0)")
    finally:
        m.close_db()


@bug(L2, raises=sqlite3.ProgrammingError)
def test_L2_reopen_after_close(m, tmp_path):
    """Major 5: close_db() leaves the closed connection in _conn, so the store can't be reopened."""
    db = str(tmp_path / "results.sqlite")
    m.get_db(db)
    m.close_db()
    m.get_db(db)
    try:
        m.record_result("job-1", *RESULT)
    finally:
        m.close_db()
    assert committed_jobs(db) == ["job-1"]


@bug(L2)
def test_L2_lazy_init_race(m, tmp_path, monkeypatch):
    """Major 6: two threads in get_db() at the same time open two connections."""
    monkeypatch.setattr(m, "DEFAULT_DB", str(tmp_path / "results.sqlite"))
    real_connect = sqlite3.connect
    calls, opened, got = [], [], {}
    second_arrived = threading.Event()

    def other_thread():
        got["other"] = m.get_db()

    def connect(*args, **kwargs):
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            got["thread"] = threading.Thread(target=other_thread, daemon=True)
            got["thread"].start()
            # PR: the other thread also finds _conn is None and arrives here at once.
            # Fixed: it is waiting for the lock, so this wait just times out.
            second_arrived.wait(timeout=0.3)
        else:
            second_arrived.set()
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    try:
        mine = m.get_db()
        got["thread"].join(timeout=2)
        assert len(calls) == 1, f"{len(calls)} connections opened"
        assert got.get("other") is mine
    finally:
        for conn in opened:
            conn.close()


@fine(L2)
def test_L2_counter_is_locked(m, tmp_path, monkeypatch):
    """Red herring: _pending is a global counter, but it is only touched under _lock."""
    db = str(tmp_path / "results.sqlite")
    monkeypatch.setattr(m, "COMMIT_EVERY", 10, raising=False)
    m.get_db(db)  # opened before the worker threads start

    def worker(k):
        for i in range(25):
            m.record_result(f"job-{k}-{i:02d}", *RESULT)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    try:
        assert len(committed_jobs(db)) == 100  # ten full batches, all committed
    finally:
        m.close_db()


@fine(L2)
def test_L2_nothing_connects_at_import(m):
    """Red herring: the module-level lock and atexit hook are cheap; the connection is lazy."""
    assert m._conn is None


# ------------------------------------------------------------------ L3: prediction service

TOKENS = {"alice-token": ("alice", {"alpha"}), "bob-token": ("bob", {"beta"})}


class FakeStore:
    def __init__(self, structures=None):
        self.structures = structures or {}
        self.records = {}

    def ping(self):
        pass

    def get_structure(self, structure_id):
        return self.structures.get(structure_id, {"id": structure_id, "n_atoms": 10.0})

    def save_predictions(self, job_id, *, project, conditions, uptakes):
        self.records[job_id] = {
            "job_id": job_id, "project": project,
            "conditions": dict(conditions), "uptakes": dict(uptakes),
        }

    def load_predictions(self, job_id):
        return dict(self.records[job_id])


class FakeModel:
    """Predicts scale * n_atoms. Until load_weights() runs, scale is 0.0 (think random init)."""

    def __init__(self, scale=1.0, on_load=None):
        self.scale, self._trained_scale, self.on_load = 0.0, scale, on_load
        self.calls, self.loads = [], 0

    def load_weights(self, path):
        self.loads += 1
        if self.on_load:
            self.on_load()
        self.scale = self._trained_scale

    def predict(self, structures, *, temperature_k, pressure_bar):
        self.calls.append({"n": len(structures), "temperature_k": temperature_k})
        return [self.scale * s["n_atoms"] for s in structures]


def setup_service(m, store=None, model=None, model_factory=None):
    store = store or FakeStore()
    if model_factory is None:
        model = model or FakeModel()
        model_factory = lambda: model  # noqa: E731

    def verify_token(token):
        if token not in TOKENS:
            raise PermissionError("invalid token")
        name, projects = TOKENS[token]
        return m.User(name, frozenset(projects))

    m.configure(verify_token=verify_token, store=store, model_factory=model_factory)
    return store


def request(token, **body):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return {"headers": headers, "json": body}


def predict_request(token="alice-token", ids=("MOF-5",), project="alpha", **extra):
    return request(token, project=project, structure_ids=list(ids), **extra)


def load_another_worker(m):
    """A second copy of the module: what a second worker process or a restart starts with."""
    name = m.__name__ + "_worker2"
    spec = importlib.util.spec_from_file_location(name, m.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@bug(L3)
def test_L3_anonymous_request_runs_as_previous_user(m):
    """Blocker 1: a request without a token runs as whoever made the previous request."""
    setup_service(m)
    job_id = m.handle_predict(predict_request("alice-token"))["body"]["job_id"]
    anonymous = m.handle_get_results(request(None, job_id=job_id))
    assert anonymous["status"] in (401, 403)


@bug(L3)
def test_L3_concurrent_requests_share_current_user(m):
    """Blocker 1: concurrent requests share CURRENT_USER, so alice can read bob's job."""
    alice_loading, bob_loading = threading.Event(), threading.Event()

    class SlowStore(FakeStore):
        def load_predictions(self, job_id):
            if threading.current_thread().name == "alice":
                alice_loading.set()
                bob_loading.wait(timeout=2)  # bob's request arrives while alice's is in flight
            else:
                bob_loading.set()
            return super().load_predictions(job_id)

    store = SlowStore()
    store.records["pred-bob"] = {"job_id": "pred-bob", "project": "beta", "uptakes": {"MOF-5": 9.9}}
    setup_service(m, store=store)
    responses = {}

    def get_results(who, token):
        responses[who] = m.handle_get_results(request(token, job_id="pred-bob"))

    alice = threading.Thread(target=get_results, args=("alice", "alice-token"), name="alice")
    alice.start()
    assert alice_loading.wait(timeout=2)
    bob = threading.Thread(target=get_results, args=("bob", "bob-token"), name="bob")
    bob.start()
    bob.join(timeout=2)
    alice.join(timeout=2)
    assert responses["bob"]["status"] == 200
    assert responses["alice"]["status"] == 403


@bug(L3)
def test_L3_model_published_before_weights_load(m):
    """Blocker 2: _model is visible before load_weights() finishes, so a concurrent request uses it."""
    responses = {}

    def while_loading():
        other = threading.Thread(
            target=lambda: responses.update(other=m.handle_predict(predict_request()))
        )
        other.start()
        # PR: the other request sees _model is set and finishes with the unloaded model.
        # Fixed: it waits for _model_lock until loading is done, so this join times out.
        other.join(timeout=0.3)
        responses["thread"] = other

    setup_service(m, model=FakeModel(on_load=while_loading))
    first = m.handle_predict(predict_request())
    responses["thread"].join(timeout=2)
    assert first["body"]["uptake_mmol_g"] == {"MOF-5": 10.0}
    assert responses["other"]["body"]["uptake_mmol_g"] == {"MOF-5": 10.0}


@bug(L3)
def test_L3_request_conditions_leak_into_defaults(m):
    """Blocker 3: handle_predict updates DEFAULT_CONDITIONS in place, so later requests inherit them."""
    model = FakeModel()
    setup_service(m, model=model)
    m.handle_predict(predict_request(conditions={"temperature_k": 77.0}))
    later = m.handle_predict(predict_request())
    assert later["body"]["conditions"] == {"temperature_k": 298.0, "pressure_bar": 1.0}
    assert model.calls[-1]["temperature_k"] == 298.0


@bug(L3)
def test_L3_job_ids_repeat_across_processes(m):
    """Blocker 4: job IDs come from a per-process counter, so another process or a restart reuses them."""
    worker2 = load_another_worker(m)
    try:
        job_ids = []
        for module in (m, worker2):
            setup_service(module)
            job_ids.append(module.handle_predict(predict_request())["body"]["job_id"])
        assert job_ids[0] != job_ids[1]
    finally:
        sys.modules.pop(worker2.__name__, None)


@bug(L3)
def test_L3_one_model_call_per_request(m):
    """Major 7: the model is called once per structure instead of once per request."""
    model = FakeModel()
    setup_service(m, model=model)
    m.handle_predict(predict_request(ids=["MOF-5", "HKUST-1", "ZIF-8"]))
    assert len(model.calls) == 1


@bug(L3)
def test_L3_request_size_is_capped(m):
    """Major 8: nothing caps how many structures one request can ask for."""
    setup_service(m)
    limit = getattr(m, "MAX_STRUCTURES_PER_REQUEST", 1_000)
    response = m.handle_predict(predict_request(ids=[f"MOF-{i}" for i in range(limit + 1)]))
    assert response["status"] == 400


@bug(L3)
def test_L3_metrics_lose_updates(m, monkeypatch):
    """Minor 9: METRICS[...] += ... is a read-modify-write, so a concurrent request's update is lost."""
    calls = collections.Counter()

    def perf_counter():
        me = threading.current_thread().name
        calls[me] += 1
        if me == "MainThread" and calls[me] == 2:
            # The main request is descheduled here (in the PR, after it has read
            # METRICS["seconds"]) while another worker thread runs a whole request.
            other = threading.Thread(target=m.handle_predict, args=(request(None),), name="other")
            other.start()
            other.join(timeout=2)
        return float(calls[me] - 1)  # 0.0 when a request starts, 1.0 when it ends

    monkeypatch.setattr(m, "time", types.SimpleNamespace(perf_counter=perf_counter))
    m.handle_predict(request(None))
    assert m.METRICS["requests"] == 2
    assert m.METRICS["seconds"] == 2.0


@bug(L3)
def test_L3_configure_drops_state_from_old_dependencies(m):
    """Minor 11: configure() keeps the model and structures built from the old dependencies."""
    setup_service(m, model=FakeModel(scale=1.0))
    m.handle_predict(predict_request())
    new_store = FakeStore(structures={"MOF-5": {"id": "MOF-5", "n_atoms": 50.0}})
    setup_service(m, store=new_store, model=FakeModel(scale=2.0))
    response = m.handle_predict(predict_request())
    assert response["body"]["uptake_mmol_g"] == {"MOF-5": 100.0}


@fine(L3)
def test_L3_model_loaded_once_and_shared(m):
    """Red herring: one model per process, loaded once and shared, is the intended design."""
    built = []

    def factory():
        built.append(FakeModel())
        return built[-1]

    setup_service(m, model_factory=factory)
    for _ in range(3):
        assert m.handle_predict(predict_request())["status"] == 200
    assert len(built) == 1 and built[0].loads == 1


@fine(L3)
def test_L3_metrics_endpoint_returns_a_copy(m):
    """Red herring: handle_metrics reads METRICS without the lock, but returns a snapshot copy."""
    body = m.handle_metrics({})["body"]
    body["requests"] = 999
    assert m.METRICS["requests"] == 0
