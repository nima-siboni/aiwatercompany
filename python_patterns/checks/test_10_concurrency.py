"""Checks for pattern 10: worker pools, producer-consumer queues and asyncio.

Run: uv run --with pytest --with numpy pytest python_patterns/checks/test_10_concurrency.py -v
"""
import asyncio
import collections
import io
import itertools
import logging
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from harness import bug, fine

L1 = "10_concurrency/L1_parallel_featurise"
L2 = "10_concurrency/L2_dft_queue"
L3 = "10_concurrency/L3_async_predict_client"


# ------------------------------------------------------------------------- L1: thread pool

STRUCTURES = {"MOF-A": "a", "MOF-B": "bb", "MOF-C": "ccc"}


def length_features(cif):
    return np.array([float(len(cif))])


class FakeCache:
    def __init__(self, fail_on=()):
        self.items = {}
        self.fail_on = set(fail_on)
        self._changed = threading.Condition()

    def put(self, structure_id, features):
        if structure_id in self.fail_on:
            raise OSError("feature cache unavailable")
        with self._changed:
            self.items[structure_id] = features
            self._changed.notify_all()

    def wait_for(self, n, timeout=2.0):
        with self._changed:
            return self._changed.wait_for(lambda: len(self.items) >= n, timeout)


@pytest.fixture
def use_threads(m, monkeypatch):
    """The reference fix uses a process pool. These fakes can't be pickled, so use threads."""
    monkeypatch.setattr(m, "ProcessPoolExecutor", ThreadPoolExecutor, raising=False)


@bug(L1)
def test_L1_results_attributed_to_right_ids(m, use_threads):
    """Blocker 1: zip(ids, as_completed(...)) pairs IDs with results in completion order."""
    cache = FakeCache()

    def featurise(cif):
        # MOF-A finishes only once one result has been handled (cached), MOF-C only
        # once two have, so the completion order is B, A, C.
        if cif == "a":
            cache.wait_for(1)
        elif cif == "ccc":
            cache.wait_for(2)
        return length_features(cif)

    features = m.featurise_all(STRUCTURES, featurise, cache, n_workers=4)
    assert {sid: vec[0] for sid, vec in features.items()} == {"MOF-A": 1.0, "MOF-B": 2.0, "MOF-C": 3.0}


@bug(L1)
def test_L1_cache_errors_are_reported(m, use_threads, caplog):
    """Major 3: cache writes are fire-and-forget, so their exceptions are never seen."""
    with caplog.at_level(logging.WARNING):
        m.featurise_all(STRUCTURES, length_features, FakeCache(fail_on={"MOF-B"}), n_workers=2)
    assert any("MOF-B" in record.getMessage() for record in caplog.records)


@bug(L1)
def test_L1_cache_filled_while_featurising(m, use_threads):
    """Minor 4: cache writes queue behind every remaining featurisation task."""
    cache = FakeCache()
    first_cached_before_last_ran = []

    def featurise(cif):
        if cif == "ccc":
            first_cached_before_last_ran.append(cache.wait_for(1, timeout=0.3))
        return length_features(cif)

    m.featurise_all(STRUCTURES, featurise, cache, n_workers=1)
    assert first_cached_before_last_ran == [True]


@bug(L1)
def test_L1_failure_logged_with_traceback(m, use_threads, caplog):
    """Minor 5: the failure warning drops the exception, so nobody can see why it failed."""

    def featurise(cif):
        if cif == "bb":
            raise ValueError("unparseable _cell_length_a")
        return length_features(cif)

    with caplog.at_level(logging.WARNING):
        m.featurise_all(STRUCTURES, featurise, FakeCache(), n_workers=2)
    records = [r for r in caplog.records if "MOF-B" in r.getMessage()]
    assert records and records[0].exc_info is not None


@fine(L1)
def test_L1_failed_list_appended_from_threads(m, use_threads, caplog):
    """Red herring: failed.append runs in worker threads, and list.append is thread-safe in CPython."""
    structures = {f"MOF-{i}": "x" * (i % 2) for i in range(200)}  # even IDs get an empty CIF

    def featurise(cif):
        if not cif:
            raise ValueError("empty CIF")
        return length_features(cif)

    with caplog.at_level(logging.INFO):
        features = m.featurise_all(structures, featurise, FakeCache(), n_workers=8)
    assert len(features) == 100
    assert any("100 failed" in record.getMessage() for record in caplog.records)


# ------------------------------------------------------------- L2: producer-consumer queue


class FakeCluster:
    """Stand-in for the cluster client. Jobs finish at once unless a hook says otherwise."""

    def __init__(self, energies=None, fail=(), on_wait=None, on_upload=None):
        self.energies = energies or {}
        self.fail = set(fail)
        self.on_wait, self.on_upload = on_wait, on_upload
        self.timeouts = []
        self._job_ids = itertools.count()

    def submit(self, cif, functional):
        if cif in self.fail:
            raise RuntimeError("SCF did not converge")
        return (next(self._job_ids), cif)

    def wait(self, job_id, timeout=None):
        self.timeouts.append(timeout)
        if self.on_wait:
            self.on_wait(job_id)
        return {"energy_ev": self.energies.get(job_id[1], -1.0)}

    def upload_artifacts(self, job_id):
        if self.on_upload:
            self.on_upload(job_id)


def candidate(i, formula=None):
    return {"id": f"MOF-{i}", "formula": formula or f"Zn{i}O", "cif": f"cif-{i}"}


def run_in_thread(fn, *args, timeout):
    """Call fn in a daemon thread. Returns (outcome, finished) and never hangs the suite."""
    outcome = {}

    def target():
        try:
            outcome["value"] = fn(*args)
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    return outcome, not thread.is_alive()


def threads_left_running(before, timeout=0.5):
    """Threads started since `before` that are still alive after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        extra = [t for t in threading.enumerate() if t not in before and t.is_alive()]
        if not extra or time.monotonic() > deadline:
            return extra
        time.sleep(0.01)


@bug(L2)
def test_L2_energies_keyed_by_unique_id(m):
    """Blocker 1: energies are keyed by formula, so polymorphs overwrite each other."""
    cluster = FakeCluster(energies={"cif-1": -101.0, "cif-2": -99.5})
    polymorphs = [candidate(1, "Zn4O13C24H12"), candidate(2, "Zn4O13C24H12")]
    outcome, finished = run_in_thread(m.DFTCampaign(cluster, n_workers=2).run, polymorphs, timeout=2)
    assert finished
    assert sorted(outcome["value"].values()) == [-101.0, -99.5]


@bug(L2)
def test_L2_failed_job_does_not_hang_run(m):
    """Blocker 2: a failed job skips task_done(), so tasks.join() never returns."""
    campaign = m.DFTCampaign(FakeCluster(fail={"cif-2"}), n_workers=2)
    outcome, finished = run_in_thread(
        campaign.run, [candidate(1), candidate(2), candidate(3)], timeout=0.5
    )
    assert finished, "run() is still blocked in tasks.join()"
    assert campaign.failures == ["MOF-2"]
    assert sorted(outcome["value"]) == ["MOF-1", "MOF-3"]


@bug(L2)
def test_L2_every_worker_stops(m):
    """Major 3: one _STOP for n workers, so n - 1 workers stay blocked on get() forever."""
    before = set(threading.enumerate())
    campaign = m.DFTCampaign(FakeCluster(), n_workers=4)
    _, finished = run_in_thread(campaign.run, [candidate(1)], timeout=2)
    assert finished
    leftover = threads_left_running(before)
    try:
        assert not leftover, f"{len(leftover)} worker threads never exited"
    finally:
        for _ in leftover:
            campaign.tasks.put(m._STOP)


@bug(L2)
def test_L2_upload_outside_lock(m):
    """Major 4: upload_artifacts runs under _lock, so uploads from different workers can't overlap."""
    barrier = threading.Barrier(2, timeout=0.5)
    overlapped = []

    def on_upload(job_id):
        try:
            barrier.wait()
            overlapped.append(True)
        except threading.BrokenBarrierError:
            overlapped.append(False)

    campaign = m.DFTCampaign(FakeCluster(on_upload=on_upload), n_workers=2)
    _, finished = run_in_thread(campaign.run, [candidate(1), candidate(2)], timeout=3)
    assert finished
    assert overlapped == [True, True]


@bug(L2)
def test_L2_queue_applies_backpressure(m):
    """Major 5: the queue is unbounded, so the producer drains the whole stream into memory."""
    campaign = m.DFTCampaign(FakeCluster(), n_workers=1)
    exhausted, first_wait = threading.Event(), threading.Event()
    queue_sizes = []

    def stream():
        for i in range(100):
            queue_sizes.append(campaign.tasks.qsize())
            yield candidate(i)
        exhausted.set()

    def on_wait(job_id):
        if not first_wait.is_set():
            first_wait.set()
            exhausted.wait(timeout=0.3)  # the only worker is busy with a long DFT job

    campaign.client.on_wait = on_wait
    _, finished = run_in_thread(campaign.run, stream(), timeout=3)
    assert finished
    assert max(queue_sizes) <= 2 * campaign.n_workers


@bug(L2)
def test_L2_producer_error_stops_workers(m):
    """Minor 6: if the candidate stream raises, no sentinel is sent and the workers never stop."""
    before = set(threading.enumerate())
    campaign = m.DFTCampaign(FakeCluster(), n_workers=2)

    def stream():
        yield candidate(1)
        raise ConnectionError("generative model service went away")

    with pytest.raises(ConnectionError):
        campaign.run(stream())
    leftover = threads_left_running(before)
    try:
        assert not leftover, f"{len(leftover)} worker threads never exited"
    finally:
        for _ in leftover:
            campaign.tasks.put(m._STOP)


@bug(L2)
def test_L2_wait_has_timeout(m):
    """Minor 7: client.wait() is called without a timeout, so a stuck job blocks its worker forever."""
    cluster = FakeCluster()
    _, finished = run_in_thread(m.DFTCampaign(cluster, n_workers=1).run, [candidate(1)], timeout=2)
    assert finished
    assert cluster.timeouts and None not in cluster.timeouts


@fine(L2)
def test_L2_threads_overlap_cluster_waits(m):
    """Red herring: threads suit this I/O-bound work; the workers' waits on the cluster overlap."""
    barrier = threading.Barrier(4, timeout=2)
    cluster = FakeCluster(on_wait=lambda job_id: barrier.wait())
    outcome, finished = run_in_thread(
        m.DFTCampaign(cluster, n_workers=4).run, [candidate(i) for i in range(4)], timeout=3
    )
    assert finished and len(outcome["value"]) == 4


@fine(L2)
def test_L2_failures_appended_from_threads(m):
    """Red herring: failures.append runs in several worker threads; list.append is thread-safe.

    (On the PR, run() then hangs because of Blocker 2; this only checks the list.)
    """
    failing = {f"cif-{i}" for i in range(0, 40, 2)}
    campaign = m.DFTCampaign(FakeCluster(fail=failing), n_workers=4)
    run_in_thread(campaign.run, [candidate(i) for i in range(40)], timeout=0.5)
    assert sorted(campaign.failures) == sorted(f"MOF-{i}" for i in range(0, 40, 2))


# ------------------------------------------------------------------- L3: asyncio fan-out


class FakeAPI:
    """Async stand-in for the HTTP transport, with knobs for delays and failures."""

    def __init__(self, m, delays=None, transient=(), transient_once=(), fail=(), hang=(), save_s=0.0):
        self.m = m
        self.delays, self.save_s = delays or {}, save_s
        self.transient, self.transient_once = set(transient), set(transient_once)
        self.fail, self.hang = set(fail), set(hang)
        self.calls = collections.Counter()
        self.saved = []
        self.version_calls = 0
        self.in_flight = self.max_in_flight = 0

    async def __call__(self, method, path, payload):
        if path.startswith("/models/"):
            self.version_calls += 1
            return {"version": "v7"}
        if path == "/results":
            await asyncio.sleep(self.save_s)
            self.saved.append(payload["candidate_id"])
            return {}
        cif = payload["cif"]
        self.calls[cif] += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delays.get(cif, 0))
            if cif in self.hang:
                await asyncio.Event().wait()  # never set
            if cif in self.transient or (cif in self.transient_once and self.calls[cif] == 1):
                raise self.m.TransientError("503 Service Unavailable")
            if cif in self.fail:
                raise ValueError("400 Bad Request: could not parse CIF")
            return {"uptake_mmol_g": 1.0}
        finally:
            self.in_flight -= 1


def cand(i):
    return {"id": f"MOF-{i}", "cif": f"cif-{i}"}


def on_event_loop_thread():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@pytest.fixture
def blocking_calls(monkeypatch):
    """Stub the blocking calls the PR makes and record whether each ran on the event-loop thread."""
    calls = {"urlopen": [], "sleep": []}

    def urlopen(url, timeout=None):
        calls["urlopen"].append(on_event_loop_thread())
        return io.BytesIO(b'{"version": "v7"}')

    def sleep(seconds):
        calls["sleep"].append(on_event_loop_thread())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", sleep)
    return calls


@bug(L3)
def test_L3_results_in_input_order(m, blocking_calls):
    """Blocker 1: predictions come back in completion order, not in input order."""
    api = FakeAPI(m, delays={"cif-0": 0.004, "cif-1": 0.002, "cif-2": 0.0})
    predictions = m.screen([cand(0), cand(1), cand(2)], api)
    assert [p.candidate_id for p in predictions] == ["MOF-0", "MOF-1", "MOF-2"]


@bug(L3)
def test_L3_saves_are_awaited(m, blocking_calls):
    """Blocker 2: fire-and-forget save tasks that are still pending get cancelled by asyncio.run."""
    api = FakeAPI(m, save_s=0.01)  # a save takes 10 ms, like a real network write
    candidates = [cand(i) for i in range(5)]
    m.screen(candidates, api)
    assert sorted(api.saved) == [c["id"] for c in candidates]


@bug(L3)
def test_L3_exhausted_retries_are_not_zero_uptake(m, blocking_calls, monkeypatch):
    """Blocker 3: once the retries run out, the failed prediction comes back as 0.0 mmol/g."""
    monkeypatch.setattr(m, "RETRY_DELAY_S", 0.0)
    api = FakeAPI(m, transient={"cif-1"})
    by_id = {p.candidate_id: p for p in m.screen([cand(0), cand(1)], api)}
    assert by_id["MOF-1"].uptake_mmol_g is None
    assert by_id["MOF-0"].uptake_mmol_g == 1.0


@bug(L3)
def test_L3_no_blocking_urlopen_on_event_loop(m, blocking_calls):
    """Major 4: _model_version calls urllib's blocking urlopen on the event-loop thread."""
    m.screen([cand(0), cand(1)], FakeAPI(m))
    assert not any(blocking_calls["urlopen"])


@bug(L3)
def test_L3_retry_does_not_block_event_loop(m, blocking_calls, monkeypatch):
    """Major 5: the retry loop calls time.sleep, which freezes every in-flight request."""
    monkeypatch.setattr(m, "RETRY_DELAY_S", 0.0)
    m.screen([cand(0)], FakeAPI(m, transient_once={"cif-0"}))
    assert not any(blocking_calls["sleep"])


@bug(L3)
def test_L3_concurrency_is_bounded(m, blocking_calls):
    """Major 6: every candidate is in flight at once; nothing bounds the fan-out."""
    api = FakeAPI(m)
    m.screen([cand(i) for i in range(200)], api)
    assert api.max_in_flight <= 64


@bug(L3, raises=ValueError)
def test_L3_one_failure_keeps_other_results(m, blocking_calls):
    """Major 7: the first non-transient error propagates and discards every other prediction."""
    api = FakeAPI(m, fail={"cif-1"})
    by_id = {p.candidate_id: p for p in m.screen([cand(i) for i in range(4)], api)}
    assert by_id["MOF-1"].uptake_mmol_g is None
    assert [by_id[f"MOF-{i}"].uptake_mmol_g for i in (0, 2, 3)] == [1.0, 1.0, 1.0]


@bug(L3)
def test_L3_deadline_cancels_and_reports_unfinished(m, blocking_calls):
    """Major 8: at the deadline, unfinished predictions are silently dropped and left running."""
    api = FakeAPI(m, hang={"cif-1"})

    async def main():
        predictions = await m.PredictionClient(api).predict_many(
            [cand(0), cand(1), cand(2)], deadline_s=0.05
        )
        return predictions, api.in_flight

    predictions, still_running = asyncio.run(main())
    assert still_running == 0, "unfinished requests keep running after predict_many returns"
    assert [p.candidate_id for p in predictions] == ["MOF-0", "MOF-1", "MOF-2"]
    assert predictions[1].uptake_mmol_g is None


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
@bug(L3, raises=RuntimeError)
def test_L3_screen_from_a_running_loop(m, blocking_calls):
    """Major 9: screen() calls asyncio.run, which raises inside Jupyter's running event loop."""
    api = FakeAPI(m)

    async def notebook_cell():
        return m.screen([cand(0)], api)

    predictions = asyncio.run(notebook_cell())
    assert [p.candidate_id for p in predictions] == ["MOF-0"]


@bug(L3)
def test_L3_model_version_fetched_once(m, blocking_calls):
    """Minor 10: the model version is fetched once per candidate instead of once per run."""
    api = FakeAPI(m)
    m.screen([cand(i) for i in range(5)], api)
    assert len(blocking_calls["urlopen"]) + api.version_calls == 1


@bug(L3)
def test_L3_per_request_timeout(m, blocking_calls, monkeypatch):
    """Minor 11: no per-request timeout, so a hung call is only cut off by the overall deadline."""
    monkeypatch.setattr(m, "REQUEST_TIMEOUT_S", 0.02, raising=False)
    api = FakeAPI(m, hang={"cif-0"})
    started = time.monotonic()
    predictions = m.screen([cand(0), cand(1)], api, deadline_s=0.3)
    assert time.monotonic() - started < 0.2
    assert predictions[0].uptake_mmol_g is None and predictions[1].uptake_mmol_g == 1.0


@fine(L3)
def test_L3_shared_list_append_from_tasks(m, blocking_calls):
    """Red herring: many tasks append to self.saved without a lock; on one event loop that's safe."""
    api = FakeAPI(m)
    candidates = [cand(i) for i in range(50)]

    async def main():
        client = m.PredictionClient(api)
        await client.predict_many(candidates)
        await asyncio.sleep(0.01)  # let the PR's fire-and-forget saves finish (Blocker 2)
        return client

    client = asyncio.run(main())
    assert sorted(client.saved) == sorted(c["id"] for c in candidates)


@fine(L3)
def test_L3_only_transient_errors_are_retried(m, blocking_calls, monkeypatch):
    """Red herring: retrying only TransientError is right; a 400 must not be retried."""
    monkeypatch.setattr(m, "RETRY_DELAY_S", 0.0)
    api = FakeAPI(m, fail={"cif-0"})
    try:
        m.screen([cand(0)], api)
    except ValueError:
        pass  # the PR lets the 400 escape; that's Major 7, not what this checks
    assert api.calls["cif-0"] == 1
