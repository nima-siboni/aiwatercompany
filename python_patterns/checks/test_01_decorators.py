"""Checks for the 01_decorators drills: every planted bug fails on the PR code and passes on the fix."""
import asyncio
import contextlib
import logging
import threading
import time
import types
import urllib.error

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


class FakeClock:
    """A clock that only moves when someone sleeps on it."""

    def __init__(self, start: float = 0.0):
        self.t = start
        self.slept = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        self.slept += seconds


# --- L1: timing / registry decorators on featurisers -------------------------------------

L1 = "01_decorators/L1_featuriser_timing"


def _structure(species, a=10.0):
    return types.SimpleNamespace(lattice=np.eye(3) * a, species=list(species))


MOF = _structure(["Zn"] * 4 + ["O"] * 13 + ["C"] * 24 + ["H"] * 12)  # 53 atoms, 1000 A^3


@bug(L1)
def test_L1_registry_collapses(m):
    """Blocker 1: without functools.wraps every featuriser registers as "wrapper"."""
    assert sorted(m.FEATURISERS) == ["metal_fraction", "n_elements", "volume_per_atom"]


@bug(L1)
def test_L1_return_value_dropped(m):
    """Blocker 2: the wrapper drops the return value, and dtype=float turns None into NaN."""
    assert m.volume_per_atom(MOF) == pytest.approx(1000 / 53)
    _, X = m.featurise([MOF, MOF])
    assert np.isfinite(X).all()


@bug(L1, raises=TypeError)
def test_L1_kwargs_not_forwarded(m):
    """Major 3: the wrapper doesn't accept or forward keyword arguments."""
    assert m.volume_per_atom(MOF, include_hydrogen=False) == pytest.approx(1000 / 41)


@bug(L1)
def test_L1_wall_clock_timing(m, monkeypatch):
    """Minor 4: time.time() is a wall clock, so a backwards NTP step records a negative time."""
    wall = iter([1000.0, 990.0])  # stepped back 10 s during the call
    perf = iter([5.0, 5.002])
    monkeypatch.setattr(time, "time", lambda: next(wall, 990.0))
    monkeypatch.setattr(time, "perf_counter", lambda: next(perf, 5.002))
    m.n_elements(MOF)
    assert m.TIMINGS["n_elements"] >= 0


@bug(L1)
def test_L1_timing_lost_on_exception(m):
    """Minor 5: no try/finally, so a featuriser that raises is never timed."""
    with contextlib.suppress(ZeroDivisionError):
        m.metal_fraction(_structure([]))
    assert "metal_fraction" in m.TIMINGS


@fine(L1)
def test_L1_registry_entries_are_timed(m):
    """Red herring: @register above @timed is the right order; whatever is registered is timed."""
    for featuriser in m.FEATURISERS.values():
        featuriser(MOF)
    assert len(m.TIMINGS) == len(m.FEATURISERS) > 0


# --- L2: retry decorator factory around sim-api calls ------------------------------------

L2 = "01_decorators/L2_retry_backoff"


@pytest.fixture
def sleeps(monkeypatch):
    """Record time.sleep durations instead of sleeping."""
    recorded = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


def _flaky(m, outcomes, **retry_kwargs):
    """A retried function that raises ConnectionError for each True in `outcomes`, then succeeds."""
    calls = []
    remaining = iter(outcomes)

    @m.retry(**retry_kwargs)
    def call():
        calls.append(1)
        if next(remaining, False):
            raise ConnectionError("connection reset by peer")
        return "ok"

    return call, calls


def _http_error(path, code, msg):
    return urllib.error.HTTPError(f"http://sim-api.internal:8080{path}", code, msg, None, None)


@bug(L2)
def test_L2_gives_up_by_returning_none(m, sleeps):
    """Blocker 1: when every attempt fails, the wrapper falls off the loop and returns None."""
    call, _ = _flaky(m, [True] * 10, max_attempts=3)
    assert _raised(ConnectionError, call)


@bug(L2)
def test_L2_attempt_count_off_by_one(m, sleeps):
    """Blocker 1: range(1, max_attempts) makes one attempt too few, and none for max_attempts=1."""
    call, calls = _flaky(m, [], max_attempts=1)
    assert call() == "ok" and len(calls) == 1
    call, calls = _flaky(m, [True] * 10, max_attempts=3)
    _raised(ConnectionError, call)
    assert len(calls) == 3


@bug(L2)
def test_L2_bare_retry_returns_function(m, monkeypatch):
    """Blocker 2: bare @retry on get_job_status returns a function, never a status."""
    monkeypatch.setattr(m, "_request", lambda method, path, payload=None: {"status": "COMPLETED"})
    assert m.get_job_status("job-1") == "COMPLETED"


@bug(L2)
def test_L2_backoff_state_leaks_between_calls(m, sleeps):
    """Major 3: nonlocal delay keeps doubling across calls, so later calls start with long sleeps."""
    call, _ = _flaky(m, [True, False, True, False], max_attempts=4, base_delay=1.0)
    call()
    call()
    assert len(sleeps) == 2 and sleeps[1] <= 1.0


@bug(L2)
def test_L2_client_errors_are_retried(m, monkeypatch, sleeps):
    """Major 4: a 404 is retried like a transient error (and then swallowed)."""
    calls = []

    def fake_request(method, path, payload=None):
        calls.append(path)
        raise _http_error(path, 404, "Not Found")

    monkeypatch.setattr(m, "_request", fake_request)
    assert _raised(urllib.error.HTTPError, m.get_result, "job-404")
    assert len(calls) == 1


@bug(L2)
def test_L2_submit_retry_creates_duplicate_jobs(m, monkeypatch, sleeps):
    """Major 5: retrying the non-idempotent POST queues the relaxation twice."""
    server_jobs = {}  # what sim-api has queued, by idempotency key
    attempts = []

    def fake_request(method, path, payload=None):
        attempts.append(payload)
        key = payload.get("idempotency_key") or f"no-key-{len(attempts)}"
        server_jobs.setdefault(key, f"job-{len(server_jobs) + 1}")
        if len(attempts) == 1:
            raise TimeoutError("timed out")  # sim-api queued the job; the client never heard back
        return {"job_id": server_jobs[key]}

    monkeypatch.setattr(m, "_request", fake_request)
    m.submit_job("MOF-5")
    assert len(server_jobs) == 1


@bug(L2)
def test_L2_failed_relaxation_is_resubmitted(m, monkeypatch, sleeps):
    """Major 6: @retry on relax resubmits a job that FAILED, then returns None."""
    submitted = []

    def fake_request(method, path, payload=None):
        if method == "POST":
            submitted.append(payload["structure_id"])
            return {"job_id": f"job-{len(submitted)}"}
        return {"status": "FAILED"}

    monkeypatch.setattr(m, "_request", fake_request)
    # Swap in a correctly decorated status call so this check isolates the workflow-level retry.
    monkeypatch.setattr(m, "get_job_status", lambda job_id: m._request("GET", f"/jobs/{job_id}")["status"])
    assert _raised(RuntimeError, m.relax, "MOF-5")
    assert len(submitted) == 1


@bug(L2)
def test_L2_sleeps_after_last_attempt(m, sleeps):
    """Minor 7: it sleeps after the final failed attempt, just before giving up."""
    call, calls = _flaky(m, [True] * 10, max_attempts=3)
    _raised(ConnectionError, call)
    assert len(sleeps) == len(calls) - 1


@bug(L2)
def test_L2_backoff_has_no_cap(m, sleeps):
    """Minor 8: the delay doubles without a cap (and without jitter)."""
    call, _ = _flaky(m, [True] * 20, max_attempts=12, base_delay=1.0)
    _raised(ConnectionError, call)
    assert max(sleeps) <= 60.0


@fine(L2)
def test_L2_get_requests_are_retried(m, monkeypatch, sleeps):
    """Red herring: retrying the idempotent GETs on a 503 is fine."""
    responses = iter([_http_error("/jobs/job-1/result", 503, "Service Unavailable"), {"energy_ev": -1234.5}])

    def fake_request(method, path, payload=None):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(m, "_request", fake_request)
    assert m.get_result("job-1") == {"energy_ev": -1234.5}


# --- L3: decorated service client (sync + async) -----------------------------------------

L3 = "01_decorators/L3_surrogate_client"
TOKEN = "sk-live-7f3a9c2e"


def _client(m, monkeypatch, *, fail_if=None, transient_failures=0):
    """A client whose HTTP call is faked: uptake = first feature of each row."""
    client = m.SurrogateClient("http://surrogate.internal", TOKEN)
    state = {"failures_left": transient_failures}

    def fake_post(path, payload):
        rows = payload["features"]
        if state["failures_left"] > 0:
            state["failures_left"] -= 1
            raise m.TransientError("HTTP 503 from /v1/predict")
        if fail_if is not None and fail_if(rows):
            raise m.TransientError("HTTP 503 from /v1/predict")
        return {"uptake": [row[0] for row in rows]}

    monkeypatch.setattr(client, "_post", fake_post)
    return client


@bug(L3)
def test_L3_token_in_logs(m, monkeypatch, caplog):
    """Blocker 1: log_call logs `self`, whose dataclass repr includes the API token."""
    caplog.set_level(logging.DEBUG)
    client = _client(m, monkeypatch)
    client.predict(np.ones((2, 3)))
    assert TOKEN not in caplog.text
    assert TOKEN not in repr(client)


@bug(L3)
def test_L3_nan_features_sent_as_zero(m, monkeypatch):
    """Blocker 2: validate_features silently turns NaN descriptors into 0 instead of rejecting them."""
    client = _client(m, monkeypatch)
    assert _raised(ValueError, client.predict, np.array([[1.0, np.nan, 3.0]]))


@bug(L3)
def test_L3_caller_array_mutated(m, monkeypatch):
    """Blocker 2: for float32 input, the NaN-to-0 replacement is written into the caller's array."""
    client = _client(m, monkeypatch)
    X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    with contextlib.suppress(ValueError):
        client.predict(X)
    assert np.isnan(X[0, 1])


@bug(L3)
def test_L3_failed_batch_shifts_predictions(m, monkeypatch):
    """Blocker 3: predict_many drops failed batches, so predictions shift onto the wrong candidates."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    client = _client(m, monkeypatch, fail_if=lambda rows: rows[0][0] == 4.0)  # the 2nd batch always fails
    X = np.arange(12, dtype=float).reshape(6, 2)  # uptake = first column: 0, 2, 4, 6, 8, 10
    preds = asyncio.run(client.predict_many(X, batch_size=2))
    assert len(preds) == len(X)
    ok = ~np.isnan(preds)
    np.testing.assert_array_equal(preds[ok], X[ok, 0])


@bug(L3)
def test_L3_async_retry_never_retries(m, monkeypatch):
    """Major 4: sync @retry on an async method returns the coroutine, so nothing is retried."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    client = _client(m, monkeypatch, transient_failures=1)
    try:
        preds = asyncio.run(client.predict_async(np.ones((2, 3))))
    except m.TransientError:
        raise AssertionError("one transient failure was not retried")
    assert len(preds) == 2


@bug(L3)
def test_L3_async_latency_is_zero(m, monkeypatch):
    """Major 4: sync @timed on an async method times coroutine creation, not the request."""
    clock = FakeClock(start=100.0)
    monkeypatch.setattr(time, "perf_counter", clock.now)
    client = m.SurrogateClient("http://surrogate.internal", TOKEN)

    def slow_post(path, payload):
        clock.sleep(0.5)  # the request takes 0.5 s
        return {"uptake": [0.0] * len(payload["features"])}

    monkeypatch.setattr(client, "_post", slow_post)
    asyncio.run(client.predict_async(np.ones((2, 3))))
    recorded = [t for samples in m.LATENCIES.values() for t in samples]
    assert recorded and max(recorded) >= 0.5


@bug(L3)
def test_L3_limiter_blocks_event_loop(m, monkeypatch):
    """Major 5: on predict_async the limiter sleeps on the event-loop thread (and only throttles coroutine creation)."""
    loop_thread = threading.get_ident()
    sleeping_threads = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeping_threads.append(threading.get_ident()))
    monkeypatch.setattr(time, "time", lambda: 1_000.0)  # frozen wall clock: every call lands in one window
    client = _client(m, monkeypatch)
    asyncio.run(client.predict_many(np.ones((30, 2)), batch_size=1))
    assert loop_thread not in sleeping_threads


@bug(L3)
def test_L3_limiter_records_pre_sleep_time(m, monkeypatch):
    """Major 6: the limiter stores the time before it slept, so later bursts exceed the limit."""
    clock = FakeClock()
    monkeypatch.setattr(time, "time", clock.now)
    monkeypatch.setattr(time, "monotonic", clock.now)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    sent = []

    @m.RateLimiter(calls=2, period=1.0)
    def send():
        sent.append(clock.now())

    for _ in range(6):
        send()
    # No 1-second window may contain more than 2 calls.
    assert all(later - earlier >= 1.0 for earlier, later in zip(sent, sent[2:])), sent


@bug(L3)
def test_L3_limiter_uses_wall_clock(m, monkeypatch):
    """Minor 12: a backwards wall-clock step (NTP) makes the limiter sleep for an hour."""
    steady = FakeClock()
    wall = {"offset": 3_600.0}
    monkeypatch.setattr(time, "monotonic", steady.now)
    monkeypatch.setattr(time, "time", lambda: steady.now() + wall["offset"])
    monkeypatch.setattr(time, "sleep", steady.sleep)

    @m.RateLimiter(calls=2, period=1.0)
    def send():
        pass

    send()
    send()
    wall["offset"] = 0.0  # the wall clock steps back an hour
    send()
    assert steady.slept <= 1.0


@bug(L3)
def test_L3_latencies_grow_without_bound(m):
    """Minor 13: LATENCIES keeps every sample for the life of the process."""

    @m.timed
    def ping():
        return None

    for _ in range(20_000):
        ping()
    assert all(len(samples) < 20_000 for samples in m.LATENCIES.values())


@fine(L3)
def test_L3_sync_retry_is_correct(m, monkeypatch):
    """Red herring: the retry decorator itself is fine on sync code."""
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    calls = []

    @m.retry(attempts=3)
    def always_fails():
        calls.append(1)
        raise m.TransientError("HTTP 503")

    assert _raised(m.TransientError, always_fails)
    assert len(calls) == 3 and sleeps == [0.5, 1.0]


@fine(L3)
def test_L3_features_by_keyword(m, monkeypatch):
    """Red herring: validate_features finds `features` whether it's passed by position or keyword."""
    client = _client(m, monkeypatch)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(client.predict(features=X), [1.0, 3.0])
    np.testing.assert_allclose(client.predict(X), [1.0, 3.0])
