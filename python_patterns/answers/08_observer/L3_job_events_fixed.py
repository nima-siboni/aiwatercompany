"""
Reference fix for PR #283 (python_patterns/exercises/08_observer/L3_job_events.py).
See the answer key for the reasoning.
"""
import fnmatch
import inspect
import logging
import sqlite3
import threading
import time
import weakref
from collections import defaultdict, deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobEvent:
    job_id: str
    status: str  # "queued", "running", "finished" or "failed"
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Every subscriber gets the same event, so the payload is a read-only copy.
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


Callback = Callable[[JobEvent], None]


class JobEvents:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[weakref.ref]] = defaultdict(list)
        self._lock = threading.Lock()
        self._local = threading.local()

    def subscribe(self, pattern: str, callback: Callback) -> None:
        # obj.method builds a new bound-method object on every access, so a plain weakref to
        # it is dead at once. WeakMethod tracks the instance and the function instead.
        ref = weakref.WeakMethod(callback) if inspect.ismethod(callback) else weakref.ref(callback)
        with self._lock:
            self._subscribers[pattern].append(ref)

    def unsubscribe(self, pattern: str, callback: Callback) -> None:
        with self._lock:
            refs = self._subscribers.get(pattern, [])
            self._subscribers[pattern] = [r for r in refs if r() is not None and r() != callback]

    def emit(self, event: JobEvent) -> None:
        pending = getattr(self._local, "pending", None)
        if pending is not None:  # emitted from inside a callback: deliver after the current event
            pending.append(event)
            return
        self._local.pending = pending = deque([event])
        try:
            while pending:
                self._deliver(pending.popleft())
        finally:
            self._local.pending = None

    def _deliver(self, event: JobEvent) -> None:
        with self._lock:  # snapshot under the lock, call outside it
            targets = []
            for pattern, refs in self._subscribers.items():
                refs[:] = [r for r in refs if r() is not None]
                if fnmatch.fnmatch(event.status, pattern):
                    targets.extend(refs)
        for ref in targets:
            callback = ref()
            if callback is None:
                continue
            try:
                callback(event)
            except Exception:
                logger.exception("subscriber %r failed on %r for job %s", callback, event.status, event.job_id)


class JobRunner:
    """Runs jobs on worker threads and reports every status change on `events`."""

    def __init__(self, run_fn: Callable[[str, dict], dict], events: JobEvents, n_workers: int = 8):
        self.run_fn = run_fn
        self.events = events
        self.pool = ThreadPoolExecutor(n_workers)

    def submit(self, job_id: str, params: dict) -> None:
        self.events.emit(JobEvent(job_id, "queued", params))
        future = self.pool.submit(self._run, job_id, params)
        future.add_done_callback(partial(self._report_crash, job_id, params))

    def _run(self, job_id: str, params: dict) -> None:
        self.events.emit(JobEvent(job_id, "running", params))
        try:
            result = self.run_fn(job_id, params)
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            self.events.emit(JobEvent(job_id, "failed", {"error": repr(exc), "params": params}))
            return
        self.events.emit(JobEvent(job_id, "finished", result))

    def _report_crash(self, job_id: str, params: dict, future: Future) -> None:
        exc = None if future.cancelled() else future.exception()
        if exc is not None:
            logger.error("job %s crashed in the runner", job_id, exc_info=exc)
            self.events.emit(JobEvent(job_id, "failed", {"error": repr(exc), "params": params}))

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)


class ResultsStore:
    """Saves every finished job to SQLite."""

    def __init__(self, events: JobEvents, path: str = "results.db"):
        # Callbacks run on whichever worker thread emits, not on the thread that built us.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS results (job_id TEXT PRIMARY KEY, uptake REAL, raw_output TEXT)"
        )
        events.subscribe("finished", self)

    def __call__(self, event: JobEvent) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO results VALUES (?, ?, ?)",
                (event.job_id, event.payload["uptake"], event.payload.get("raw_output")),
            )


class ProgressLog:
    """Most recent events for the dashboard, minus the bulky simulator output."""

    def __init__(self, events: JobEvents, maxlen: int = 1000):
        self.entries: deque[JobEvent] = deque(maxlen=maxlen)
        events.subscribe("*", self)

    def __call__(self, event: JobEvent) -> None:
        slim = {k: v for k, v in event.payload.items() if k != "raw_output"}
        self.entries.append(replace(event, payload=slim))


class RetryOnFailure:
    """Resubmits a failed job, up to `max_retries` times."""

    def __init__(self, runner: JobRunner, max_retries: int = 2):
        self.runner = runner
        self.max_retries = max_retries
        self.attempts: dict[str, int] = defaultdict(int)
        runner.events.subscribe("failed", self)

    def __call__(self, event: JobEvent) -> None:
        if self.attempts[event.job_id] < self.max_retries:
            self.attempts[event.job_id] += 1
            self.runner.submit(event.job_id, dict(event.payload["params"]))


class RetrainTrigger:
    """Starts a surrogate retraining round after every `every` finished jobs."""

    def __init__(self, events: JobEvents, retrain: Callable[[], None], every: int = 50):
        self.retrain = retrain
        self.every = every
        self.n_new = 0
        self._lock = threading.Lock()
        # Retraining takes minutes: run it off the job workers, one round at a time.
        self._executor = ThreadPoolExecutor(max_workers=1)
        events.subscribe("finished", self.on_finished)

    def on_finished(self, event: JobEvent) -> None:
        with self._lock:
            self.n_new += 1
            if self.n_new < self.every:
                return
            self.n_new = 0
        self._executor.submit(self.retrain).add_done_callback(_log_if_failed)


def _log_if_failed(future: Future) -> None:
    if not future.cancelled() and future.exception() is not None:
        logger.error("retraining failed", exc_info=future.exception())
