"""
PR #283: Job-status pub/sub for the orchestration service

The orchestration service runs DFT and GCMC jobs on a pool of worker threads.
Other components need to hear about status changes: the results store saves
finished jobs, the dashboard shows recent activity, failed jobs get retried,
and the active-learning loop retrains the surrogate model once enough new
labels have arrived. This PR adds a small `JobEvents` hub. Components subscribe
a callback to a status pattern ("finished", "fail*", or "*" for everything),
and the runner emits an event at every status change. The hub holds callbacks
by weak reference, so a component that has been garbage-collected stops
receiving events instead of being kept alive by the hub.
"""
import fnmatch
import sqlite3
import time
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class JobEvent:
    job_id: str
    status: str  # "queued", "running", "finished" or "failed"
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


Callback = Callable[[JobEvent], None]


class JobEvents:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[weakref.ref]] = defaultdict(list)

    def subscribe(self, pattern: str, callback: Callback) -> None:
        self._subscribers[pattern].append(weakref.ref(callback))

    def unsubscribe(self, pattern: str, callback: Callback) -> None:
        refs = self._subscribers.get(pattern, [])
        self._subscribers[pattern] = [r for r in refs if r() != callback]

    def emit(self, event: JobEvent) -> None:
        for pattern, refs in self._subscribers.items():
            if not fnmatch.fnmatch(event.status, pattern):
                continue
            for ref in refs:
                callback = ref()
                if callback is None:
                    continue
                try:
                    callback(event)
                except Exception:
                    pass


class JobRunner:
    """Runs jobs on worker threads and reports every status change on `events`."""

    def __init__(self, run_fn: Callable[[str, dict], dict], events: JobEvents, n_workers: int = 8):
        self.run_fn = run_fn
        self.events = events
        self.pool = ThreadPoolExecutor(n_workers)

    def submit(self, job_id: str, params: dict) -> None:
        self.events.emit(JobEvent(job_id, "queued", params))
        self.pool.submit(self._run, job_id, params)

    def _run(self, job_id: str, params: dict) -> None:
        self.events.emit(JobEvent(job_id, "running", params))
        try:
            result = self.run_fn(job_id, params)
        except Exception as exc:
            self.events.emit(JobEvent(job_id, "failed", {"error": str(exc), "params": params}))
            return
        self.events.emit(JobEvent(job_id, "finished", result))

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)


class ResultsStore:
    """Saves every finished job to SQLite."""

    def __init__(self, events: JobEvents, path: str = "results.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS results (job_id TEXT PRIMARY KEY, uptake REAL, raw_output TEXT)"
        )
        events.subscribe("finished", self)

    def __call__(self, event: JobEvent) -> None:
        with self.conn:
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
        event.payload.pop("raw_output", None)
        self.entries.append(event)


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
            self.runner.submit(event.job_id, event.payload["params"])


class RetrainTrigger:
    """Starts a surrogate retraining round after every `every` finished jobs."""

    def __init__(self, events: JobEvents, retrain: Callable[[], None], every: int = 50):
        self.retrain = retrain
        self.every = every
        self.n_new = 0
        events.subscribe("finished", self.on_finished)

    def on_finished(self, event: JobEvent) -> None:
        self.n_new += 1
        if self.n_new >= self.every:
            self.n_new = 0
            self.retrain()
