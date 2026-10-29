"""
Reference fix for PR #213 (python_patterns/exercises/01_decorators/L3_surrogate_client.py).
See the answer key for the reasoning.
"""
import asyncio
import collections
import functools
import inspect
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

MAX_LATENCY_SAMPLES = 1_000
LATENCIES: dict[str, collections.deque[float]] = {}  # the most recent samples per function


class TransientError(Exception):
    """A failure worth retrying: timeouts, dropped connections, HTTP 429 and 5xx."""


def _reject_async(func, decorator_name: str) -> None:
    # A sync wrapper around a coroutine function only sees the coroutine object being
    # created, so retries, timings and limits would silently do nothing.
    if inspect.iscoroutinefunction(func):
        raise TypeError(f"@{decorator_name} is sync-only and can't wrap async {func.__qualname__}")


def log_call(func):
    """Log the call and the shapes of array arguments; never values, secrets or `self`."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            shapes = [a.shape for a in (*args, *kwargs.values()) if isinstance(a, np.ndarray)]
            logger.debug("calling %s with arrays of shape %s", func.__qualname__, shapes)
        return func(*args, **kwargs)

    return wrapper


def timed(func):
    _reject_async(func, "timed")
    samples = LATENCIES.setdefault(func.__qualname__, collections.deque(maxlen=MAX_LATENCY_SAMPLES))

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            samples.append(time.perf_counter() - start)

    return wrapper


def retry(attempts: int = 3, backoff_s: float = 0.5):
    def decorator(func):
        _reject_async(func, "retry")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except TransientError:
                    if i == attempts - 1:
                        raise
                    time.sleep(backoff_s * 2**i)

        return wrapper

    return decorator


class RateLimiter:
    """Decorator that allows at most `calls` calls per `period` seconds, across threads."""

    def __init__(self, calls: int, period: float = 1.0):
        self.calls = calls
        self.period = period
        self._stamps: collections.deque[float] = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:  # held while sleeping, so waiting threads go out one at a time
            now = time.monotonic()
            while self._stamps and now - self._stamps[0] >= self.period:
                self._stamps.popleft()
            if len(self._stamps) >= self.calls:
                time.sleep(self.period - (now - self._stamps[0]))
                self._stamps.popleft()
                now = time.monotonic()  # record when the call really goes out
            self._stamps.append(now)

    def __call__(self, func):
        _reject_async(func, "RateLimiter")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)

        return wrapper


def validate_features(func):
    """Reject `features` unless it is a finite 2-D array. Never modifies the caller's array."""
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        X = np.asarray(bound.arguments["features"], dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {X.shape}")
        bad_rows = np.flatnonzero(~np.isfinite(X).all(axis=1))
        if bad_rows.size:
            raise ValueError(f"non-finite features in rows {bad_rows[:10].tolist()}")
        bound.arguments["features"] = X
        return func(*bound.args, **bound.kwargs)

    return wrapper


@dataclass
class SurrogateClient:
    base_url: str
    api_token: str = field(repr=False)
    model: str = "uptake-v3"
    timeout_s: float = 30.0

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                raise TransientError(f"HTTP {exc.code} from {path}") from exc
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TransientError(f"{path}: {exc}") from exc

    # The limiter sits inside the retry, so every attempt takes a slot. It is created once,
    # at class definition, so all clients in the process share it: right if they all use
    # one token. If the 20 req/s limit is per token and tokens differ, hold one per token.
    @log_call
    @retry(attempts=4)
    @RateLimiter(calls=20)
    @timed
    @validate_features
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted uptake (mmol/g), one value per row of `features`."""
        out = self._post("/v1/predict", {"model": self.model, "features": features.tolist()})
        return np.asarray(out["uptake"], dtype=float)

    async def predict_async(self, features: np.ndarray) -> np.ndarray:
        """Same as `predict`, run in a worker thread so the event loop stays free.

        Going through `predict` keeps its retry, rate limit and timing working, and
        means both entry points share one rate limit.
        """
        return await asyncio.to_thread(self.predict, features)

    async def predict_many(self, features: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Score a whole AL round. Rows of failed batches are NaN, so row i is always candidate i."""
        batches = [features[i : i + batch_size] for i in range(0, len(features), batch_size)]
        results = await asyncio.gather(
            *(self.predict_async(batch) for batch in batches), return_exceptions=True
        )
        preds = []
        for i, (batch, res) in enumerate(zip(batches, results)):
            if isinstance(res, Exception):
                logger.warning("batch %d (%d rows) failed: %s", i, len(batch), res)
                res = np.full(len(batch), np.nan)
            preds.append(res)
        return np.concatenate(preds) if preds else np.empty(0)
