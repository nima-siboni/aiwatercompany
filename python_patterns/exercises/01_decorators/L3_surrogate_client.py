"""
PR #213: Client for the surrogate-model service

Each active-learning (AL) round scores about 50k candidate MOFs with the
surrogate-model service, which predicts CO2 uptake (mmol/g) from a vector of
structure descriptors. This wraps the HTTP API in a client and moves the
cross-cutting concerns into decorators: call logging, latency metrics, retries
on transient errors and client-side rate limiting (the service allows 20
requests/s per API token). One client is shared by the AL worker threads, and
`predict_many` scores a whole round concurrently with asyncio.
"""
import asyncio
import functools
import inspect
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

LATENCIES: dict[str, list[float]] = {}


class TransientError(Exception):
    """A failure worth retrying: timeouts, dropped connections, HTTP 429 and 5xx."""


def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"calling {func.__qualname__} args={args} kwargs={kwargs}")
        return func(*args, **kwargs)

    return wrapper


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            LATENCIES.setdefault(func.__qualname__, []).append(time.perf_counter() - start)

    return wrapper


def retry(attempts: int = 3, backoff_s: float = 0.5):
    def decorator(func):
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
    """Decorator that allows at most `calls` calls per `period` seconds."""

    def __init__(self, calls: int, period: float = 1.0):
        self.calls = calls
        self.period = period
        self._stamps: list[float] = []

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            self._stamps = [t for t in self._stamps if now - t < self.period]
            if len(self._stamps) >= self.calls:
                time.sleep(self.period - (now - self._stamps[0]))
            self._stamps.append(now)
            return func(*args, **kwargs)

        return wrapper


def validate_features(func):
    """Make sure `features` is a 2-D float32 array before it goes over the wire."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        X = np.asarray(bound.arguments["features"], dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {X.shape}")
        bound.arguments["features"] = np.nan_to_num(X, copy=False)
        return func(*bound.args, **bound.kwargs)

    return wrapper


@dataclass
class SurrogateClient:
    base_url: str
    api_token: str
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

    @log_call
    @RateLimiter(calls=20)
    @retry(attempts=4)
    @timed
    @validate_features
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted uptake (mmol/g), one value per row of `features`."""
        out = self._post("/v1/predict", {"model": self.model, "features": features.tolist()})
        return np.asarray(out["uptake"], dtype=float)

    @log_call
    @RateLimiter(calls=20)
    @retry(attempts=4)
    @timed
    @validate_features
    async def predict_async(self, features: np.ndarray) -> np.ndarray:
        """Same as `predict`, without blocking the event loop."""
        payload = {"model": self.model, "features": features.tolist()}
        out = await asyncio.to_thread(self._post, "/v1/predict", payload)
        return np.asarray(out["uptake"], dtype=float)

    async def predict_many(self, features: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Score a whole AL round. Failed batches are logged and skipped."""
        batches = [features[i : i + batch_size] for i in range(0, len(features), batch_size)]
        results = await asyncio.gather(
            *(self.predict_async(batch) for batch in batches), return_exceptions=True
        )
        preds = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.warning("batch %d failed: %s", i, res)
                continue
            preds.append(res)
        return np.concatenate(preds)
