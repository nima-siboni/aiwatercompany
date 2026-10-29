"""
Reference fix for PR #223 (python_patterns/exercises/02_caching/L3_prediction_cache.py).
See the answer key for the reasoning.
"""
import hashlib
import logging
import os
import pickle
import tempfile
import threading
from collections import Counter, OrderedDict
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

_MISSING = object()


class Model(Protocol):
    version: str  # id of the training run that produced the weights

    def predict(self, X: np.ndarray, temperature: float, pressure: float) -> np.ndarray:
        """An (n, 2) array of [mean, std] for the n rows of X."""


class PredictionCache:
    """Thread-safe least-recently-used cache. Stores and returns copies, so callers can't edit it."""

    def __init__(self, max_entries: int = 200_000):
        self.max_entries = max_entries
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.key_hits: Counter[str] = Counter()  # only for keys still in the cache

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def get(self, key: str, default=None):
        """One atomic lookup: a copy of the cached value, or `default`."""
        with self._lock:
            value = self._data.get(key, _MISSING)
            if value is _MISSING:
                self.misses += 1
                return default
            self._data.move_to_end(key)  # this is what makes it LRU rather than FIFO
            self.hits += 1
            self.key_hits[key] += 1
        return value.copy()

    def __getitem__(self, key: str) -> np.ndarray:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            self._data[key] = np.array(value, copy=True)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                evicted, _ = self._data.popitem(last=False)
                self.key_hits.pop(evicted, None)

    def stats(self, top: int = 10) -> dict:
        with self._lock:
            lookups = self.hits + self.misses
            return {
                "entries": len(self._data),
                "hit_rate": self.hits / lookups if lookups else 0.0,
                "hottest": self.key_hits.most_common(top),
            }

    def save(self, path: str) -> None:
        """Write a temp file and rename it, so a crash never leaves a half-written cache."""
        with self._lock:
            snapshot = OrderedDict(self._data)
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(snapshot, f)
            os.replace(tmp_path, path)
        except BaseException:
            os.unlink(tmp_path)
            raise

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            entries = pickle.load(f)
        for key, value in entries.items():
            self[key] = value


def make_key(features: np.ndarray, temperature: float, pressure: float) -> str:
    """Same in every process: SHA-256 of the shape, the float64 bytes and the conditions."""
    x = np.ascontiguousarray(features, dtype=np.float64)
    digest = hashlib.sha256(f"{x.shape}|{float(temperature)!r}|{float(pressure)!r}|".encode())
    digest.update(x.tobytes())
    return digest.hexdigest()


class SurrogateService:
    def __init__(
        self,
        load_model,
        model_path: str,
        cache: PredictionCache | None = None,
        warm_start_path: str | None = None,
    ):
        self._load_model = load_model
        self.model: Model = load_model(model_path)
        self.cache = cache if cache is not None else PredictionCache()  # an empty cache is falsy
        self.warm_start_path = warm_start_path
        if warm_start_path and os.path.exists(warm_start_path):
            try:
                self.cache.load(warm_start_path)
                logger.info("warm start with %d cached predictions", len(self.cache))
            except Exception:
                logger.exception("ignoring unreadable warm-start file %s", warm_start_path)

    def shutdown(self) -> None:
        """Called on SIGTERM: keep the cache for the next start."""
        if self.warm_start_path:
            self.cache.save(self.warm_start_path)

    def reload_model(self, model_path: str) -> None:
        """Swap in a retrained model. Old entries stop matching because keys include the version."""
        self.model = self._load_model(model_path)
        logger.info("loaded model %s from %s", self.model.version, model_path)

    @staticmethod
    def _key(model: Model, features: np.ndarray, temperature: float, pressure: float) -> str:
        return f"{model.version}:{make_key(features, temperature, pressure)}"

    def predict_one(self, features: np.ndarray, temperature: float, pressure: float) -> np.ndarray | None:
        """[mean, std] for one structure, or None if the model failed."""
        model = self.model  # read once, so the key and the prediction use the same model
        key = self._key(model, features, temperature, pressure)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        try:
            pred = np.asarray(model.predict(features[None, :], temperature, pressure)[0], dtype=float)
        except Exception:
            logger.exception("prediction failed")
            return None  # not cached, so the next request tries again
        self.cache[key] = pred
        return pred

    def predict_batch(self, X: np.ndarray, temperature: float, pressure: float) -> list[np.ndarray | None]:
        """One model call for all the cache misses in X."""
        model = self.model
        keys = [self._key(model, x, temperature, pressure) for x in X]
        out = [self.cache.get(key) for key in keys]
        missing = [i for i, pred in enumerate(out) if pred is None]
        if missing:
            try:
                preds = np.asarray(model.predict(X[missing], temperature, pressure), dtype=float)
            except Exception:
                logger.exception("prediction failed for %d structures", len(missing))
                return out  # the misses stay None and are not cached
            for i, pred in zip(missing, preds):
                self.cache[keys[i]] = pred
                out[i] = pred
        return out

    def predict_ucb(self, features: np.ndarray, temperature: float, pressure: float, beta: float = 2.0) -> float | None:
        """Upper confidence bound used by the AL acquisition step."""
        pred = self.predict_one(features, temperature, pressure)
        if pred is None:
            return None
        return float(pred[0] + beta * pred[1])
