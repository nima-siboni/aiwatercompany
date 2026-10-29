"""
PR #223: Prediction cache for the surrogate service

The surrogate service answers about two million uptake predictions a day, and
most active-learning (AL) rounds re-score candidates we have already scored.
This puts an in-process LRU cache in front of the model, keyed on the feature
vector and the conditions, with hit statistics for the dashboard and a warm
start from disk after a restart. A prediction is [mean, std] of the CO2 uptake
in mmol/g; the AL loop ranks candidates by their upper confidence bound (UCB),
mean + beta * std. Request threads share one service, and the AL loop calls
`reload_model` whenever it publishes a retrained model.
"""
import logging
import os
import pickle
from collections import Counter, OrderedDict
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Model(Protocol):
    version: str  # id of the training run that produced the weights

    def predict(self, X: np.ndarray, temperature: float, pressure: float) -> np.ndarray:
        """An (n, 2) array of [mean, std] for the n rows of X."""


class PredictionCache:
    """Least-recently-used cache of model outputs."""

    def __init__(self, max_entries: int = 200_000):
        self.max_entries = max_entries
        self._data: OrderedDict[int, np.ndarray | None] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.key_hits: Counter[int] = Counter()  # for the "hottest candidates" panel

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: int) -> bool:
        return key in self._data

    def __getitem__(self, key: int) -> np.ndarray | None:
        self.hits += 1
        self.key_hits[key] += 1
        return self._data[key]

    def __setitem__(self, key: int, value: np.ndarray | None) -> None:
        self.misses += 1
        self._data[key] = value
        if len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def stats(self, top: int = 10) -> dict:
        lookups = self.hits + self.misses
        return {
            "entries": len(self._data),
            "hit_rate": self.hits / lookups if lookups else 0.0,
            "hottest": self.key_hits.most_common(top),
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._data, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._data.update(pickle.load(f))


def make_key(features: np.ndarray, temperature: float, pressure: float) -> int:
    return hash((features.tobytes(), temperature, pressure))


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
        self.cache = cache or PredictionCache()
        self.warm_start_path = warm_start_path
        if warm_start_path and os.path.exists(warm_start_path):
            self.cache.load(warm_start_path)
            logger.info("warm start with %d cached predictions", len(self.cache))

    def shutdown(self) -> None:
        """Called on SIGTERM: keep the cache for the next start."""
        if self.warm_start_path:
            self.cache.save(self.warm_start_path)

    def reload_model(self, model_path: str) -> None:
        """Swap in a retrained model without restarting the service."""
        self.model = self._load_model(model_path)
        logger.info("loaded model %s from %s", self.model.version, model_path)

    def predict_one(self, features: np.ndarray, temperature: float, pressure: float) -> np.ndarray | None:
        """[mean, std] for one structure, or None if the model failed."""
        key = make_key(features, temperature, pressure)
        if key in self.cache:
            return self.cache[key]
        try:
            pred = self.model.predict(features[None, :], temperature, pressure)[0]
        except Exception:
            logger.exception("prediction failed")
            pred = None
        self.cache[key] = pred
        return pred

    def predict_batch(self, X: np.ndarray, temperature: float, pressure: float) -> list[np.ndarray | None]:
        return [self.predict_one(x, temperature, pressure) for x in X]

    def predict_ucb(self, features: np.ndarray, temperature: float, pressure: float, beta: float = 2.0) -> float | None:
        """Upper confidence bound used by the AL acquisition step."""
        pred = self.predict_one(features, temperature, pressure)
        if pred is None:
            return None
        pred[0] += beta * pred[1]
        return float(pred[0])
