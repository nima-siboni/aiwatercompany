"""
Reference fix for PR #261 (python_patterns/exercises/06_strategy_template/L1_acquisition.py).
See the answer key for the reasoning.
"""
from abc import ABC, abstractmethod

import numpy as np


class Acquisition(ABC):
    """Scores candidates. Higher is better for every strategy; select() takes the top scores."""

    @abstractmethod
    def score(self, mean: np.ndarray, std: np.ndarray, cost: np.ndarray | None = None) -> np.ndarray:
        """One score per candidate. `cost` is the per-candidate simulation cost, if known."""


class Greedy(Acquisition):
    def score(self, mean, std, cost=None):
        return mean


class UCB(Acquisition):
    def __init__(self, beta: float = 2.0):
        self.beta = beta

    def score(self, mean, std, cost=None):
        return mean + self.beta * std


class TargetUptake(Acquisition):
    def __init__(self, target: float):
        self.target = target

    def score(self, mean, std, cost=None):
        return -np.abs(mean - self.target)  # closest to the target scores highest


class CostAwareUCB(UCB):
    """UCB per unit of simulation cost: bigger unit cells take longer in GCMC."""

    def score(self, mean, std, cost=None):
        if cost is None:
            raise ValueError("CostAwareUCB needs the per-candidate simulation cost")
        return super().score(mean, std) / cost


def select(acq: Acquisition, mean: np.ndarray, std: np.ndarray, batch_size: int, cost=None) -> np.ndarray:
    """Indices of the candidates to simulate this round, best first."""
    if batch_size < 0:
        raise ValueError(f"batch_size must be >= 0, got {batch_size}")
    scores = np.asarray(acq.score(mean, std, cost=cost), dtype=float)
    scores = np.where(np.isfinite(scores), scores, -np.inf)  # argsort puts NaN last, i.e. "best"
    return np.argsort(-scores, kind="stable")[:batch_size]
