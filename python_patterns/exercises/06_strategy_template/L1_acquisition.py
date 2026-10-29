"""
PR #261: Pluggable acquisition functions for the active-learning loop

Each round the surrogate predicts the mean and standard deviation of CO2
uptake (mmol/g) for every unlabelled MOF, and an acquisition function decides
which ones go to GCMC (the adsorption simulation that labels them) next.
Campaigns can now choose greedy, UCB (optimistic: mean plus beta standard
deviations), a target window (for separations we want uptake close to a
value, not as high as possible) or cost-aware UCB. The loop calls select()
once per round with batch_size = min(96, remaining budget).
"""
import numpy as np


class Acquisition:
    """Scores candidates; the highest scores are simulated first."""

    def score(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return NotImplemented


class Greedy(Acquisition):
    def score(self, mean, std):
        return mean


class UCB(Acquisition):
    def __init__(self, beta: float = 2.0):
        self.beta = beta

    def score(self, mean, std):
        return mean + self.beta * std


class TargetUptake(Acquisition):
    def __init__(self, target: float):
        self.target = target

    def score(self, mean, std):
        return np.abs(mean - self.target)


class CostAwareUCB(UCB):
    """UCB per unit of simulation cost: bigger unit cells take longer in GCMC."""

    def score(self, mean, std, cost):
        return super().score(mean, std) / cost


def select(acq: Acquisition, mean: np.ndarray, std: np.ndarray, batch_size: int, cost=None) -> np.ndarray:
    """Indices of the candidates to simulate this round."""
    if isinstance(acq, CostAwareUCB):
        scores = acq.score(mean, std, cost)
    else:
        scores = acq.score(mean, std)
    return np.argsort(scores)[-batch_size:]
