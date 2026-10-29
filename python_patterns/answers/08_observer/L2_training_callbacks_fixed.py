"""
Reference fix for PR #282 (python_patterns/exercises/08_observer/L2_training_callbacks.py).
See the answer key for the reasoning.
"""
import math
import os

import numpy as np


class LinearSurrogate:
    """Linear model trained with plain SGD; stands in for the MLP in tests."""

    def __init__(self, n_features: int, lr: float = 0.01):
        self.weights = np.zeros(n_features)
        self.lr = lr

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((X @ self.weights - y) ** 2))

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        err = X @ self.weights - y
        self.weights -= self.lr * 2 * X.T @ err / len(y)
        return float(np.mean(err**2))


class Callback:
    def on_train_begin(self, trainer: "Trainer") -> None:
        pass

    def on_batch_end(self, trainer: "Trainer", batch: int, logs: dict) -> None:
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        pass


class History(Callback):
    def __init__(self):
        self.epochs: list[dict] = []

    def on_epoch_end(self, trainer, epoch, logs):
        self.epochs.append(dict(logs))


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 5, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor, self.patience, self.mode = monitor, patience, mode
        self._reset()

    def _reset(self) -> None:
        self.best = math.inf if self.mode == "min" else -math.inf
        self.wait = 0

    def on_train_begin(self, trainer):
        self._reset()

    def on_epoch_end(self, trainer, epoch, logs):
        value = logs[self.monitor]
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best, self.wait = value, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.stop_training = True


class TerminateOnNaN(Callback):
    def on_batch_end(self, trainer, batch, logs):
        if not math.isfinite(logs["loss"]):
            trainer.stop_training = True


class ModelCheckpoint(Callback):
    """Keeps the best weights so far on disk."""

    def __init__(self, path: str, monitor: str = "val_loss"):
        self.path, self.monitor = path, monitor
        self.best = math.inf

    def on_train_begin(self, trainer):
        self.best = math.inf

    def on_epoch_end(self, trainer, epoch, logs):
        if logs[self.monitor] < self.best:
            self.best = logs[self.monitor]
            # Write then rename: a crash mid-write can't destroy the previous checkpoint, and
            # the file ends up at exactly self.path (np.save appends .npy to other names).
            tmp = f"{self.path}.tmp.npy"
            np.save(tmp, trainer.model.weights)
            os.replace(tmp, self.path)


class Trainer:
    def __init__(self, model: LinearSurrogate, callbacks: list[Callback] | None = None, seed: int = 0):
        self.model = model
        self.callbacks = callbacks or []
        self.rng = np.random.default_rng(seed)
        self.stop_training = False

    def fit(self, X, y, X_val, y_val, epochs: int = 100, batch_size: int = 32) -> None:
        self.stop_training = False
        for cb in self.callbacks:
            cb.on_train_begin(self)
        for epoch in range(epochs):
            order = self.rng.permutation(len(y))
            for batch, start in enumerate(range(0, len(y), batch_size)):
                idx = order[start:start + batch_size]
                batch_logs = {"loss": self.model.step(X[idx], y[idx])}
                for cb in self.callbacks:
                    cb.on_batch_end(self, batch, batch_logs)
                if self.stop_training:
                    break
            logs = {"loss": batch_logs["loss"], "val_loss": self.model.loss(X_val, y_val)}
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)
            if self.stop_training:
                break
