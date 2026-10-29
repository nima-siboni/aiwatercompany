"""
PR #282: Callback hooks for the surrogate-model trainer

The surrogate model predicts CO2 uptake from MOF descriptors, so that only
promising candidates go to GCMC simulation. Early stopping, checkpointing and
the loss history used to be hard-wired into `fit`. This PR moves them into
Keras-style callbacks: the trainer calls each callback's hooks after every
batch and epoch with a `logs` dict of metrics, and any callback can end
training by setting `trainer.stop_training`. The active-learning loop calls
`fit` on the same trainer every round as new labels arrive, then reloads the
best checkpoint.
"""
import math

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
        self.epochs.append(logs)


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 5, mode: str = "min"):
        self.monitor, self.patience, self.mode = monitor, patience, mode
        self.best = math.inf
        self.wait = 0

    def on_epoch_end(self, trainer, epoch, logs):
        if logs[self.monitor] < self.best:
            self.best, self.wait = logs[self.monitor], 0
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

    def __init__(self, path: str):
        self.path = path
        self.best = math.inf

    def on_batch_end(self, trainer, batch, logs):
        if logs["loss"] < self.best:
            self.best = logs["loss"]
            np.save(self.path, trainer.model.weights)


class Trainer:
    def __init__(self, model: LinearSurrogate, callbacks: list[Callback] | None = None, seed: int = 0):
        self.model = model
        self.callbacks = callbacks or []
        self.rng = np.random.default_rng(seed)
        self.stop_training = False

    def fit(self, X, y, X_val, y_val, epochs: int = 100, batch_size: int = 32) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(self)
        logs: dict[str, float] = {}
        for epoch in range(epochs):
            order = self.rng.permutation(len(y))
            for batch, start in enumerate(range(0, len(y), batch_size)):
                idx = order[start:start + batch_size]
                logs["loss"] = self.model.step(X[idx], y[idx])
                for cb in self.callbacks:
                    cb.on_batch_end(self, batch, logs)
            logs["val_loss"] = self.model.loss(X_val, y_val)
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)
            if self.stop_training:
                break
