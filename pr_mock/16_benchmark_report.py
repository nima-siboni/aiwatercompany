"""
PR #276: Benchmark report for the stability model

Evaluates our GNN ensemble, which predicts the energy above the convex hull (eV/atom) of
hypothetical crystals, and its use as a stable/unstable filter in screening. Produces the
table for the quarterly model review: MAE and R2, F1 for "stable" (E_hull <= 0) at the best
threshold, calibration of the ensemble's uncertainty (coverage of central intervals), and 95%
bootstrap intervals for our MAE and for the production model's, to show whether we beat it.
"""
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from stability_gnn import GNN, featurize, train_member  # train_member(model, X, y) -> trained model


def split(df, n_folds=5):
    """Hold out whole compositions, so no test formula is seen in training."""
    return list(GroupKFold(n_splits=n_folds).split(df, groups=df["formula"]))


def train_ensemble(X, y, n_members=5):
    members = []
    for _ in range(n_members):
        torch.manual_seed(0)  # reproducible members
        members.append(train_member(GNN(), X, y))
    return members


@torch.no_grad()
def predict(members, X):
    mus, sigmas = zip(*(m(X) for m in members))  # each member predicts a mean and a std
    return torch.stack(mus).mean(0).numpy(), torch.stack(sigmas).mean(0).numpy()


def coverage(y, mu, sigma, levels=(0.5, 0.8, 0.9, 0.95)):
    """Fraction of test points inside the central interval at each confidence level."""
    return {p: float(np.mean(np.abs(y - mu) <= norm.ppf(p) * sigma)) for p in levels}


def best_threshold(y, mu):
    thresholds = np.linspace(-0.1, 0.2, 61)
    scores = [f1_score(y <= 0, mu <= t) for t in thresholds]
    return thresholds[int(np.argmax(scores))], max(scores)


def bootstrap_ci(y, pred, metric, n_boot=1000):
    stats = []
    for _ in range(n_boot):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), len(y), replace=False)
        stats.append(metric(y[idx], pred[idx]))
    return np.percentile(stats, [2.5, 97.5])


def report(df, production_pred):
    X = featurize(df["relaxed_structure"])
    y = df["e_hull"].to_numpy()
    train_idx, test_idx = split(df)[0]
    members = train_ensemble(X[train_idx], y[train_idx])
    mu, sigma = predict(members, X[test_idx])
    y_test, prod = y[test_idx], production_pred[test_idx]
    threshold, f1 = best_threshold(y_test, mu)
    return {
        "mae": mean_absolute_error(y_test, mu),
        "r2": r2_score(mu, y_test),
        "f1_stable": f1,
        "threshold": threshold,
        "coverage": coverage(y_test, mu, sigma),
        "mae_ci": bootstrap_ci(y_test, mu, mean_absolute_error),
        "production_mae_ci": bootstrap_ci(y_test, prod, mean_absolute_error),
    }
