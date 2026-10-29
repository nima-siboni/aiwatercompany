"""
PR #241: Batch Bayesian optimisation to choose the next GCMC runs

Each round we can afford 16 GCMC simulations. This replaces "simulate the top predicted MOFs"
with a GP + expected-improvement loop over the hypothetical-MOF library (Zeo++ descriptors),
seeded with last year's GCMC results. The objective is CO2 uptake at 0.15 bar and 298 K
(maximise), but the loop also supports minimising, for the parasitic-energy campaign.
`benchmark()` replays the loop on the fully simulated hMOF subset, where every label is known,
and compares it with random selection for the paper figure.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from gcmc import run_gcmc  # blocking; one simulation takes 2-6 hours on 32 cores

FEATURES = ["lcd", "pld", "density", "vsa", "void_fraction"]  # Å, Å, g/cm3, m2/cm3, -
MMOL_PER_CM3_STP = 22.414
BATCH = 16


def load_library(path):
    df = pd.read_csv(path)
    return df[df.zeopp_ok & (df.pld > 3.3)]  # 3.3 Å: kinetic diameter of CO2


def load_seed(path):
    seed = pd.read_csv(path)  # columns: mof_id, uptake_cm3_g
    seed["uptake"] = seed["uptake_cm3_g"] * MMOL_PER_CM3_STP
    return seed[["mof_id", "uptake"]]


def gcmc_oracle(mof_ids):
    return np.array([run_gcmc(m, pressure_bar=0.15, temperature_k=298) for m in mof_ids])


def expected_improvement(mu, sd, best, maximize=True, xi=0.01):
    sd = np.maximum(sd, 1e-9)
    imp = mu - best - xi if maximize else best - mu - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def fit_gp(X, y):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)
    return gp.fit(X, y)


def propose(gp, pool, best, maximize):
    mu, sd = gp.predict(pool[FEATURES].to_numpy(), return_std=True)
    ei = expected_improvement(mu, sd, best, maximize)
    return np.argsort(-ei)[:BATCH]


def run_campaign(pool, labelled, oracle, n_rounds=10, maximize=True):
    pool = pool[~pool.mof_id.isin(labelled.mof_id)].reset_index(drop=True)
    sign = 1.0 if maximize else -1.0
    curve = []
    for _ in range(n_rounds):
        X = labelled[FEATURES].to_numpy()
        y = sign * labelled["uptake"].to_numpy()
        gp = fit_gp(X, y)
        picks = propose(gp, pool, best=y.max(), maximize=maximize)

        batch = pool.iloc[picks].copy()
        batch["uptake"] = oracle(batch.mof_id)
        labelled = pd.concat([labelled, batch])
        pool = pool.drop(picks)
        curve.append(labelled["uptake"].max())
    return labelled, curve


def random_baseline(pool, oracle, n_rounds, rng):
    found = [oracle(rng.choice(pool.mof_id.to_numpy(), size=BATCH)).max() for _ in range(n_rounds)]
    return list(np.maximum.accumulate(found))


def benchmark(library_csv, seed_csv, lookup, n_rounds=10, seed=0):
    """lookup: dict mof_id -> uptake (mmol/g) for the fully simulated subset."""
    pool = load_library(library_csv)
    labelled = load_seed(seed_csv).merge(pool, on="mof_id")
    oracle = lambda ids: np.array([lookup[m] for m in ids])
    _, bo_curve = run_campaign(pool, labelled, oracle, n_rounds)
    rand_curve = random_baseline(pool, oracle, n_rounds, np.random.default_rng(seed))
    return bo_curve, rand_curve


if __name__ == "__main__":
    pool = load_library("hmof_library.csv")
    labelled = load_seed("gcmc_2025.csv").merge(pool, on="mof_id")
    labelled, curve = run_campaign(pool, labelled, gcmc_oracle)
    labelled.to_csv("campaign_results.csv", index=False)
