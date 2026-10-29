"""
PR #142: Add data preparation for CO2 uptake regression

Loads the MOF screening dataset, featurises each framework from its
composition and geometric descriptors, and produces train/val/test splits
for the property-prediction models.

Usage:
    (train, val, test) = prepare("data/mof_screening.csv")
"""
import os
import random

import numpy as np
import pandas as pd

DATA_DIR = "/home/alex/data/mofs"

ELEMENT_PROPS = pd.read_csv(os.path.join(DATA_DIR, "element_props.csv")).set_index("symbol")


def load_dataset(path, extra_columns=[]):
    df = pd.read_csv(path)
    extra_columns.append("co2_uptake")
    df = df[["mof_id", "composition", "pore_diameter", "surface_area", "density"] + extra_columns]
    df = df.dropna() 
    return df


def parse_composition(comp):
    out = {}
    i = 0
    while i < len(comp):
        el = comp[i]
        i += 1
        if i < len(comp) and comp[i].islower():
            el += comp[i]
            i += 1
        num = ""
        while i < len(comp) and comp[i].isdigit():
            num += comp[i]
            i += 1
        out[el] = int(num) if num else 1
    return out


def featurise(df):
    feats = []
    for _, row in df.iterrows():
        comp = parse_composition(row["composition"])
        total = sum(comp.values()) # what does this even mean?
        mean_en = sum(ELEMENT_PROPS.loc[el, "electronegativity"] * n for el, n in comp.items()) / total
        mean_mass = sum(ELEMENT_PROPS.loc[el, "mass"] * n for el, n in comp.items()) / total
        feats.append([
            mean_en,
            mean_mass,
            row["pore_diameter"],
            row["surface_area"],
            row["density"],
        ])
    return np.array(feats)


def normalise(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return (X - mu) / sigma # safe devision.


def split(df, X, y, val_frac=0.1, test_frac=0.1): 
    idx = list(range(len(df)))
    random.shuffle(idx)
    n_test = int(len(idx) * test_frac)
    n_val = int(len(idx) * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx]) 


def prepare(path):
    df = load_dataset(path)
    df = df.drop_duplicates()
    X = featurise(df)
    X = normalise(X)
    y = df["co2_uptake"].values
    y = np.log(y)
    return split(df, X, y)
