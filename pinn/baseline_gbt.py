"""Gradient-boosted-tree baseline for the surrogate.

Trains a gradient-boosted-tree regressor on the same design parameters and the
same train/test split as the network, predicting the failure load factor
directly. This is the standard tabular-regression baseline against which the
neural surrogate is compared.

Run:  python baseline_gbt.py
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from config import get_config
from data import load_archetype

ARCHS = ["deepBeam", "hammerhead", "multiColumnBent", "pileCap"]


def stats(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    err = pred - true
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((np.abs(err) / np.clip(true, 1e-6, None)).mean() * 100.0)
    ss_tot = ((true - true.mean()) ** 2).sum()
    r2 = float(1.0 - (err ** 2).sum() / max(ss_tot, 1e-12))
    return mape, rmse, r2


print(f"{'archetype':<18}{'MAPE %':>9}{'RMSE':>9}{'R2 (gen)':>10}{'R2 (all)':>10}")
for arch in ARCHS:
    cfg = get_config(archetype=arch)
    data = load_archetype(cfg)
    X = data.theta.numpy()
    y = data.lambda_f.numpy()
    tr = data.train_idx.numpy()
    te = data.test_idx.numpy()

    gbt = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                    learning_rate=0.05, random_state=cfg.seed)
    gbt.fit(X[tr], y[tr])
    pred, true = gbt.predict(X[te]), y[te]

    gen = true < cfg.lambda_max - 1e-3
    mape_g, rmse_g, r2_g = stats(pred[gen], true[gen])
    _, _, r2_all = stats(pred, true)
    print(f"{arch:<18}{mape_g:>9.2f}{rmse_g:>9.3f}{r2_g:>10.3f}{r2_all:>10.3f}")
