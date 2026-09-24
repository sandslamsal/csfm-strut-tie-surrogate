"""Tabular baseline panel for the surrogate.

Trains a panel of standard tabular regressors on the same design parameters,
the same per-archetype train/test split and the same target (failure load
factor) as the neural surrogate.

Baselines: ridge linear, degree-2 polynomial ridge, k-nearest-neighbours,
support-vector regression (RBF), random forest, gradient-boosted trees and
Gaussian-process regression. Inputs (data.theta) are already standardised to
zero mean / unit variance over the training split, as the kernel/distance
models require.

Run:  python baselines.py
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

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


def make_models(seed: int) -> dict:
    return {
        "Ridge (linear)": Ridge(alpha=1.0),
        "Poly-2 ridge": make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0)),
        "k-NN (k=7)": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "SVR (RBF)": SVR(C=10.0, gamma="scale", epsilon=0.01),
        "Random forest": RandomForestRegressor(
            n_estimators=400, max_depth=None, random_state=seed, n_jobs=-1),
        "Grad-boosted trees": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed),
        "Gaussian process": GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=2.0)
            + WhiteKernel(noise_level=0.05),
            normalize_y=True, n_restarts_optimizer=2, random_state=seed),
    }


def main() -> None:
    rows: dict[str, dict[str, tuple[float, float]]] = {}
    for arch in ARCHS:
        cfg = get_config(archetype=arch)
        data = load_archetype(cfg)
        X = data.theta.numpy()
        y = data.lambda_f.numpy()
        tr = data.train_idx.numpy()
        te = data.test_idx.numpy()
        gen = y[te] < cfg.lambda_max - 1e-3  # genuine-failure test designs

        for name, model in make_models(cfg.seed).items():
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            mape, _, r2 = stats(pred[gen], y[te][gen])
            rows.setdefault(name, {})[arch] = (mape, r2)

    # ---- print a per-archetype MAPE / R2 table ---------------------------
    print(f"\n{'model':<20}" + "".join(f"{a:>20}" for a in ARCHS))
    print(f"{'':<20}" + "".join(f"{'MAPE%':>10}{'R2':>10}" for _ in ARCHS))
    for name, perarch in rows.items():
        line = f"{name:<20}"
        for a in ARCHS:
            mape, r2 = perarch[a]
            line += f"{mape:>10.1f}{r2:>10.3f}"
        print(line)


if __name__ == "__main__":
    main()
