"""Per-archetype calibration and interval-width/error analysis.

The ensemble-spread/error correlation (pooled rho ~0.49) is informative but not
strong, so calibration is checked per archetype and against interval width.
Using the already-trained bagged ensembles, this script produces:

  * a 2x2 grid of per-archetype reliability diagrams (raw ensemble sigma read
    as a Gaussian interval vs split-conformal), so the calibration is shown to
    hold for each archetype, not just pooled; and
  * an interval-width/error table: test designs binned into terciles by their
    90% conformal half-width, reporting the mean half-width and the mean
    absolute error in each bin, so that wider intervals are shown to fall on
    the harder designs.

Run (after make_ensemble.py has trained the ensembles):  python calibration.py
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import get_config
from data import load_archetype
from model import STMNet
from make_ensemble import conformal_q, predict, _erfinv, K, CENSOR, SIG_FLOOR
from figstyle import panel, legend_below, save

ARCHS = [
    ("deepBeam",        "Deep beam",         "#2B63A6"),
    ("hammerhead",      "Hammerhead",        "#1F8A70"),
    ("multiColumnBent", "Multi-column bent", "#D9761A"),
    ("pileCap",         "Pile cap",          "#B8352B"),
]
PNG = "../figures/calibration.pdf"
TEX = "../figures/calibration.tex"



def load_ensemble(arch: str, theta_dim: int, n_members: int, cfg):
    models = []
    ens = os.path.join(cfg.out_dir, arch, "ensemble")
    for f in sorted(os.listdir(ens)):
        if f.endswith(".pt"):
            m = STMNet(theta_dim, n_members, cfg)
            m.load_state_dict(torch.load(os.path.join(ens, f),
                              map_location="cpu",
                              weights_only=False)["state_dict"])
            m.eval()
            models.append(m)
    return models


def archetype_data(arch: str) -> dict:
    cfg = get_config(archetype=arch)
    data = load_archetype(cfg)
    models = load_ensemble(arch, len(data.theta_keys), data.n_members, cfg)
    cal_mean, cal_std = predict(models, data.theta[data.val_idx])
    test_mean, test_std = predict(models, data.theta[data.test_idx])
    cal_true = np.asarray(data.lambda_f[data.val_idx].tolist(), dtype=float)
    test_true = np.asarray(data.lambda_f[data.test_idx].tolist(), dtype=float)
    ck, tk = cal_true < CENSOR, test_true < CENSOR
    cal_std = np.clip(cal_std[ck], SIG_FLOOR, None)
    test_std = np.clip(test_std[tk], SIG_FLOOR, None)
    cal_scores = np.abs(cal_true[ck] - cal_mean[ck]) / cal_std
    return {"test_true": test_true[tk], "test_mean": test_mean[tk],
            "test_std": test_std, "cal_scores": cal_scores}


def main() -> None:
    res = {a: archetype_data(a) for a, _, _ in ARCHS}

    # ---- 2x2 per-archetype reliability diagrams --------------------------
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.38, "wspace": 0.15})
    levels = np.linspace(0.10, 0.95, 28)
    for i, (ax, (arch, label, colour)) in enumerate(zip(axes.ravel(), ARCHS)):
        r = res[arch]
        err = np.abs(r["test_mean"] - r["test_true"])
        zs = np.array([math.sqrt(2) * _erfinv(2 * p - 1) for p in levels])
        raw = np.array([float(np.mean(err <= z * r["test_std"])) for z in zs])
        conf = np.array([float(np.mean(err <= conformal_q(r["cal_scores"], p)
                                       * r["test_std"])) for p in levels])
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="0.45")
        h_raw, = ax.plot(levels, raw, "-", color="#B8352B", lw=1.5,
                         label="raw ensemble $\\sigma$")
        h_conf, = ax.plot(levels, conf, "-", color="0.15", lw=1.8,
                          label="conformal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        panel(ax, "abcd"[i], label)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, lw=0.4, color="0.90")
        ax.set_axisbelow(True)
        ax.set_xlabel("Target coverage" if i >= 2 else "", fontsize=9)
        ax.set_ylabel("Observed coverage" if i % 2 == 0 else "", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    legend_below(fig, [h_raw, h_conf], ["Raw ensemble spread", "Conformal interval"], ncol=2, y=0.005)
    save(fig, PNG)
    plt.close(fig)
    print(f"wrote {PNG}")

    with open(TEX, "w") as fh:
        fh.write(r"""%% Figure: per-archetype calibration rendered by pinn/calibration.py.
\begin{figure}[!htb]
  \centering
  \includegraphics[width=\linewidth]{calibration.pdf}
  \caption{Per-archetype reliability diagrams. For every archetype the raw
    ensemble standard deviation, read as a Gaussian interval, is
    under-dispersed and falls below the diagonal, whereas the
    $\sigma$-normalised split-conformal interval tracks the target coverage.}
  \label{fig:calibration}
\end{figure}
""")

    # ---- interval-width / error tercile analysis -------------------------
    print("\n=== interval width vs error (90% conformal, test designs) ===")
    print(f"{'archetype':16s} {'bin':>8s} {'n':>4s} "
          f"{'half-width':>11s} {'mean |err|':>11s}")
    for arch, label, _ in ARCHS:
        r = res[arch]
        q = conformal_q(r["cal_scores"], 0.90)
        half = q * r["test_std"]
        err = np.abs(r["test_mean"] - r["test_true"])
        order = np.argsort(half)
        thirds = np.array_split(order, 3)
        for name, bin_idx in zip(("narrow", "medium", "wide"), thirds):
            print(f"{label:16s} {name:>8s} {len(bin_idx):4d} "
                  f"{half[bin_idx].mean():11.3f} {err[bin_idx].mean():11.3f}")
        print()


if __name__ == "__main__":
    main()
