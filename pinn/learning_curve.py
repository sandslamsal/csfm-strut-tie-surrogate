"""Learning curves and repeated-split stability.

Each network has about 85k parameters and 378-525 training designs, so
overfitting must be checked. This script reports:

  * a learning curve -- held-out test accuracy as a function of training-set
    size (fractions of the training split), repeated over several random
    subsamples, so the trend and its scatter are explicit; and
  * repeated-split stability -- the headline configuration retrained on several
    independent train/val/test splits, reported as mean +/- std, so the
    accuracy is shown not to hinge on one lucky split.

Writes ../figures/learning_curve.pdf (+ .tex) and prints the repeated-split
table.

Run:  python learning_curve.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import get_config
from data import load_archetype
from model import STMNet
from train import batch_loss

CACHE = "runs/learning_curve_data.json"   # cached curves -> replot without retraining

ARCHS = [
    ("deepBeam",        "Deep beam",         "o", "#2B63A6"),
    ("hammerhead",      "Hammerhead",        "s", "#1F8A70"),
    ("multiColumnBent", "Multi-column bent", "^", "#D9761A"),
    ("pileCap",         "Pile cap",          "D", "#B8352B"),
]
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
N_REPEAT = 3                 # subsample repeats per fraction
SPLIT_SEEDS = [20260517, 1, 7, 42, 2024]   # repeated-split seeds (first = paper)

PNG = "../figures/learning_curve.pdf"
TEX = "../figures/learning_curve.tex"



def train_inmem(data, cfg, train_idx) -> STMNet:
    torch.manual_seed(cfg.seed)
    model = STMNet(len(data.theta_keys), data.n_members, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    for _ in range(cfg.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            l_lam, l_F, l_eq = batch_loss(model, data, idx, cfg)
            (l_lam + cfg.w_sup * l_F + cfg.w_eq * l_eq).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            opt.zero_grad()
        sched.step()
    model.eval()
    return model


def test_metrics(model, data, cfg) -> tuple[float, float]:
    idx = data.test_idx
    gen = data.lambda_f[idx] < cfg.lambda_max - 1e-3
    idx = idx[gen]
    with torch.no_grad():
        pred = model(data.theta[idx])[0].numpy()
    true = data.lambda_f[idx].numpy()
    err = pred - true
    mape = float((np.abs(err) / np.clip(true, 1e-6, None)).mean() * 100)
    r2 = float(1 - (err ** 2).sum() / max(((true - true.mean()) ** 2).sum(), 1e-12))
    return mape, r2


def learning_curve(arch: str) -> tuple[list, list, list]:
    cfg = get_config(archetype=arch)
    data = load_archetype(cfg)
    ntr = len(data.train_idx)
    xs, mean_mape, std_mape = [], [], []
    for frac in FRACS:
        n = max(8, int(frac * ntr))
        reps = []
        for r in range(N_REPEAT):
            g = torch.Generator().manual_seed(1000 + r)
            sub = data.train_idx[torch.randperm(ntr, generator=g)[:n]]
            m = train_inmem(data, cfg, sub)
            reps.append(test_metrics(m, data, cfg)[0])
        xs.append(n)
        mean_mape.append(float(np.mean(reps)))
        std_mape.append(float(np.std(reps)))
        print(f"  [{arch}] n_train={n:4d}  MAPE={np.mean(reps):5.2f} "
              f"+/- {np.std(reps):4.2f}")
    return xs, mean_mape, std_mape


def repeated_splits(arch: str) -> tuple[float, float, float, float]:
    mapes, r2s = [], []
    for seed in SPLIT_SEEDS:
        cfg = get_config(archetype=arch, seed=seed)
        data = load_archetype(cfg)
        m = train_inmem(data, cfg, data.train_idx)
        mape, r2 = test_metrics(m, data, cfg)
        mapes.append(mape)
        r2s.append(r2)
    return (float(np.mean(mapes)), float(np.std(mapes)),
            float(np.mean(r2s)), float(np.std(r2s)))


def main() -> None:
    # Plot against the fraction of each archetype's training set, so every
    # curve runs to 100% and they end aligned (the archetypes differ in size:
    # the hammerhead retains fewer designs, so its full set is smaller).
    frac_pct = [int(f * 100) for f in FRACS]
    cached = os.path.exists(CACHE)
    if cached:
        curves = json.load(open(CACHE))
        print(f"loaded cached learning-curve data from {CACHE}")
    else:
        print("=== learning curves (test MAPE vs training-set size) ===")
        curves = {}
        for arch, _, _, _ in ARCHS:
            xs, mm, sm = learning_curve(arch)
            curves[arch] = {"frac": frac_pct, "mape": mm, "std": sm,
                            "ntr": xs[-1]}
        os.makedirs("runs", exist_ok=True)
        json.dump(curves, open(CACHE, "w"), indent=2)

    from figstyle import COLOUR, MARKER, LABEL, ORDER, tidy, panel, legend_below, INK2
    # (b) needs the width sweep of experiment E6
    for cand in ("../revision1/experiments/e6_width_sweep.json", "../experiments/e6_width_sweep.json"):
        if os.path.exists(cand):
            width = json.load(open(cand))
            break
    else:
        width = None

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"wspace": 0.45})
    handles = []
    for arch in ORDER:
        c = curves[arch]
        n = np.array(c["frac"]) / 100.0 * c["ntr"]
        mm, sm = np.array(c["mape"]), np.array(c["std"])
        col = COLOUR[arch]
        ax.fill_between(n, mm - sm, mm + sm, color=col, alpha=0.14, lw=0, zorder=1)
        h, = ax.plot(n, mm, "-", marker=MARKER[arch], color=col, lw=1.7, ms=5.2,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=3, label=LABEL[arch])
        handles.append(h)
        ax.annotate(f"{mm[-1]:.1f}%", (n[-1], mm[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=col, va="center")
    ax.set_xlabel("Training designs")
    ax.set_ylabel("Test MAPE on the failure load (%)")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 24)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    tidy(ax)
    panel(ax, "a", "Learning curves")

    if width is not None:
        for arch in ORDER:
            rows = width[arch]["widths"]
            params = np.array([r["params"] for r in rows])
            te = np.array([r["test_mape"] for r in rows])
            tr = np.array([r["train_mape"] for r in rows])
            sd = np.array([r["test_mape_sd"] for r in rows])
            col = COLOUR[arch]
            bx.fill_between(params, te - sd, te + sd, color=col, alpha=0.14, lw=0, zorder=1)
            bx.plot(params, te, "-", marker=MARKER[arch], color=col, lw=1.7, ms=5.2,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            bx.plot(params, tr, "--", color=col, lw=1.2, zorder=2)
        p128 = width["deepBeam"]["widths"][3]["params"]
        bx.axvline(p128, ls=":", lw=0.9, color=INK2, zorder=1)
        bx.text(p128 * 1.12, 0.35, "Width 128", fontsize=8, color=INK2, va="bottom")
        bx.set_xscale("log")
        bx.set_xlabel("Trainable parameters (six hidden layers)")
        bx.set_ylabel("MAPE on the failure load (%)")
        bx.set_ylim(0, 11)
        tidy(bx)
        from matplotlib.lines import Line2D
        style = [Line2D([0], [0], color=INK2, lw=1.7, marker="o", ms=4, markeredgecolor="white"),
                 Line2D([0], [0], color=INK2, lw=1.2, ls="--")]
        bx.legend(style, ["Test split", "Training split"], loc="upper right", fontsize=8, bbox_to_anchor=(1.0, 0.98))
        panel(bx, "b", "Network-width sweep")
    fig.tight_layout(w_pad=2.0)
    fig.subplots_adjust(bottom=0.22)
    legend_below(fig, handles, [LABEL[a] for a in ORDER], ncol=4, y=0.005)
    fig.savefig(PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PNG}")

    with open(TEX, "w") as fh:
        fh.write(r"""%% Figure: learning curves rendered by pinn/learning_curve.py.
\begin{figure}[!htb]
  \centering
  \includegraphics[width=\linewidth]{learning_curve.pdf}
  \caption{Data efficiency and network size. (a)~Learning curves: held-out
    test mean absolute percentage error against the number of training
    designs for each archetype, each point the mean of three independent
    random subsamples and the band one standard deviation; the label gives
    the error at the full training set. (b)~Network-width sweep at six hidden
    layers, three seeds: test error (solid, band one standard deviation) and
    training error (dashed) against the number of trainable parameters. The
    train/test gap does not grow with size and the test error is flat beyond
    the reported width of 128.}
  \label{fig:learning_curve}
\end{figure}
""")

    if cached:
        return
    print("\n=== repeated-split stability (5 independent splits) ===")
    print(f"{'archetype':16s} {'MAPE% mean':>11s} {'std':>6s} "
          f"{'R2 mean':>9s} {'std':>7s}")
    for arch, label, _, _ in ARCHS:
        mm, ms, rm, rs = repeated_splits(arch)
        print(f"{label:16s} {mm:11.2f} {ms:6.2f} {rm:9.3f} {rs:7.3f}")


if __name__ == "__main__":
    main()
