"""Generate the Results parity figure from the trained models.

Loads each archetype's trained network, predicts the failure load factor on
its held-out test split, and renders a two-panel publication figure:

  (a) surrogate-vs-reference-solver parity scatter with +/-5% and +/-10% tolerance
      bands and a per-archetype coefficient of determination;
  (b) the distribution of the signed prediction error per archetype.

Writes, in ../figures/:
  results_parity.pdf   -- the rendered figure (600 dpi)
  results_parity.tex   -- a thin \\includegraphics wrapper + caption

Run (after training all four archetypes):
    python make_parity.py
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import get_config
from data import load_archetype
from model import STMNet
from figstyle import panel

# archetype -> (legend label, marker, colour) -- colours match the csfd palette
ARCHS = [
    ("deepBeam",        "Deep beam",         "o", "#2B63A6"),
    ("hammerhead",      "Hammerhead",        "s", "#1F8A70"),
    ("multiColumnBent", "Multi-column bent", "^", "#D9761A"),
    ("pileCap",         "Pile cap",          "D", "#B8352B"),
]

PNG = "../figures/results_parity.pdf"
TEX = "../figures/results_parity.tex"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.9, "savefig.dpi": 600, "pdf.fonttype": 42,
    "mathtext.fontset": "cm",
})


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def collect() -> list:
    """Return [(label, marker, colour, true, pred), ...] for every archetype."""
    series = []
    for arch, label, mark, colour in ARCHS:
        cfg = get_config(archetype=arch)
        data = load_archetype(cfg)
        ckpt = torch.load(os.path.join(cfg.out_dir, arch, "model.pt"),
                          map_location="cpu", weights_only=False)
        model = STMNet(len(data.theta_keys), data.n_members, cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            pred, _ = model(data.theta[data.test_idx])
        true = np.asarray(data.lambda_f[data.test_idx].tolist(), dtype=float)
        pred = np.asarray(pred.tolist(), dtype=float)
        series.append((label, mark, colour, true, pred))
    return series


def parity_panel(ax, series, lo, hi) -> None:
    xs = np.array([lo, hi])
    # tolerance bands
    for fac in (0.90, 1.10):
        ax.plot(xs, xs * fac, ls=":", lw=0.8, color="0.45", zorder=1)
    for fac in (0.95, 1.05):
        ax.plot(xs, xs * fac, ls=":", lw=0.8, color="0.65", zorder=1)
    ax.plot(xs, xs, ls="--", lw=1.1, color="0.30", zorder=2)
    ax.annotate("$\\pm$5%", (hi, hi * 1.05), fontsize=6.3, color="0.45",
                ha="right", va="bottom")
    ax.annotate("$\\pm$10%", (hi, hi * 1.10), fontsize=6.3, color="0.45",
                ha="right", va="bottom")

    for label, mark, colour, true, pred in series:
        ax.scatter(true, pred, s=20, marker=mark, facecolor=colour,
                   edgecolor="white", linewidth=0.35, alpha=0.80,
                   zorder=3, label=f"{label}  ($R^2$ {r2_score(true, pred):.3f})")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xlabel("Reference solver failure load factor, $\\lambda_f$",
                  fontsize=9.2)
    ax.set_ylabel("Surrogate prediction, $\\hat{\\lambda}_f$", fontsize=9.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.4")
    ax.spines["bottom"].set_color("0.4")
    ax.tick_params(length=3, color="0.4")
    ax.grid(True, linewidth=0.4, color="0.90", zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper left", frameon=True, fontsize=7.4,
                    handletextpad=0.4, borderpad=0.6, labelspacing=0.5)
    leg.get_frame().set_edgecolor("0.80")
    leg.get_frame().set_linewidth(0.6)
    panel(ax, "a", "Parity on the held-out test designs")


def error_panel(ax, series) -> None:
    """Horizontal box-and-strip plot of the signed percentage error."""
    pos = list(range(len(series), 0, -1))   # deep beam at the top
    errs = [((p - t) / t) * 100.0 for _, _, _, t, p in series]
    colours = [c for _, _, c, _, _ in series]

    bp = ax.boxplot(errs, vert=False, positions=pos, widths=0.52,
                    patch_artist=True, showfliers=False, zorder=2,
                    medianprops=dict(color="0.12", lw=1.1),
                    whiskerprops=dict(color="0.45", lw=0.8),
                    capprops=dict(color="0.45", lw=0.8),
                    boxprops=dict(lw=0.8))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.32)
        patch.set_edgecolor(colour)

    rng = np.random.default_rng(0)
    for y, err, colour in zip(pos, errs, colours):
        ys = y + rng.uniform(-0.16, 0.16, size=err.size)
        ax.scatter(err, ys, s=8, color=colour, alpha=0.55, lw=0, zorder=3)
        mape = float(np.mean(np.abs(err)))
        ax.text(0.985, y + 0.34, f"MAPE {mape:.1f}%",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.8, color="0.30")

    ax.axvline(0.0, ls="--", lw=1.0, color="0.30", zorder=1)
    ax.set_yticks(pos)
    ax.set_yticklabels([s[0] for s in series], fontsize=8.4)
    ax.set_ylim(0.4, len(series) + 0.6)
    lim = max(np.max(np.abs(e)) for e in errs) * 1.10
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Surrogate prediction error, "
                  "$(\\hat{\\lambda}_f-\\lambda_f)/\\lambda_f$ (%)",
                  fontsize=9.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.4")
    ax.spines["bottom"].set_color("0.4")
    ax.tick_params(length=3, color="0.4")
    ax.xaxis.grid(True, linewidth=0.4, color="0.90", zorder=0)
    ax.set_axisbelow(True)
    panel(ax, "b", "Signed prediction error per archetype")


def write_tex() -> None:
    tex = r"""%% Figure: surrogate-vs-reference-solver parity and error distribution.
%% PNG rendered by pinn/make_parity.py (600 dpi) -- do not edit by hand.
\begin{figure}[!htb]
  \centering
  \includegraphics[width=\linewidth]{results_parity.pdf}
  \caption{Surrogate accuracy on the held-out test split, for all four
    archetypes. (a)~Predicted failure load factor $\hat{\lambda}_f$
    against the reference solver
    $\lambda_f$; the
    dashed line is perfect agreement and the dotted lines mark the
    $\pm5\%$ and $\pm10\%$ tolerances. (b)~Distribution of the signed
    prediction error per archetype, with the box spanning the
    interquartile range and the mean absolute percentage error annotated.
    Almost every prediction falls inside the $\pm10\%$ band. Metrics shown
    here are computed over all test designs, including the non-failing
    ones at $\lambda_f = 3.0$; Table~\ref{tab:accuracy} reports the
    stricter statistics restricted to the genuine-failure designs.}
  \label{fig:parity}
\end{figure}
"""
    with open(TEX, "w") as fh:
        fh.write(tex)


def main() -> None:
    series = collect()
    all_true = np.concatenate([s[3] for s in series])
    all_pred = np.concatenate([s[4] for s in series])
    lo = float(min(all_true.min(), all_pred.min())) - 0.10
    hi = float(max(all_true.max(), all_pred.max())) + 0.15

    fig = plt.figure(figsize=(9.2, 4.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.04], wspace=0.28)
    parity_panel(fig.add_subplot(gs[0]), series, lo, hi)
    error_panel(fig.add_subplot(gs[1]), series)

    fig.savefig(PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PNG}")

    write_tex()
    overall_mape = float(np.mean(np.abs((all_pred - all_true) / all_true)) * 100)
    print(f"wrote {TEX}  (overall R^2 {r2_score(all_true, all_pred):.3f}, "
          f"MAPE {overall_mape:.1f}%)")


if __name__ == "__main__":
    main()
