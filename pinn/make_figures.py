"""Generate the consolidated pier-cap validation figure (modern, 600 dpi).

Produces, in ../figures/:
  validation_combined.pdf  -- one figure carrying the whole two-tier story

The figure has two panels:
  (a) a parity plot, predicted vs measured ultimate load for the continuum CSFM
      analysis, the reference solver and the neural surrogate, with the 1:1 line
      The solver and surrogate cluster
      together below the line (the surrogate tracks the solver -- tier 1), while
      the continuum analysis lies nearer the line; the gap to 1:1 is the
      conservative bias against experiment (tier 2).
  (b) a compact ratio strip, the experimental-to-predicted ratio per specimen
      for the three predictions, with the unity (measured) line.

Per-specimen numbers are in the validation tables; the figure carries the
qualitative two-tier comparison.

Run (after training the pierCap surrogate):
    python make_figures.py
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
from figstyle import panel, legend_below, save

DATA = "../validation/piercaps_geevar_menon_2018.json"
SPECS = ["S1", "S2", "S3", "S4", "S5"]
RUPTURE = {"S2"}   # solver-predicted reinforcement rupture

# reference-solver total-load predictions (kN) -- scripts/validatePiercaps.ts
SOLVER = [1890, 1994, 2233, 2526, 2771]

# modern flat palette
C_SOLV = "#2B63A6"   # reference solver (blue)
C_CONT = "#D9761A"   # continuum CSFM, Kaufmann et al. (orange)
C_SURR = "#7850A8"   # neural surrogate (purple)
C_SAFE = "#2C8754"   # conservative-region green



def _modern(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.4")
    ax.spines["bottom"].set_color("0.4")
    ax.tick_params(length=3, color="0.4")
    ax.grid(True, linewidth=0.5, color="0.92", zorder=0)
    ax.set_axisbelow(True)


def combined_figure(path: str, measured, series) -> None:
    """series: list of (label, 5 values, colour, marker)."""
    meas = np.asarray(measured, float)
    fig, (axp, axr) = plt.subplots(
        1, 2, figsize=(7.0, 3.3),
        gridspec_kw={"width_ratios": [1.5, 0.9], "wspace": 0.36})

    hi = max(meas.max(), max(np.asarray(v).max() for _, v, _, _ in series)) * 1.10

    # ---------- panel (a): parity, three predictions ----------
    axp.plot([0, hi], [0, hi], ls="--", color="0.45", lw=1.2, zorder=2)
    axp.text(hi * 0.985, hi * 0.955, "1:1", color="0.45", fontsize=9,
             ha="right", va="top", rotation=45)

    means = []
    handles = []
    for label, vals, colour, marker in series:
        vals = np.asarray(vals, float)
        h = axp.scatter(meas, vals, s=66, color=colour, edgecolor="white",
                        linewidth=1.0, marker=marker, zorder=5, label=label)
        handles.append(h)
        means.append((label, float(np.mean(meas / vals))))

    axp.set_xlim(0, hi)
    axp.set_ylim(0, hi)
    axp.set_aspect("equal")
    axp.set_xlabel("Measured load $P$ (kN)", fontsize=10)
    axp.set_ylabel("Predicted load $P$ (kN)", fontsize=10)
    _modern(axp)
    # compact mean-ratio annotation (no per-point labels; see tables)
    txt = "Mean $P_{exp}/P_{pred}$\n" + "\n".join(
        f"{lab.split(' (')[0]}: {m:.2f}" for lab, m in means)
    axp.text(0.03, 0.97, txt, transform=axp.transAxes, fontsize=8,
             ha="left", va="top", color="0.25",
             bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                       edgecolor="0.82", linewidth=0.6))
    panel(axp, "a", "Predicted against measured")

    # ---------- panel (b): compact ratio strip ----------
    y = np.arange(len(SPECS))[::-1]
    axr.axvline(1.0, color="0.45", lw=1.2, zorder=1)
    axr.text(1.0, -0.75, "Measured", color="0.45", fontsize=8,
             ha="center", va="top")
    n = len(series)
    for j, (label, vals, colour, marker) in enumerate(series):
        ratio = meas / np.asarray(vals, float)
        off = (j - (n - 1) / 2) * 0.22
        axr.scatter(ratio, y + off, s=34, color=colour, edgecolor="white",
                    linewidth=0.7, marker=marker, zorder=5)
    axr.set_yticks(y)
    axr.set_yticklabels(SPECS, fontsize=10)
    for tick in axr.get_yticklabels():
        if tick.get_text() in RUPTURE:
            tick.set_color(C_SOLV)
            tick.set_fontweight("bold")
    axr.set_ylim(-1.0, len(SPECS) - 0.4)
    axr.set_xlabel("$P_{exp}/P_{pred}$", fontsize=10)
    _modern(axr)
    panel(axr, "b", "Ratio per specimen")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    legend_below(fig, handles, [h.get_label() for h in handles], ncol=3, y=0.005)
    save(fig, path)
    plt.close(fig)
    print(f"wrote {path}")


def surrogate_predictions(specimens: list, geom: dict,
                          bands: list) -> list[float]:
    """Total-load prediction of the trained pier-cap surrogate (kN)."""
    cfg = get_config(archetype="pierCap")
    data = load_archetype(cfg)
    ckpt = torch.load(os.path.join(cfg.out_dir, "pierCap", "model.pt"),
                      map_location="cpu", weights_only=False)
    model = STMNet(len(data.theta_keys), data.n_members, cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    out = []
    for s in specimens:
        exp_total = s["measured"]["Pu_total_kN"]
        n_bars = (s["reinforcement"]["primary_As1"]["count"]
                  + (s["reinforcement"]["additional_As2"]["count"]
                     if s["reinforcement"]["additional_As2"] else 0))
        raw = {
            "capWidth": geom["topCapWidth"],
            "stemWidth": geom["lowerStemWidth"],
            "capBandHeight": bands[0], "taperHeight": bands[1],
            "stemHeight": bands[2],
            "thickness": geom["outOfPlaneThickness_b"],
            "loadPlate": s["loadPlate_lb_mm"],
            "columnLoad": exp_total * 1e3,
            "supportPlate": 150.0,
            "edgeDistance": geom["dimension_a"],
            "mainDia": s["reinforcement"]["primary_As1"]["barDia_mm"],
            "mainCount": n_bars,
            "fck": s["materials"]["concrete"]["fc_MPa"],
            "fy": s["materials"]["mainReinf"]["fy_MPa"],
        }
        vec = torch.tensor([[raw[k] for k in data.theta_keys]],
                           dtype=torch.float32)
        theta = (vec - data.theta_mean) / data.theta_std
        with torch.no_grad():
            lam_f, _ = model(theta)
        out.append(float(lam_f) * exp_total)
    return out


def main() -> None:
    dataset = json.load(open(DATA))
    geom = dataset["geometry"]["reportedFigureDimensions_mm"]
    bands = geom["heightBands"]
    specimens = dataset["specimens"]

    measured = [s["measured"]["Pu_total_kN"] for s in specimens]
    book = [s["bookCsfmPrediction_perSupport_kN"]["M0"] * 4 for s in specimens]
    surrogate = surrogate_predictions(specimens, geom, bands)

    combined_figure(
        "../figures/validation_combined.pdf", measured,
        [("Continuum CSFM (Kaufmann et al.)", book, C_CONT, "s"),
         ("Reference solver", SOLVER, C_SOLV, "o"),
         ("Neural surrogate", surrogate, C_SURR, "D")])


if __name__ == "__main__":
    main()
