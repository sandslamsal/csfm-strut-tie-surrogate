"""Figure for the two-series solver validation (E4 + the pier caps)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from common import FIG_DIR, HERE

from figstyle import tidy, panel, legend_below, save, INK2
txt = open(HERE / "e4_li_beams.log").read()
li = json.loads(txt[txt.index("{"):])
# (id, measured, predicted, predicted mode): every deep beam is predicted to crush
beams = [(r["id"], r["Pu"], r["Pcalc"], "crushing") for r in li["rows"]]
# pier caps: solver predictions of solver/scripts/validatePiercaps.ts, Geevar and Menon series
caps = [("S1", 2224, 2002, "rupture"), ("S2", 3068, 1994, "rupture"), ("S3", 3436, 3264, "crushing"),
        ("S4", 3608, 3608, "crushing"), ("S5", 3464, 3810, "rupture")]
C_CAP, C_BEAM = "#2B63A6", "#B8352B"


def scatter(ax, xs, ys, modes, colour, marker):
    for x, y, m in zip(xs, ys, modes):
        filled = m == "crushing"
        ax.scatter([x], [y], s=38, marker=marker, zorder=5, linewidths=1.2,
                   facecolor=colour if filled else "white", edgecolor=colour)


fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"wspace": 0.45})
lo, hi = 0, 4500
a.plot([lo, hi], [lo, hi], "--", lw=1.0, color="0.3", zorder=1)
scatter(a, [c[1] for c in caps], [c[2] for c in caps], [c[3] for c in caps], C_CAP, "o")
scatter(a, [bm[1] for bm in beams], [bm[2] for bm in beams], [bm[3] for bm in beams], C_BEAM, "s")
offsets = {"S1": (6, 4), "S2": (6, -9), "S3": (6, -9), "S4": (6, -9), "S5": (-17, 3)}
for sid, pe, pc, _ in caps:
    a.annotate(sid, (pe, pc), fontsize=8, xytext=offsets[sid], textcoords="offset points", color="0.25")
a.text(1000, 1600, "Deep beams", fontsize=8, color=C_BEAM, ha="center")
a.set_xticks([0, 1000, 2000, 3000, 4000]); a.set_yticks([0, 1000, 2000, 3000, 4000])
a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel("Measured ultimate load $P_{exp}$ (kN)")
a.set_ylabel("Reference-solver prediction $P_{calc}$ (kN)")
a.set_aspect("equal")
panel(a, "a", "Predicted against measured load")

ratios_cap = [pe / pc for _, pe, pc, _ in caps]
ratios_beam = [pe / pc for _, pe, pc, _ in beams]
x1 = np.arange(len(caps)); x2 = np.arange(len(beams)) + len(caps) + 1
b.axhline(1.0, ls="--", lw=1.0, color="0.3", zorder=1)
scatter(b, x1, ratios_cap, [c[3] for c in caps], C_CAP, "o")
scatter(b, x2, ratios_beam, [bm[3] for bm in beams], C_BEAM, "s")
b.hlines(np.mean(ratios_cap), x1[0] - 0.4, x1[-1] + 0.4, color=C_CAP, lw=1.4)
b.hlines(np.mean(ratios_beam), x2[0] - 0.4, x2[-1] + 0.4, color=C_BEAM, lw=1.4)
b.text(x1.mean(), 1.72, f"Mean {np.mean(ratios_cap):.2f}\nCoV {np.std(ratios_cap)/np.mean(ratios_cap):.2f}",
       ha="center", va="center", fontsize=8, color=C_CAP)
b.text(x2.mean(), 1.40, f"Mean {np.mean(ratios_beam):.2f}, CoV {np.std(ratios_beam)/np.mean(ratios_beam):.2f}",
       ha="center", va="center", fontsize=8, color=C_BEAM)
b.set_xticks(list(x1) + list(x2))
b.set_xticklabels([c[0] for c in caps] + [bm[0] for bm in beams], fontsize=8, rotation=45)
b.set_ylim(0.5, 1.9)
b.set_xlim(-1.0, 13.6)
b.set_ylabel("$P_{exp}/P_{calc}$")
panel(b, "b", "Ratio per specimen")
for ax in (a, b):
    tidy(ax)
fig.tight_layout()
fig.subplots_adjust(bottom=0.27)
handles = [Line2D([0], [0], ls="", marker="o", ms=6, mfc=C_CAP, mec=C_CAP),
           Line2D([0], [0], ls="", marker="s", ms=6, mfc=C_BEAM, mec=C_BEAM),
           Line2D([0], [0], ls="", marker="o", ms=6, mfc="white", mec=INK2, mew=1.2),
           Line2D([0], [0], ls="--", lw=1.0, color="0.3")]
legend_below(fig, handles, ["Pier caps (5)", "Deep beams (8)", "Open: tie rupture predicted", "1:1 line"],
             ncol=4, y=0.005, columnspacing=1.2)
save(fig, FIG_DIR / "validation_series.pdf")
print("wrote", FIG_DIR / "validation_series.pdf")
