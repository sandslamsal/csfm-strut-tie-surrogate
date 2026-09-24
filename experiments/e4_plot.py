"""Figure for the two-series solver validation (E4 + the pier caps)."""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import FIG_DIR, HERE

from figstyle import tidy, panel
txt = open(HERE / "e4_li_beams.log").read()
li = json.loads(txt[txt.index("{"):])
beams = [(r["id"], r["Pu"], r["Pcalc"]) for r in li["rows"]]
# pier caps: Table (validation) of the manuscript, Geevar and Menon series
caps = [("S1", 2224, 1890), ("S2", 3068, 1994), ("S3", 3436, 2233),
        ("S4", 3608, 2526), ("S5", 3464, 2771)]
C_CAP, C_BEAM = "#2B63A6", "#B8352B"

fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.7))
lo, hi = 500, 4000
a.plot([lo, hi], [lo, hi], "--", lw=1.0, color="0.3", label="1:1")
a.plot([lo, hi], [lo / 1.39, hi / 1.39], ":", lw=0.9, color=C_CAP)
a.text(2350, 2350 / 1.39 - 420, "mean 1.39", fontsize=7, color=C_CAP)
a.plot([lo, hi], [lo / 0.45, hi / 0.45], ":", lw=0.9, color=C_BEAM)
a.text(1250, 1250 / 0.45 + 150, "mean 0.45", fontsize=7, color=C_BEAM)
a.scatter([c[1] for c in caps], [c[2] for c in caps], s=34, marker="o", color=C_CAP,
          edgecolor="white", lw=0.5, zorder=5, label="pier caps, Geevar and Menon (5)")
a.scatter([bm[1] for bm in beams], [bm[2] for bm in beams], s=34, marker="s", color=C_BEAM,
          edgecolor="white", lw=0.5, zorder=5, label="deep beams, Li et al. (8)")
for sid, pe, pc in caps:
    a.annotate(sid, (pe, pc), fontsize=6.5, xytext=(4, -7), textcoords="offset points", color="0.25")
a.set_xticks([1000, 2000, 3000, 4000]); a.set_yticks([1000, 2000, 3000, 4000])
a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel("Measured ultimate load $P_{exp}$ (kN)")
a.set_ylabel("Reference-solver prediction $P_{calc}$ (kN)")
a.set_aspect("equal")
a.legend(fontsize=7, frameon=False, loc="lower right")
panel(a, "a", "Predicted against measured load")

ratios_cap = [pe / pc for _, pe, pc in caps]
ratios_beam = [pe / pc for _, pe, pc in beams]
x1 = np.arange(len(caps)); x2 = np.arange(len(beams)) + len(caps) + 1
b.axhline(1.0, ls="--", lw=1.0, color="0.3")
b.scatter(x1, ratios_cap, s=34, color=C_CAP, zorder=5)
b.scatter(x2, ratios_beam, s=34, marker="s", color=C_BEAM, zorder=5)
b.hlines(np.mean(ratios_cap), x1[0] - 0.4, x1[-1] + 0.4, color=C_CAP, lw=1.4)
b.hlines(np.mean(ratios_beam), x2[0] - 0.4, x2[-1] + 0.4, color=C_BEAM, lw=1.4)
b.text(x1.mean(), 1.70, f"mean {np.mean(ratios_cap):.2f}, CoV {np.std(ratios_cap)/np.mean(ratios_cap):.2f}",
       ha="center", va="center", fontsize=7.5, color=C_CAP)
b.text(x2.mean(), 0.22, f"mean {np.mean(ratios_beam):.2f}, CoV {np.std(ratios_beam)/np.mean(ratios_beam):.2f}",
       ha="center", va="center", fontsize=7.5, color=C_BEAM)
b.set_xticks(list(x1) + list(x2))
b.set_xticklabels([c[0] for c in caps] + [bm[0] for bm in beams], fontsize=7, rotation=45)
b.set_ylim(0, 1.8)
b.set_ylabel("$P_{exp}/P_{calc}$")
panel(b, "b", "Ratio per specimen")
for ax in (a, b):
    tidy(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "validation_series.pdf", bbox_inches="tight")
print("wrote", FIG_DIR / "validation_series.pdf")
