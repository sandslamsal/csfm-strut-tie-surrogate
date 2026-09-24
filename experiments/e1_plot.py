"""Figure for the configuration-change experiment (E1): from scratch vs warm start."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import FIG_DIR, HERE
from figstyle import tidy, panel, INK2

C_SCRATCH, C_WARM = "#B8352B", "#2B63A6"
r = json.load(open(HERE / "e1_config_change.json"))
fig, ax = plt.subplots(figsize=(5.6, 3.7))
handles, labels = [], []
for key, lab, col, mk in (("scratch", "trained from scratch", C_SCRATCH, "o"),
                          ("warm", "warm start from the single-load network", C_WARM, "s")):
    rows = r[key]
    h = ax.errorbar([x["n"] for x in rows], [x["mape"] for x in rows], yerr=[x["sd"] for x in rows],
                    fmt="-" + mk, color=col, lw=1.6, ms=5, capsize=2.5, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    handles.append(h); labels.append(lab)
    handles.append(h); labels.append(lab)
    ax.annotate(lab.split(" from")[0] if key == "scratch" else "warm start",
                (rows[-1]["n"], rows[-1]["mape"]), xytext=(6, 0), textcoords="offset points",
                fontsize=7.4, color=INK2, va="center")
# the reference: the single-load deep beam at its full training set
ax.axhline(5.6, ls="--", lw=0.9, color=INK2, zorder=2)
ax.text(140, 4.3, "single-load deep beam at its full training set: 5.6%", fontsize=7, color=INK2)
# the data saving: a warm start at 131 designs matches training from scratch at 393
w131 = next(x for x in r["warm"] if x["n"] == 131)["mape"]
s393 = next(x for x in r["scratch"] if x["n"] == 393)["mape"]
ax.text(330, 21.5, "warm start at 131 designs matches\ntraining from scratch at 393", fontsize=7.2,
        color=INK2, ha="center", va="center")
for x, yv in ((131, w131), (393, s393)):
    ax.annotate("", xy=(x, yv + 1.2), xytext=(330, 19.6),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=0.0"))
ax.set_xlabel("Training designs of the new configuration")
ax.set_ylabel("Test MAPE on the failure load (%)")
ax.set_xlim(20, 560)
ax.set_ylim(0, 32)
ax.set_xticks([52, 131, 262, 393, 525])
tidy(ax)
ax.legend(handles, labels, loc="upper right", fontsize=7.6)
panel(ax, "a", "Two-point-load deep beam: accuracy against training designs")
fig.tight_layout()
fig.savefig(FIG_DIR / "config_change.pdf", bbox_inches="tight")
print("wrote", FIG_DIR / "config_change.pdf")
