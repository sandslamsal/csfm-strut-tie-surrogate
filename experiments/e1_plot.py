"""Figure for the configuration-change cost (E1): from scratch vs warm start."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import FIG_DIR, HERE

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.9,
                     "mathtext.fontset": "cm", "pdf.fonttype": 42})
r = json.load(open(HERE / "e1_config_change.json"))
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for key, lab, col, mk in (("scratch", "trained from scratch", "#BE342E", "o"),
                          ("warm", "warm start from the single-load network", "#26629E", "s")):
    rows = r[key]
    ax.errorbar([x["n"] for x in rows], [x["mape"] for x in rows], yerr=[x["sd"] for x in rows],
                fmt="-" + mk, color=col, lw=1.4, ms=4.5, capsize=2.5, label=lab)
ax.axhline(5.6, ls="--", lw=0.9, color="0.3")
ax.text(150, 4.0, "single-load deep beam, full data", fontsize=7, ha="left", color="0.3")
ax.set_xlabel("Training designs of the new configuration")
ax.set_ylabel("Test MAPE on the failure load (%)")
ax.set_ylim(0, 32)
ax.legend(fontsize=7.5, frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, lw=0.4, color="0.92"); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG_DIR / "config_change.pdf", bbox_inches="tight")
print("wrote", FIG_DIR / "config_change.pdf")
