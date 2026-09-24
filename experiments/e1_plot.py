"""Figure for the configuration-change experiment (E1): the two truss
configurations of the same deep beam, and the accuracy of a network trained
from scratch against one warm-started from the single-load network."""
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from common import FIG_DIR, HERE
from figstyle import tidy, panel, legend_below, INK, INK2, save

C_SCRATCH, C_WARM = "#B8352B", "#2B63A6"
C_STRUT, C_TIE, C_ZERO = "#2B63A6", "#B8352B", "0.72"


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def first_design(cands, arch):
    for c in cands:
        if c.exists():
            return next(d for d in load(c)["designs"] if d["archetype"] == arch)
    raise FileNotFoundError(cands)


single = first_design([HERE.parents[1] / "data" / "dataset.json", HERE.parent / "data" / "dataset.json.gz",
                       HERE.parent / "data" / "dataset.json"], "deepBeam")
double = first_design([HERE / "data" / "dataset_deepBeam2P.json", HERE.parent / "data" / "dataset_deepBeam2P.json.gz"], "deepBeam2P")


def draw_truss(ax, d, y0, title):
    """Strut and tie model in metres, members coloured by the failure-state force."""
    nodes = {n["id"]: (n["x"] / 1e3, n["y"] / 1e3 + y0) for n in d["truss"]["nodes"]}
    fixed = {n["id"]: n["fixed"] for n in d["truss"]["nodes"]}
    forces = d["labels"]["csfmStates"][-1]["forces"]
    fmax = max(abs(v) for v in forces.values())
    for m in d["truss"]["members"]:
        (x1, y1), (x2, y2) = nodes[m["ni"]], nodes[m["nj"]]
        f = forces.get(m["id"], 0.0)
        if abs(f) < 0.02 * fmax:
            ax.plot([x1, x2], [y1, y2], color=C_ZERO, lw=0.9, ls=(0, (3, 2)), zorder=2)
        else:
            col = C_STRUT if f < 0 else C_TIE
            ax.plot([x1, x2], [y1, y2], color=col, lw=1.2 + 3.0 * abs(f) / fmax, zorder=3,
                    solid_capstyle="round")
    for nid, (x, y) in nodes.items():
        ax.plot(x, y, "o", ms=5, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.9, zorder=5)
        fx, fy, _ = fixed[nid]
        if fy:  # support symbol under the node
            tri = Polygon([[x, y - 0.03], [x - 0.13, y - 0.24], [x + 0.13, y - 0.24]], closed=True,
                          facecolor="white" if not fx else "0.55", edgecolor=INK, lw=0.8, zorder=4)
            ax.add_patch(tri)
    for ld in d["truss"]["loads"]:
        x, y = nodes[ld["node"]]
        ax.plot(x, y + 0.2, marker="v", ms=9, color=INK, markeredgecolor=INK, zorder=6)
        ax.text(x + 0.16, y + 0.62, f"{abs(ld['fy']) / 1e6:.2f} MN", fontsize=8, color=INK, va="center")
    n_nodes, n_members = len(d["truss"]["nodes"]), len(d["truss"]["members"])
    xs = [v[0] for v in nodes.values()]
    ax.text(min(xs) - 0.05, y0 + 2.42, title, fontsize=8, color=INK, va="bottom")
    ax.text(max(xs) + 0.05, y0 - 0.5, f"{n_nodes} nodes, {n_members} members", fontsize=8,
            color=INK2, va="top", ha="right")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.30})
draw_truss(a1, single, 4.05, "Single midspan load")
draw_truss(a1, double, 0.0, "Two quarter-point loads")
a1.set_aspect("equal")
a1.set_xlim(-3.35, 3.55)
a1.set_ylim(-1.75, 6.85)
a1.axis("off")
a1.set_title("(a) The two loading configurations", loc="left",
             fontweight="bold", fontsize=10, pad=7, color=INK)
lines = [Line2D([0], [0], color=C_STRUT, lw=2.8), Line2D([0], [0], color=C_TIE, lw=2.8),
         Line2D([0], [0], color=C_ZERO, lw=0.9, ls=(0, (3, 2)))]
a1.legend(lines, ["Strut", "Tie", "Unstressed"],
          loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8, frameon=False,
          handlelength=2.0, columnspacing=0.9, handletextpad=0.5)

r = json.load(open(HERE / "e1_config_change.json"))
handles, labels = [], []
for key, lab, col, mk in (("scratch", "Two-point-load designs only", C_SCRATCH, "o"),
                          ("warm", "Transfer from single-load network", C_WARM, "s")):
    rows = r[key]
    n = [x["n"] for x in rows]; m = [x["mape"] for x in rows]; sd = [x["sd"] for x in rows]
    a2.fill_between(n, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)], color=col,
                    alpha=0.14, lw=0, zorder=1)
    h, = a2.plot(n, m, "-", marker=mk, color=col, lw=1.7, ms=5.2, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=3)
    handles.append(h); labels.append(lab)
    a2.annotate(f"{m[-1]:.1f}%", (n[-1], m[-1]), xytext=(6, 5 if key == "scratch" else -5),
                textcoords="offset points", fontsize=8, color=col, va="center")
a2.axhline(5.6, ls="--", lw=0.9, color=INK2, zorder=2)
a2.text(60, 3.6, "Single-load beam, 525 designs: 5.6%", fontsize=8,
        color=INK2, ha="left", va="center")
w131 = next(x for x in r["warm"] if x["n"] == 131)["mape"]
s393 = next(x for x in r["scratch"] if x["n"] == 393)["mape"]
a2.text(548, 16.5, f"Transfer, 131 designs: {w131:.1f}%\nTwo-point-load designs only, 393 designs: {s393:.1f}%",
        fontsize=8, color=INK2, ha="right", va="center", linespacing=1.4)
a2.set_xlabel("Two-point-load training designs")
a2.set_ylabel("Test MAPE on the failure load (%)")
a2.set_xlim(20, 600); a2.set_ylim(0, 32)
a2.set_xticks([52, 131, 262, 393, 525])
tidy(a2)
a2.legend(handles, labels, loc="upper right", fontsize=8, handlelength=2.2)
panel(a2, "b", "Test error against training designs")
fig.tight_layout(w_pad=1.0)
fig.subplots_adjust(left=0.01, bottom=0.14)
save(fig, FIG_DIR / "config_change.pdf")
print("wrote", FIG_DIR / "config_change.pdf")
