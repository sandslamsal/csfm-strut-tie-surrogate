"""Shared figure style for every plot of the paper (vector PDF output).

One palette, one set of rc parameters and one way of labelling panels, so the
figures read as a set. The four archetype colours were validated for
colour-vision safety and contrast against a white page.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

# archetype -> colour (blue, green, orange, red); validated categorical set
COLOUR = {
    "deepBeam": "#2B63A6",
    "hammerhead": "#1F8A70",
    "multiColumnBent": "#D9761A",
    "pileCap": "#B8352B",
}
MARKER = {"deepBeam": "o", "hammerhead": "s", "multiColumnBent": "^", "pileCap": "D"}
LABEL = {"deepBeam": "Deep beam", "hammerhead": "Hammerhead",
         "multiColumnBent": "Multi-column bent", "pileCap": "Pile cap"}
ORDER = ["deepBeam", "hammerhead", "multiColumnBent", "pileCap"]
INK, INK2, GRID = "0.15", "0.40", "0.90"

# every figure is drawn at the text width (7.0 in) so that 1 pt here is 1 pt on the page
TEXTWIDTH = 7.0
RC = {
    "font.family": "STIXGeneral", "font.size": 9, "mathtext.fontset": "stix",
    "axes.linewidth": 0.8, "axes.edgecolor": "0.35", "axes.labelcolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 7.0, "axes.labelsize": 9,
    "xtick.color": "0.35", "ytick.color": "0.35", "xtick.labelcolor": INK,
    "ytick.labelcolor": INK, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.fontsize": 8.5, "legend.frameon": False, "legend.handlelength": 1.8,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "savefig.dpi": 600, "pdf.fonttype": 42, "figure.dpi": 100,
}
plt.rcParams.update(RC)


def tidy(ax, grid="both"):
    """Recessive frame and grid: no top/right spines, faint grid behind data."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, lw=0.45, color=GRID, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, tag, title=""):
    """Bold panel tag and title, left-aligned on the axes, same height everywhere."""
    text = f"({tag}) {title}" if title else f"({tag})"
    ax.set_title(text, loc="left", fontweight="bold", fontsize=10, pad=7, color=INK)


def legend_below(fig, handles, labels, ncol=None, y=-0.02, **kw):
    """One shared legend centred under all panels."""
    ncol = ncol or len(labels)
    opts = dict(loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y),
                fontsize=8.5, handletextpad=0.5, columnspacing=1.6)
    opts.update(kw)
    return fig.legend(handles, labels, **opts)


def save(fig, path, pad=0.03):
    """Write the figure exactly TEXTWIDTH inches wide, cropped tightly only in
    the vertical direction, so that every figure included at the text width
    prints at the same scale and its fonts keep their nominal sizes."""
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    w = fig.get_figwidth()
    if tb.x0 < -0.01 or tb.x1 > w + 0.01:
        print(f"[figstyle] {os.path.basename(path)}: content spans "
              f"{tb.x0:.2f}..{tb.x1:.2f} in, outside 0..{w:.2f}")
    fig.savefig(path, bbox_inches=Bbox([[0.0, tb.y0 - pad], [w, tb.y1 + pad]]))
    print(f"wrote {path}  ({w:.2f} x {tb.y1 - tb.y0 + 2 * pad:.2f} in)")
