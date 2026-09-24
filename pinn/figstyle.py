"""Shared figure style for every plot of the paper (vector PDF output).

One palette, one set of rc parameters and one way of labelling panels, so the
figures read as a set. The four archetype colours were validated for
colour-vision safety and contrast against a white page.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

RC = {
    "font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
    "axes.linewidth": 0.8, "axes.edgecolor": "0.35", "axes.labelcolor": INK,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 7.0, "axes.labelsize": 9,
    "xtick.color": "0.35", "ytick.color": "0.35", "xtick.labelcolor": INK,
    "ytick.labelcolor": INK, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.fontsize": 7.8, "legend.frameon": False, "legend.handlelength": 1.8,
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
    ax.set_title(text, loc="left", fontweight="bold", fontsize=9.5, pad=7, color=INK)


def legend_below(fig, handles, labels, ncol=None, y=-0.02):
    """One shared legend centred under all panels."""
    ncol = ncol or len(labels)
    return fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
                      bbox_to_anchor=(0.5, y), fontsize=8, handletextpad=0.5,
                      columnspacing=1.6)
