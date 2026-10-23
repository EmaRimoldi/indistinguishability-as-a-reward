"""One visual style for every data figure of the paper.

Every figure is drawn at the width it is printed with (ICLR text width 5.5 in, no
rescaling by \\includegraphics), so the sizes below are the sizes on the page:
Times text with Computer Modern math as in the manuscript, 8 pt labels and titles,
7 pt ticks and legends, 0.6 pt axes, 1.2 pt data lines.

Colour. Each domain has one pastel hue: FILL for areas and bars, and LINE, a
darker tone of the same hue, for lines, markers, edges and hatching. The LINE
tones pass a colour-vision-deficiency check on every pair (protanopia and
deuteranopia, Machado et al. 2009: Delta E >= 9.6 in OKLab x 100), reach 3:1
contrast on white, and are spread in lightness (CIE L* 32, 42, 53, 61) so the
domains stay apart in grayscale print. LIGHT (LINE half-way to white) marks the
incorrect responses of a domain.

Training conditions never take a colour: Indist. is solid, Quality horizontal
hatching / dashed / triangle, Oracle diagonal hatching / dotted / diamond.
"""

from __future__ import annotations

from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

TEXT_WIDTH = 5.5  # inches, ICLR \textwidth
RASTER_DIR = Path(os.environ.get("IAR_FIGURE_RASTER_DIR", Path(__file__).resolve().parents[2] / "runs/figures/png"))
RASTER_DPI = 600

DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
LABEL = {"commonsenseqa": "CommonsenseQA", "gsm8k": "GSM8K", "kodcode": "KodCode", "spider": "Spider"}
SHORT = {"commonsenseqa": "CQA", "gsm8k": "GSM8K", "kodcode": "KodCode", "spider": "Spider"}
LINE = {"commonsenseqa": "#2863AB", "gsm8k": "#C8853D", "kodcode": "#018E7D", "spider": "#73336F"}
FILL = {"commonsenseqa": "#C2DAF9", "gsm8k": "#F8D7B9", "kodcode": "#B8E6DC", "spider": "#E8C9E4"}

INK = "#1F1F1F"  # text and axes
MUTED = "#5C5C5C"  # secondary text, reference lines
GRID = "#E3E3E3"
KEY_DARK, KEY_LIGHT, KEY_FILL = "#4D4D4D", "#A6A6A6", "#DCDCDC"  # legend keys valid for every domain

CONDITIONS = ("grpo_identity", "grpo_quality", "grpo_oracle")
CONDITION = {"grpo_identity": "Indist.", "grpo_quality": "Quality", "grpo_oracle": "Oracle"}
HATCH = {"grpo_identity": None, "grpo_quality": "---", "grpo_oracle": "////"}
DASH = {"grpo_identity": "solid", "grpo_quality": (0, (3.6, 1.6)), "grpo_oracle": (0, (1.0, 1.4))}
MARKER = {"grpo_identity": "o", "grpo_quality": "^", "grpo_oracle": "D"}
TEACHER_DASH = (0, (7.0, 2.6))  # reference: the teacher's own completions

MEAN_LW, SEED_LW, SEED_ALPHA = 1.2, 0.5, 0.35
REF_LW, ERR_LW = 0.6, 0.8
MARKER_SIZE = 3.2

STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.titlepad": 4.0,
    "axes.labelpad": 2.0,
    "lines.linewidth": MEAN_LW,
    "lines.markersize": MARKER_SIZE,
    "lines.markeredgewidth": 0.5,
    "patch.linewidth": 0.6,
    "hatch.linewidth": 0.5,
    "errorbar.capsize": 1.6,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.2,
    "legend.borderpad": 0.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": RASTER_DPI,
    "figure.dpi": 150,
}


def light(hex_color: str, amount: float = 0.5) -> str:
    """``hex_color`` moved ``amount`` of the way to white."""
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02X}" for c in rgb)


LIGHT = {domain: light(color) for domain, color in LINE.items()}


def subplots(nrows: int = 1, ncols: int = 1, *, width: float = 1.0, height: float, **kwargs):
    """Figure at its printed size: ``width`` as a fraction of the text width, ``height`` in inches."""
    mpl.rcParams.update(STYLE)
    fig, axes = plt.subplots(nrows, ncols, figsize=(TEXT_WIDTH * width, height),
                             layout="constrained", **kwargs)
    fig.get_layout_engine().set(w_pad=1.5 / 72, h_pad=1.5 / 72, wspace=0.02, hspace=0.02)
    return fig, axes


def legend_top(fig, handles: list, ncol: int | None = None, **kwargs) -> None:
    """One frameless legend row above all panels."""
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol or len(handles), **kwargs)


def domain_keys(labels: dict[str, str] = LABEL, marker: str | None = "o") -> list[Line2D]:
    return [Line2D([0], [0], color=LINE[d], lw=MEAN_LW, marker=marker, markersize=MARKER_SIZE,
                   markeredgecolor="white", label=labels[d]) for d in DOMAINS]


def condition_line_keys(conditions: tuple[str, ...] = CONDITIONS) -> list[Line2D]:
    return [Line2D([0], [0], color=KEY_DARK, lw=MEAN_LW, ls=DASH[c], marker=MARKER[c],
                   markersize=MARKER_SIZE, markeredgecolor="white", label=CONDITION[c])
            for c in conditions]


def condition_bar_keys(conditions: tuple[str, ...] = CONDITIONS) -> list[Patch]:
    return [Patch(facecolor=KEY_FILL, edgecolor=KEY_DARK, hatch=HATCH[c], linewidth=0.6,
                  label=CONDITION[c]) for c in conditions]


def teacher_key() -> Line2D:
    return Line2D([0], [0], color=MUTED, lw=0.9, ls=TEACHER_DASH, label="Teacher")


def label_ends(ax, ends: dict[str, tuple[float, float]], min_gap: float = 7.5) -> None:
    """Direct labels right of each series' last point (x, y), spaced at least ``min_gap``
    points apart; a hairline leads to the point when a label had to move. Keeps the series
    identifiable in grayscale print."""
    fig = ax.figure
    fig.canvas.draw()  # settle the layout first: data -> display depends on it
    items = sorted(((ax.transData.transform(xy)[1], label, xy) for label, xy in ends.items()),
                   reverse=True)
    gap = min_gap * fig.dpi / 72
    ys = [items[0][0]]  # the top label stays at its point, the others only move down
    for y, _, _ in items[1:]:
        ys.append(min(y, ys[-1] - gap))
    for new, (old, label, xy) in zip(ys, items):
        dy = (new - old) * 72 / fig.dpi
        leader = {"arrowstyle": "-", "color": MUTED, "lw": 0.4, "shrinkA": 1.0, "shrinkB": 2.0,
                  "relpos": (0.0, 0.5)}
        ax.annotate(label, xy=xy, xytext=(10, dy), textcoords="offset points", ha="left",
                    va="center", fontsize=7, color=INK, annotation_clip=False,
                    arrowprops=leader if abs(dy) > 1.5 else None)


def save(fig, pdf: Path) -> None:
    """Vector PDF for the paper plus a 600 dpi PNG, both at the printed size."""
    fig.savefig(pdf)
    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RASTER_DIR / f"{pdf.stem}.png", dpi=RASTER_DPI, facecolor="white")
    plt.close(fig)
