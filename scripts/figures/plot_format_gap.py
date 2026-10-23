#!/usr/bin/env python3
"""How much of the held-out gain is answer format.

For CommonsenseQA and GSM8K, one stacked bar for A0 and one for Indist.: the solid
part is strict accuracy, the pale part the extra accuracy obtained when responses
rejected for their format are read by their last answer letter or number. Indist.
is the mean of the three seeds. Reads format_gap_data.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from iar_repro.records import results_root

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DOMAINS = [("commonsenseqa", "CommonsenseQA", "#0072B2"), ("gsm8k", "GSM8K", "#E69F00")]
SEEDS = ("identity_s23", "identity_s42", "identity_s7")


def shade(hex_color: str, lighten: float) -> tuple[float, float, float]:
    rgb = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(c + (1.0 - c) * lighten for c in rgb)


def plot(data: dict, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.linewidth": .65})
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.1), sharey=True)
    for ax, (domain, title, color) in zip(axes, DOMAINS):
        d = data[domain]
        strict = [d["a0"]["strict_pct"], np.mean([d[s]["strict_pct"] for s in SEEDS])]
        tolerant = [d["a0"]["tolerant_pct"], np.mean([d[s]["tolerant_pct"] for s in SEEDS])]
        x = np.arange(2)
        ax.bar(x, strict, .6, color=color, edgecolor="white", linewidth=.5)
        ax.bar(x, np.subtract(tolerant, strict), .6, bottom=strict, color=shade(color, .7),
               edgecolor=color, linewidth=.5, hatch="////")
        for xi, s, t in zip(x, strict, tolerant):
            ax.text(xi, t + 1.2, f"+{t - s:.1f}", ha="center", fontsize=7.5, fontweight="bold")
        ax.set_xticks(x, ["$A_0$", "Indist."])
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_ylim(40, 80)
        ax.tick_params(labelsize=7.5, length=2.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out accuracy (%)", fontweight="bold", fontsize=8)
    guide = [Patch(facecolor="#6f6f6f", label="Strict parser"),
             Patch(facecolor=shade("#6f6f6f", .7), edgecolor="#6f6f6f", hatch="////",
                   label="Recovered by reading the answer leniently")]
    fig.legend(handles=guide, loc="upper center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(.53, 1.06), handlelength=1.3, columnspacing=1.2)
    fig.subplots_adjust(left=.14, right=.99, bottom=.12, top=.8, wspace=.12)
    fig.savefig(output, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=results_root() / "figure_data/format_gap_data.json")
    parser.add_argument("--output", type=Path, default=here / "format_gap.pdf")
    args = parser.parse_args()
    plot(json.loads(args.data.read_text()), args.output)


if __name__ == "__main__":
    main()
