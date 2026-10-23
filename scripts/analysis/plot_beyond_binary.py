"""Figure: where a binary reward is blind and the continuous attribution is not.

KodCode groups where all four rollouts fail: within-group centered log p_own by
    within-group rank of the fraction of hidden-test assertions passed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from iar_repro.records import analysis_inputs, materialize  # noqa: E402

GROUP = 4


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]

# Same palette as the other figures: one colour per dataset.
DOMAIN_COLORS = {"commonsenseqa": "#0072B2", "gsm8k": "#E69F00",
                 "kodcode": "#009E73", "spider": "#CC79A7"}
BLUE, ORANGE, AQUA = "#0072B2", "#E69F00", "#56B4E9"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
LABELS = ("CQA", "GSM8K", "KodCode", "Spider")


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def kodcode_rank_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ALIGN = analysis_inputs()
    PARTIAL = materialize("results/partial/kodcode.jsonl.gz")
    correct = {r["call_id"]: int(r["correct"]) for r in read_jsonl(ALIGN / "rows_kodcode_p_own.jsonl")}
    groups = defaultdict(list)
    for task in read_jsonl(PARTIAL):
        for row in task["rows"]:
            if row.get("partial_total"):
                groups[(row["dataset_index"], row["occurrence"])].append(row)
    per_rank = defaultdict(list)
    for g in groups.values():
        if len(g) != GROUP or any(correct[r["call_id"]] for r in g):
            continue
        q = np.array([r["q_partial"] for r in g])
        if len(set(q)) == 1:
            continue
        lp = np.array([r["log_p_own"] for r in g])
        ranks = q.argsort().argsort()
        for rank, value in zip(ranks, lp - lp.mean(), strict=True):
            per_rank[int(rank)].append(value)
    means = np.array([np.mean(per_rank[k]) for k in range(GROUP)])
    rng = np.random.default_rng(23)
    ci = np.array([[np.percentile([rng.choice(per_rank[k], len(per_rank[k])).mean()
                                   for _ in range(2000)], p) for p in (2.5, 97.5)]
                   for k in range(GROUP)])
    return np.arange(1, GROUP + 1), means, ci


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.signal.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 1, figsize=(3.4, 2.4), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    ranks, means, ci = kodcode_rank_curve()
    ax.errorbar(ranks, means, yerr=[means - ci[:, 0], ci[:, 1] - means],
                color=DOMAIN_COLORS["kodcode"], lw=2,
                marker="o", ms=4, capsize=2)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(ranks, ["lowest", "2", "3", "highest"])
    ax.set_xlabel("Rank of hidden-test pass fraction", color=MUTED)
    ax.set_ylabel(r"Centered $\log p_{\rm ind}$", color=MUTED)
    ax.set_title("KodCode all-incorrect groups", fontsize=8, color=INK)
    style(ax)

    fig.tight_layout(w_pad=1.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
