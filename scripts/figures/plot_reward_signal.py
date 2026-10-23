#!/usr/bin/env python3
"""How much signal each reward gives GRPO, and whether it points the right way.

Three panels built from the same data and computations as the separate figures
they replace (plot_bars in plot_reward_contrast.py, the KodCode panel of
scripts/revision_20260922/plot_beyond_binary.py, and plot_correctness_deciles.py):

(a) groups pulled toward a correct response net of those pulled away: the
    share of the 2,000 four-response groups of a run whose rewards are not all
    equal, times (2 x direction - 1), where direction is the share of mixed
    groups in which the reward ranks the correct responses higher;
(b) KodCode groups in which all four programs fail the hidden tests:
    within-group centered log p_ind by the within-group rank of the fraction of
    hidden-test assertions passed, with 95% bootstrap intervals;
(c) share of correct Indist. training rollouts by decile of p_ind, one thin
    line per seed under the bold mean.

Reads reward_contrast_data.json, correctness_deciles_seeds.json and, for panel
(b), the archived rollout and partial-execution records through rollout_signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from iar_repro.records import results_root

import matplotlib

matplotlib.use("Agg")
import figstyle as fs
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from plot_beyond_binary import kodcode_rank_curve  # noqa: E402

DOMAINS = list(fs.DOMAINS)
METHODS = [(c, fs.CONDITION[c]) for c in fs.CONDITIONS]


def plot(data: dict, deciles: dict, output: Path) -> None:
    fig, axes = fs.subplots(1, 3, height=2.2, gridspec_kw={"width_ratios": [1.38, .95, .92]})

    # (a) one colour per domain; the condition is the texture of the bar.
    x = np.arange(len(DOMAINS))
    width = 0.26
    for index, (condition, _) in enumerate(METHODS):
        net = []
        for domain in DOMAINS:
            entry = data["domains"][domain][condition]
            share = entry["contrast"]["mean"]
            direction = entry["direction"]["mean"] / 100.0
            net.append(share * (2 * direction - 1))
        axes[0].bar(x + (index - 1) * width, net, width * .86,
                    color=[fs.FILL[d] for d in DOMAINS], edgecolor=[fs.LINE[d] for d in DOMAINS],
                    hatch=fs.HATCH[condition], linewidth=.6)
    axes[0].set_ylabel("Net correct-favoring groups (%)")
    axes[0].set_title("(a) Direction of the GRPO signal")
    axes[0].set_xticks(x, [fs.SHORT[d] for d in DOMAINS])
    axes[0].tick_params(axis="x", length=0)
    axes[0].set_ylim(0, 105)

    ax = axes[1]
    ranks, means, ci = kodcode_rank_curve()
    ax.axhline(0, color=fs.MUTED, lw=fs.REF_LW)
    ax.errorbar(ranks, means, yerr=[means - ci[:, 0], ci[:, 1] - means],
                color=fs.LINE["kodcode"], lw=fs.MEAN_LW, marker=fs.MARKER["grpo_identity"],
                markeredgecolor="white", elinewidth=fs.ERR_LW, capthick=fs.ERR_LW)
    ax.set_xticks(ranks, ["lowest", "2", "3", "highest"])
    ax.set_xlabel("Rank of hidden-test pass fraction")
    ax.set_ylabel(r"Group-centered $\log p_{\mathrm{ind}}$")
    ax.set_title("(b) All-fail KodCode groups")

    ax = axes[2]
    ends = {}
    for domain in DOMAINS:
        per_seed = np.array([[r["correct_rate"] for r in deciles["deciles"][domain][s]]
                             for s in deciles["seeds"]])
        bins = np.arange(1, per_seed.shape[1] + 1)
        for curve in per_seed:
            ax.plot(bins, 100 * curve, color=fs.LINE[domain], lw=fs.SEED_LW, alpha=fs.SEED_ALPHA,
                    zorder=2)
        ax.plot(bins, 100 * per_seed.mean(axis=0), color=fs.LINE[domain], lw=fs.MEAN_LW,
                marker=fs.MARKER["grpo_identity"], markeredgecolor="white", zorder=3)
        ends[fs.SHORT[domain]] = (float(bins[-1]), float(100 * per_seed.mean(axis=0)[-1]))
    ax.set_xticks([1, 4, 7, 10])
    ax.set_xlim(.6, 10.4)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Decile of $p_{\mathrm{ind}}$")
    ax.set_ylabel("Correct rollouts (%)")
    ax.set_title("(c) Correctness vs. reward")

    fig.legend(handles=fs.condition_bar_keys(), loc="outside upper left", ncol=3,
               handlelength=1.6)
    fs.label_ends(axes[2], ends)
    fs.save(fig, output)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=results_root() / "figure_data/reward_contrast_data.json")
    parser.add_argument("--deciles", type=Path, default=results_root() / "figure_data/correctness_deciles_seeds.json")
    parser.add_argument("--output", type=Path, default=here / "reward_signal.pdf")
    args = parser.parse_args()
    plot(json.loads(args.data.read_text()), json.loads(args.deciles.read_text()), args.output)


if __name__ == "__main__":
    main()
