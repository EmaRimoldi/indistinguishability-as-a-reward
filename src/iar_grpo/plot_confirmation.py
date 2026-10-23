from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COLORS = {12: "#6F4E9C", 17: "#2A9D8F"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-seed HPO confirmation")
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(create_confirmation_plots(Path(args.batch_dir)), indent=2))


def create_confirmation_plots(batch_dir: Path) -> dict[str, Any]:
    global plt, np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    batch_dir = batch_dir.expanduser().resolve()
    summary = json.loads(
        (batch_dir / "confirmation_summary.json").read_text(encoding="utf-8")
    )
    candidates = sorted(summary["candidates"], key=lambda row: int(row["trial_number"]))
    if not candidates:
        raise RuntimeError("No confirmation candidates to plot")
    output_dir = batch_dir / "figures"
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.7), constrained_layout=True)
    plot_pooled_lcb(axes[0, 0], candidates)
    plot_seed_pown(axes[0, 1], candidates)
    plot_hard_fooling(axes[1, 0], candidates)
    plot_seed_quality(axes[1, 1], candidates)
    fig.suptitle("Fresh multi-seed confirmation", fontsize=15)
    png = output_dir / "confirmation_overview.png"
    pdf = output_dir / "confirmation_overview.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "schema_version": 1,
        "candidate_trials": [int(row["trial_number"]) for row in candidates],
        "seeds": sorted(
            {int(seed["seed"]) for row in candidates for seed in row["seeds"]}
        ),
        "png": str(png),
        "pdf": str(pdf),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def plot_pooled_lcb(axis: Any, candidates: list[dict[str, Any]]) -> None:
    x = np.arange(len(candidates))
    means = [float(row["pooled_p_own_delta_mean"]) for row in candidates]
    lcbs = [float(row["pooled_p_own_delta_lcb"]) for row in candidates]
    colors = [candidate_color(row) for row in candidates]
    axis.bar(x, means, color=colors, alpha=0.82, label=r"mean $\Delta p_{own}$")
    for index, (mean, lcb) in enumerate(zip(means, lcbs, strict=True)):
        axis.vlines(index, lcb, mean, color="#222222", linewidth=2.2)
        axis.scatter(index, lcb, marker="_", s=150, color="#222222", zorder=3)
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_xticks(x, candidate_labels(candidates))
    axis.set_ylabel("paired validation change")
    axis.set_title("Pooled effect and one-sided 95% LCB")
    axis.grid(axis="y", alpha=0.18)


def plot_seed_pown(axis: Any, candidates: list[dict[str, Any]]) -> None:
    seeds = sorted({int(seed["seed"]) for row in candidates for seed in row["seeds"]})
    x = np.arange(len(seeds))
    width = 0.34
    for index, row in enumerate(candidates):
        by_seed = {int(seed["seed"]): seed for seed in row["seeds"]}
        offset = (index - (len(candidates) - 1) / 2) * width
        axis.bar(
            x + offset,
            [float(by_seed[seed]["p_own_delta"]) for seed in seeds],
            width=width,
            color=candidate_color(row),
            label=f"trial {row['trial_number']}",
        )
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_xticks(x, [str(seed) for seed in seeds])
    axis.set_xlabel("seed")
    axis.set_ylabel(r"$\Delta p_{own}$")
    axis.set_title("Effect by fresh seed")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.18)


def plot_hard_fooling(axis: Any, candidates: list[dict[str, Any]]) -> None:
    x = np.arange(len(candidates))
    width = 0.34
    axis.bar(
        x - width / 2,
        [float(row["mean_sampled_fooling_delta"]) for row in candidates],
        width=width,
        color="#E45756",
        label="sampled fooling",
    )
    axis.bar(
        x + width / 2,
        [float(row["mean_argmax_fooling_delta"]) for row in candidates],
        width=width,
        color="#72B7B2",
        label="argmax fooling",
    )
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_xticks(x, candidate_labels(candidates))
    axis.set_ylabel("mean paired change")
    axis.set_title("Fooling-rate confirmation")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.18)


def plot_seed_quality(axis: Any, candidates: list[dict[str, Any]]) -> None:
    seeds = sorted({int(seed["seed"]) for row in candidates for seed in row["seeds"]})
    x = np.arange(len(seeds))
    width = 0.34
    for index, row in enumerate(candidates):
        by_seed = {int(seed["seed"]): seed for seed in row["seeds"]}
        offset = (index - (len(candidates) - 1) / 2) * width
        axis.bar(
            x + offset,
            [float(by_seed[seed]["pass_at_1_delta"]) for seed in seeds],
            width=width,
            color=candidate_color(row),
            label=f"trial {row['trial_number']}",
        )
    axis.axhline(-0.02, color="#B22222", linestyle="--", linewidth=1, label="quality gate")
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xticks(x, [str(seed) for seed in seeds])
    axis.set_xlabel("seed")
    axis.set_ylabel(r"$\Delta$ pass@1")
    axis.set_title("External quality by seed")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.18)


def candidate_color(row: dict[str, Any]) -> str:
    return COLORS.get(int(row["trial_number"]), "#4C78A8")


def candidate_labels(candidates: list[dict[str, Any]]) -> list[str]:
    return [f"trial {row['trial_number']}" for row in candidates]


if __name__ == "__main__":
    main()
