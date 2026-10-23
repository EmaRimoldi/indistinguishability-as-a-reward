from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COLORS = {
    "initial": "#4C78A8",
    "tpe": "#F58518",
    "refinement": "#54A24B",
    "other": "#9D9D9D",
    "p_own": "#6F4E9C",
    "sampled": "#E45756",
    "argmax": "#72B7B2",
    "quality": "#FF9DA6",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the auditable GRPO HPO study")
    parser.add_argument("--study-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(create_hpo_plots(Path(args.study_dir)), indent=2))


def create_hpo_plots(study_dir: Path) -> dict[str, Any]:
    global plt, np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    study_dir = study_dir.expanduser().resolve()
    rows = json.loads((study_dir / "analysis" / "trials.json").read_text(encoding="utf-8"))
    complete = sorted(
        (row for row in rows if row.get("status") == "complete"),
        key=lambda row: int(row["trial_number"]),
    )
    if not complete:
        raise RuntimeError("No complete trials to plot")
    output_dir = study_dir / "analysis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    plot_lcb_sequence(axes[0, 0], complete)
    plot_top_deltas(axes[0, 1], complete)
    plot_soft_quality_tradeoff(axes[1, 0], complete)
    cost_rows = sorted(
        (
            row
            for row in rows
            if row.get("trial_number") is not None
            and row.get("total_openrouter_cost") is not None
        ),
        key=lambda row: int(row["trial_number"]),
    )
    plot_cumulative_cost(axes[1, 1], cost_rows)
    fig.suptitle(r"Short-trial GRPO search: reward $\log p_{\rm own}$", fontsize=15)

    png_path = output_dir / "hpo_search_overview.png"
    pdf_path = output_dir / "hpo_search_overview.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "schema_version": 1,
        "complete_trial_count": len(complete),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "note": "Loss is intentionally omitted; selection uses paired validation LCB.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def plot_lcb_sequence(axis: Any, rows: list[dict[str, Any]]) -> None:
    axis.axhline(0, color="#555555", linewidth=1, linestyle="--")
    for group in ["initial", "tpe", "refinement", "other"]:
        selected = [row for row in rows if source_group(row.get("source")) == group]
        if not selected:
            continue
        axis.scatter(
            [int(row["trial_number"]) for row in selected],
            [float(row["objective_value"]) for row in selected],
            s=52,
            color=COLORS[group],
            label=group,
            zorder=3,
        )
    best = max(rows, key=lambda row: float(row["objective_value"]))
    axis.annotate(
        f"best: {best['trial_number']}",
        (int(best["trial_number"]), float(best["objective_value"])),
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=9,
    )
    axis.set_title("Registered objective")
    axis.set_xlabel("Optuna trial")
    axis.set_ylabel(r"paired 95% LCB of $\Delta p_{\rm own}$")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.18)


def plot_top_deltas(axis: Any, rows: list[dict[str, Any]]) -> None:
    top = sorted(rows, key=lambda row: float(row["objective_value"]), reverse=True)[:8]
    top.reverse()
    y = np.arange(len(top))
    height = 0.18
    series = [
        ("p_own_delta", r"$\Delta p_{own}$", COLORS["p_own"]),
        ("hard_fooling_delta", "sampled fooling", COLORS["sampled"]),
        ("argmax_fooling_delta", "argmax fooling", COLORS["argmax"]),
        ("pass_at_1_delta", "pass@1", COLORS["quality"]),
    ]
    for index, (key, label, color) in enumerate(series):
        offset = (index - 1.5) * height
        axis.barh(
            y + offset,
            [float(row[key]) for row in top],
            height=height,
            label=label,
            color=color,
        )
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_yticks(y, [str(row["trial_number"]) for row in top])
    axis.set_xlabel("paired validation change")
    axis.set_ylabel("trial")
    axis.set_title("Leading trials: effect directions")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    axis.grid(axis="x", alpha=0.18)


def plot_soft_quality_tradeoff(axis: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        group = source_group(row.get("source"))
        axis.scatter(
            float(row["p_own_delta"]),
            float(row["pass_at_1_delta"]),
            s=44,
            color=COLORS[group],
            alpha=0.9,
        )
        if float(row["objective_value"]) > 0:
            axis.annotate(
                str(row["trial_number"]),
                (float(row["p_own_delta"]), float(row["pass_at_1_delta"])),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axis.axhline(-0.02, color="#B22222", linewidth=1, linestyle="--", label="quality gate")
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel(r"mean $\Delta p_{\rm own}$")
    axis.set_ylabel(r"$\Delta$ pass@1")
    axis.set_title("Indistinguishability–quality trade-off")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.18)


def plot_cumulative_cost(axis: Any, rows: list[dict[str, Any]]) -> None:
    trials = [int(row["trial_number"]) for row in rows]
    costs = [float(row.get("total_openrouter_cost") or 0.0) for row in rows]
    cumulative = np.cumsum(costs)
    axis.step(trials, cumulative, where="post", color="#4C78A8", linewidth=2)
    axis.scatter(trials, cumulative, color="#4C78A8", s=20)
    axis.set_xlabel("Optuna trial")
    axis.set_ylabel("cumulative OpenRouter cost (USD)")
    axis.set_title("Measured API cost")
    axis.grid(alpha=0.18)


def source_group(source: Any) -> str:
    value = str(source or "").lower()
    if value.startswith("initial"):
        return "initial"
    if value.startswith("tpe"):
        return "tpe"
    if value.startswith("refinement"):
        return "refinement"
    return "other"


if __name__ == "__main__":
    main()
