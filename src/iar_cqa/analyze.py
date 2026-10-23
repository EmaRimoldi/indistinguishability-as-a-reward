from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .data import read_jsonl, write_json
from .scoring import accuracy


SURFACE_FEATURES = (
    "chars",
    "words",
    "lines",
    "sentences",
    "bullets",
    "n_mentioned_option_letters",
    "repeated_choice_lines",
    "commas",
    "semicolons",
    "parentheses",
)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(np.corrcoef(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))[0, 1])


def candidate_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return accuracy(rows)


def baseline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "B_own": [row for row in rows if row["source"] == "B_own"],
        "A0_plain": [row for row in rows if row["source"] == "A0" and row["variant"] == "plain"],
        "A0_imitation": [row for row in rows if row["source"] == "A0" and row["variant"] == "imitation"],
    }
    result: dict[str, Any] = {"n": len(rows), "groups": {}}
    own_scores = [float(row["p_own"]) for row in groups["B_own"]]
    for name, values in groups.items():
        pown = [float(row["p_own"]) for row in values]
        emitted = [row.get("judge_token") for row in values]
        result["groups"][name] = {
            "accuracy": candidate_accuracy(values),
            "mean_p_own": mean(pown),
            "argmax_acceptance": mean(float(value > 0.5) for value in pown),
            "sampled_acceptance": mean(float(value == "1") for value in emitted),
            "emitted_acceptance": mean(float(value == "1") for value in emitted),
            "reward": {
                "mean": mean(float(row["reward"]) for row in values),
                "min": min(float(row["reward"]) for row in values),
                "max": max(float(row["reward"]) for row in values),
            },
        }
        if name != "B_own":
            result["groups"][name]["gap_S_B_minus_S_A"] = mean(own_scores) - mean(pown)
            comparison = groups["B_own"] + values
            result["groups"][name]["attribution_auc"] = auc(
                [float(row["p_own"]) for row in comparison],
                [row["source"] == "B_own" for row in comparison],
            )
    result["protocol_audit"] = protocol_audit(rows)
    return result


def protocol_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    masses = [float(row["binary_label_mass"]) for row in rows]
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
    return {
        "n": len(rows),
        "models": sorted({row.get("raw_model") for row in rows}),
        "providers": sorted({row.get("raw_provider") for row in rows}),
        "prompt_versions": sorted({row.get("judge_prompt_version") for row in rows}),
        "prompt_hashes": sorted({row.get("judge_prompt_template_sha256") for row in rows}),
        "message_roles": sorted({tuple(row.get("judge_message_roles", [])) for row in rows}),
        "system_prompt_present": sorted({row.get("judge_system_prompt_present") for row in rows}),
        "emitted_argmax_consistency": mean(float(row.get("emitted_argmax_consistent")) for row in rows),
        "binary_label_mass": {"min": min(masses), "mean": mean(masses), "max": max(masses)},
        "judge_token_counts": dict(
            Counter(
                str(row.get("judge_token"))
                if row.get("judge_token") is not None
                else "INVALID"
                for row in rows
            )
        ),
        "usage": {
            key: sum(float((row.get("judge_usage") or {}).get(key, 0) or 0) for row in rows)
            for key in usage_keys
        },
        "mean_judge_latency_seconds": mean(float(row.get("judge_latency_seconds", 0)) for row in rows),
    }


def feature_audit(rows: list[dict[str, Any]], *, within_task: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    rewards = [float(row["reward"]) for row in rows]
    group_means: dict[str, dict[str, float]] = {}
    if within_task:
        by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_id[row["id"]].append(row)
        for identifier, values in by_id.items():
            group_means[identifier] = {
                "reward": float(np.mean([float(row["reward"]) for row in values])),
                **{
                    feature: float(np.mean([float(row["features"][feature]) for row in values]))
                    for feature in SURFACE_FEATURES
                },
            }
    for feature in SURFACE_FEATURES:
        values = [float(row["features"][feature]) for row in rows]
        item = {"correlation_with_reward": correlation(values, rewards)}
        if within_task:
            centered_feature = [value - group_means[row["id"]][feature] for value, row in zip(values, rows)]
            centered_reward = [float(row["reward"]) - group_means[row["id"]]["reward"] for row in rows]
            item["within_task_correlation_with_reward"] = correlation(centered_feature, centered_reward)
        result[feature] = item
    return result


def selected_answer_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    result: dict[str, Any] = {}
    for label in ("A", "B", "C", "D", "E", None):
        indicators = [float(row["score"]["prediction"] == label) for row in rows]
        rewards = [float(row["reward"]) for row in rows]
        centered_indicators = [
            value - float(np.mean([other["score"]["prediction"] == label for other in by_id[row["id"]]]))
            for value, row in zip(indicators, rows)
        ]
        centered_rewards = [
            reward - float(np.mean([other["reward"] for other in by_id[row["id"]]]))
            for reward, row in zip(rewards, rows)
        ]
        name = "invalid" if label is None else label
        result[name] = {
            "n": sum(indicators),
            "mean_reward": mean(reward for reward, indicator in zip(rewards, indicators) if indicator),
            "within_task_indicator_correlation_with_reward": correlation(centered_indicators, centered_rewards),
        }
    return result


def probe_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    distinct_counts: Counter[int] = Counter()
    variances: list[float] = []
    mixed_groups = 0
    top_correct_credit = 0.0
    pairs = 0
    correct_wins = 0.0
    for values in by_id.values():
        rewards = [float(row["reward"]) for row in values]
        distinct_counts[len(set(round(value, 6) for value in rewards))] += 1
        variances.append(float(np.var(rewards)))
        correct = [row for row in values if row["score"]["correct"]]
        incorrect = [row for row in values if not row["score"]["correct"]]
        if correct and incorrect:
            mixed_groups += 1
            maximum = max(rewards)
            top = [row for row in values if float(row["reward"]) == maximum]
            top_correct_credit += sum(row["score"]["correct"] for row in top) / len(top)
            for correct_row in correct:
                for incorrect_row in incorrect:
                    cr = float(correct_row["reward"])
                    ir = float(incorrect_row["reward"])
                    correct_wins += float(cr > ir) + 0.5 * float(cr == ir)
                    pairs += 1
    correct_rows = [row for row in rows if row["score"]["correct"]]
    incorrect_rows = [row for row in rows if not row["score"]["correct"]]
    answer_labels = sorted({row["score"]["prediction"] for row in rows}, key=lambda value: str(value))
    return {
        "n": len(rows),
        "accuracy": candidate_accuracy(rows),
        "reward": {
            "mean": mean(float(row["reward"]) for row in rows),
            "std": float(np.std([float(row["reward"]) for row in rows])),
            "min": min(float(row["reward"]) for row in rows),
            "max": max(float(row["reward"]) for row in rows),
        },
        "p_own": {
            "mean": mean(float(row["p_own"]) for row in rows),
            "argmax_fooling": mean(float(float(row["p_own"]) > 0.5) for row in rows),
            "sampled_fooling": mean(float(row.get("judge_token") == "1") for row in rows),
            "emitted_fooling": mean(float(row.get("judge_token") == "1") for row in rows),
        },
        "group_signal": {
            "n_groups": len(by_id),
            "complete_k4_groups": sum(len(values) == 4 for values in by_id.values()),
            "distinct_reward_count_distribution": dict(sorted(distinct_counts.items())),
            "groups_with_at_least_two_distinct_rewards": sum(count for distinct, count in distinct_counts.items() if distinct >= 2),
            "mean_within_group_variance": mean(variances),
            "median_within_group_variance": float(np.median(variances)),
        },
        "correctness_relation": {
            "n_correct": len(correct_rows),
            "n_incorrect": len(incorrect_rows),
            "mean_reward_correct": mean(float(row["reward"]) for row in correct_rows),
            "mean_reward_incorrect": mean(float(row["reward"]) for row in incorrect_rows),
            "reward_auc_for_correctness": auc(
                [float(row["reward"]) for row in rows],
                [bool(row["score"]["correct"]) for row in rows],
            ),
            "mixed_groups": mixed_groups,
            "top_reward_correct_rate_mixed_groups": top_correct_credit / mixed_groups if mixed_groups else None,
            "correct_beats_incorrect_pairwise_rate": correct_wins / pairs if pairs else None,
            "n_correct_incorrect_pairs": pairs,
        },
        "mean_reward_by_predicted_answer": {
            str(label): mean(float(row["reward"]) for row in rows if row["score"]["prediction"] == label)
            for label in answer_labels
        },
        "feature_audit": feature_audit(rows, within_task=True),
        "correct_only_feature_audit": feature_audit(correct_rows, within_task=True) if correct_rows else {},
        "selected_answer_audit": selected_answer_audit(rows),
        "correct_only_selected_answer_audit": selected_answer_audit(correct_rows) if correct_rows else {},
        "protocol_audit": protocol_audit(rows),
    }


def go_no_go(baseline: dict[str, Any], probe: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    own = baseline["groups"]["B_own"]
    plain = baseline["groups"]["A0_plain"]
    feature_correlations = {
        feature: values.get("within_task_correlation_with_reward")
        for feature, values in probe["feature_audit"].items()
    }
    feature_correlations.update(
        {
            f"selected_answer_{label}": values.get("within_task_indicator_correlation_with_reward")
            for label, values in probe["selected_answer_audit"].items()
        }
    )
    finite_correlations = [
        abs(value)
        for value in feature_correlations.values()
        if value is not None and math.isfinite(value)
    ]
    checks = {
        "B_accuracy_exceeds_A0_plain": own["accuracy"]["accuracy"] - plain["accuracy"]["accuracy"]
        >= float(gates["minimum_b_minus_a_accuracy"]),
        "S_B_exceeds_S_A_plain": own["mean_p_own"] > plain["mean_p_own"],
        "plain_attribution_auc": plain["attribution_auc"] >= float(gates["minimum_attribution_auc"]),
        "reward_varies_within_K4_groups": probe["group_signal"]["groups_with_at_least_two_distinct_rewards"]
        >= int(gates["minimum_groups_with_reward_variation"]),
        "correct_rollouts_beat_incorrect_rollouts": (
            probe["correctness_relation"]["correct_beats_incorrect_pairwise_rate"] or 0.0
        )
        >= float(gates["minimum_correct_beats_incorrect_rate"]),
        "no_dominant_measured_surface_shortcut": (
            max(finite_correlations) if finite_correlations else 0.0
        )
        <= float(gates["maximum_absolute_within_task_surface_correlation"]),
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "n_passed": sum(checks.values()),
        "n_checks": len(checks),
        "thresholds": gates,
        "maximum_observed_absolute_within_task_surface_correlation": max(finite_correlations)
        if finite_correlations
        else None,
        "surface_correlations": feature_correlations,
        "protocol_diagnostics_not_used_as_gates": {
            "minimum_binary_label_mass": min(
                baseline["protocol_audit"]["binary_label_mass"]["min"],
                probe["protocol_audit"]["binary_label_mass"]["min"],
            ),
            "baseline_emitted_argmax_consistency": baseline["protocol_audit"][
                "emitted_argmax_consistency"
            ],
            "probe_emitted_argmax_consistency": probe["protocol_audit"][
                "emitted_argmax_consistency"
            ],
        },
    }


def save_plots(
    run_dir: Path,
    baseline_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    probe: dict[str, Any],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    colors = {"B_own": "#0072B2", "A0_plain": "#D55E00", "A0_imitation": "#009E73"}
    labels = {"B_own": r"$B$ own", "A0_plain": r"$A_0$ plain", "A0_imitation": r"$A_0$ imitation"}
    files: list[str] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for group in ("B_own", "A0_plain", "A0_imitation"):
        values = [float(row["p_own"]) for row in baseline_rows if (
            (group == "B_own" and row["source"] == "B_own")
            or (group != "B_own" and row["source"] == "A0" and row["variant"] == group.removeprefix("A0_"))
        )]
        ax.hist(values, bins=np.linspace(0, 1, 21), density=True, histtype="step", linewidth=2.2,
                color=colors[group], label=labels[group])
    ax.set(xlabel=r"Self-attribution score $p_{\rm own}$", ylabel="Density",
           title="Pointwise self-attribution on the reward audit")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = plot_dir / f"attribution_score_distributions.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(str(path.relative_to(run_dir)))
    plt.close(fig)

    correct = [float(row["reward"]) for row in probe_rows if row["score"]["correct"]]
    incorrect = [float(row["reward"]) for row in probe_rows if not row["score"]["correct"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    parts = ax.violinplot([incorrect, correct], positions=[0, 1], showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], ("#D55E00", "#009E73")):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)
    parts["cmeans"].set_color("#222222")
    ax.set_xticks([0, 1], [f"Incorrect\n(n={len(incorrect)})", f"Correct\n(n={len(correct)})"])
    ax.set(ylabel=r"Reward $\log P_B(1\mid x,y)$", title="Reward versus external answer correctness")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = plot_dir / f"probe_reward_by_correctness.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(str(path.relative_to(run_dir)))
    plt.close(fig)

    features = list(SURFACE_FEATURES) + [f"selected_answer_{label}" for label in ("A", "B", "C", "D", "E", "invalid")]
    overall = [
        probe["feature_audit"][name].get("within_task_correlation_with_reward")
        if name in probe["feature_audit"]
        else probe["selected_answer_audit"][name.removeprefix("selected_answer_")].get("within_task_indicator_correlation_with_reward")
        for name in features
    ]
    correct_only = [
        probe["correct_only_feature_audit"].get(name, {}).get("within_task_correlation_with_reward")
        if name in SURFACE_FEATURES
        else probe["correct_only_selected_answer_audit"].get(name.removeprefix("selected_answer_"), {}).get("within_task_indicator_correlation_with_reward")
        for name in features
    ]
    overall = [0.0 if value is None or not math.isfinite(value) else value for value in overall]
    correct_only = [0.0 if value is None or not math.isfinite(value) else value for value in correct_only]
    positions = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(positions - 0.19, overall, width=0.38, color="#0072B2", label="All rollouts")
    ax.bar(positions + 0.19, correct_only, width=0.38, color="#CC79A7", label="Correct only")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axhline(0.4, color="#999999", linewidth=0.8, linestyle="--")
    ax.axhline(-0.4, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions, [name.replace("_", " ") for name in features], rotation=35, ha="right")
    ax.set(ylabel="Within-question correlation with reward", title="Measured shortcut audit")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = plot_dir / f"shortcut_correlations.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(str(path.relative_to(run_dir)))
    plt.close(fig)

    answer_means = probe["mean_reward_by_predicted_answer"]
    answer_labels = [label for label in "ABCDE" if label in answer_means]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(answer_labels, [answer_means[label] for label in answer_labels], color="#56B4E9")
    ax.set(xlabel="Predicted answer", ylabel="Mean reward", title="Reward by selected option")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = plot_dir / f"reward_by_selected_option.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(str(path.relative_to(run_dir)))
    plt.close(fig)

    manifest = {
        "files": files,
        "reward": "absolute log-probability log P_B(label=1 | task, candidate)",
        "external_quality": "exact match of the final Answer: X line to the frozen answer key",
    }
    write_json(plot_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    baseline_rows = read_jsonl(run_dir / "baseline_judgments.jsonl")
    probe_rows = read_jsonl(run_dir / "probe_judgments.jsonl")
    baseline = baseline_summary(baseline_rows)
    probe = probe_summary(probe_rows)
    decision = go_no_go(baseline, probe, config["gates"])
    plots = save_plots(run_dir, baseline_rows, probe_rows, probe)
    summary = {"baseline": baseline, "probe": probe, "go_no_go": decision, "plots": plots}
    write_json(run_dir / "phase0_summary.json", summary)
    write_json(run_dir / "go_no_go.json", decision)
    print(json.dumps({
        "baseline_accuracy": {name: value["accuracy"]["accuracy"] for name, value in baseline["groups"].items()},
        "baseline_p_own": {name: value["mean_p_own"] for name, value in baseline["groups"].items()},
        "probe_accuracy": probe["accuracy"]["accuracy"],
        "probe_group_signal": probe["group_signal"],
        "probe_correctness_relation": probe["correctness_relation"],
        "go_no_go": decision,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
