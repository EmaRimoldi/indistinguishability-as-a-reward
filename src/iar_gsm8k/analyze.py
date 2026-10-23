from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .data import read_jsonl, write_json
from .scoring import accuracy, response_features, score_response


SURFACE_FEATURES = (
    "chars",
    "words",
    "lexical_diversity",
    "mean_word_length",
    "lines",
    "sentences",
    "numbered_steps",
    "equation_lines",
    "equals_signs",
    "arithmetic_operators",
    "commas",
    "semicolons",
    "punctuation_marks",
    "markdown_markers",
    "parentheses",
    "currency_symbols",
    "conclusion_markers",
    "answer_line_count",
    "answer_is_final_nonempty_line",
    "repeats_question",
)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    value = float(np.corrcoef(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))[0, 1])
    return value if math.isfinite(value) else None


def paired_lcb(own: list[dict[str, Any]], candidate: list[dict[str, Any]], z: float) -> dict[str, Any]:
    own_by_id = {row["id"]: row for row in own}
    candidate_by_id = {row["id"]: row for row in candidate}
    if own_by_id.keys() != candidate_by_id.keys():
        raise ValueError("B-own and candidate IDs are not paired")
    deltas = [
        float(own_by_id[key]["p_own"]) - float(candidate_by_id[key]["p_own"])
        for key in own_by_id
    ]
    delta_mean = statistics.mean(deltas)
    standard_error = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    return {
        "n": len(deltas),
        "mean_delta": delta_mean,
        "standard_error": standard_error,
        "lcb_95_one_sided": delta_mean - z * standard_error,
    }


def protocol_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    masses = [float(row["binary_label_mass"]) for row in rows]
    logps = [float(row[key]) for row in rows for key in ("logp_0", "logp_1")]
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
    return {
        "n": len(rows),
        "models": sorted({str(row.get("raw_model")) for row in rows}),
        "providers": sorted({str(row.get("raw_provider")) for row in rows}),
        "prompt_versions": sorted({str(row.get("judge_prompt_version")) for row in rows}),
        "prompt_hashes": sorted({str(row.get("judge_prompt_template_sha256")) for row in rows}),
        "message_roles": sorted({tuple(row.get("judge_message_roles", [])) for row in rows}),
        "system_prompt_present": sorted({row.get("judge_system_prompt_present") for row in rows}),
        "all_logprobs_finite": all(math.isfinite(value) for value in logps),
        "emitted_argmax_consistency": mean(float(row.get("emitted_argmax_consistent")) for row in rows),
        "single_token_output_rate": mean(float(row.get("judge_token") in {"0", "1"}) for row in rows),
        "binary_label_mass": {"min": min(masses), "mean": mean(masses), "max": max(masses)},
        "judge_token_counts": dict(Counter(str(row.get("judge_token")) for row in rows)),
        "judge_usage": {
            key: sum(float((row.get("judge_usage") or {}).get(key, 0) or 0) for row in rows)
            for key in usage_keys
        },
        "mean_judge_latency_seconds": mean(float(row.get("judge_latency_seconds", 0)) for row in rows),
    }


def baseline_summary(rows: list[dict[str, Any]], z: float) -> dict[str, Any]:
    groups = {
        "B_own": [row for row in rows if row["source"] == "B_own"],
        "A0_plain": [row for row in rows if row["source"] == "A0" and row["variant"] == "plain"],
        "A0_imitation": [row for row in rows if row["source"] == "A0" and row["variant"] == "imitation"],
    }
    if any(len(values) != 100 for values in groups.values()):
        raise ValueError({name: len(values) for name, values in groups.items()})
    result: dict[str, Any] = {"n": len(rows), "groups": {}}
    for name, values in groups.items():
        pown = [float(row["p_own"]) for row in values]
        result["groups"][name] = {
            "accuracy": accuracy(values),
            "mean_p_own": mean(pown),
            "S": mean(pown),
            "fooling_sampled": mean(float(row.get("judge_token") == "1") for row in values),
            "fooling_argmax": mean(float(value > 0.5) for value in pown),
            "mean_completion_tokens": mean(float(row.get("completion_tokens", 0)) for row in values),
            "termination_rate": mean(
                float(row.get("terminated_with_eos", row.get("terminated_normally", False))) for row in values
            ),
            "reward": {
                "mean": mean(float(row["reward"]) for row in values),
                "min": min(float(row["reward"]) for row in values),
                "max": max(float(row["reward"]) for row in values),
            },
        }
    own = groups["B_own"]
    for name in ("A0_plain", "A0_imitation"):
        values = groups[name]
        combined = own + values
        result["groups"][name].update(
            {
                "gap_S_B_minus_S_A": result["groups"]["B_own"]["S"] - result["groups"][name]["S"],
                "paired_gap": paired_lcb(own, values, z),
                "attribution_auc": auc(
                    [float(row["p_own"]) for row in combined],
                    [row["source"] == "B_own" for row in combined],
                ),
            }
        )
    result["protocol_audit"] = protocol_audit(rows)
    result["B_own_generation_usage"] = {
        key: sum(float((row.get("usage") or {}).get(key, 0) or 0) for row in groups["B_own"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
    }
    return result


def feature_audit(rows: list[dict[str, Any]], *, within_task: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
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
    rewards = [float(row["reward"]) for row in rows]
    for feature in SURFACE_FEATURES:
        values = [float(row["features"][feature]) for row in rows]
        item: dict[str, Any] = {"correlation_with_reward": correlation(values, rewards)}
        if within_task:
            centered_values = [value - group_means[row["id"]][feature] for value, row in zip(values, rows)]
            centered_rewards = [reward - group_means[row["id"]]["reward"] for reward, row in zip(rewards, rows)]
            item["within_task_correlation_with_reward"] = correlation(centered_values, centered_rewards)
        result[feature] = item
    return result


def lexical_audit(rows: list[dict[str, Any]], *, within_task: bool) -> dict[str, Any]:
    """Audit recurring lexical channels without fitting or changing the reward."""
    if not rows:
        return {"n": 0, "minimum_document_frequency": None, "top_tokens": [], "maximum_absolute_correlation": None}
    token_sets = [set(re.findall(r"[a-z]{3,}", str(row["response"]).lower())) for row in rows]
    minimum_df = max(2, math.ceil(0.05 * len(rows)))
    counts = Counter(token for tokens in token_sets for token in tokens)
    eligible = sorted(token for token, count in counts.items() if minimum_df <= count <= len(rows) - minimum_df)
    rewards = [float(row["reward"]) for row in rows]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["id"])].append(index)
    items: list[dict[str, Any]] = []
    for token in eligible:
        present = [float(token in tokens) for tokens in token_sets]
        raw = correlation(present, rewards)
        centered_present = list(present)
        centered_rewards = list(rewards)
        within = None
        if within_task:
            for indices in groups.values():
                feature_mean = float(np.mean([present[index] for index in indices]))
                reward_mean = float(np.mean([rewards[index] for index in indices]))
                for index in indices:
                    centered_present[index] -= feature_mean
                    centered_rewards[index] -= reward_mean
            within = correlation(centered_present, centered_rewards)
        items.append(
            {
                "token": token,
                "document_frequency": counts[token],
                "correlation_with_reward": raw,
                "within_task_correlation_with_reward": within,
            }
        )
    key = "within_task_correlation_with_reward" if within_task else "correlation_with_reward"
    ranked = sorted(
        items,
        key=lambda item: abs(float(item[key])) if item[key] is not None else -1.0,
        reverse=True,
    )
    finite = [abs(float(item[key])) for item in items if item[key] is not None and math.isfinite(float(item[key]))]
    return {
        "n": len(rows),
        "minimum_document_frequency": minimum_df,
        "eligible_tokens": len(items),
        "top_tokens": ranked[:20],
        "maximum_absolute_correlation": max(finite) if finite else None,
    }


def probe_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 400:
        raise ValueError(f"Expected 400 judged rollouts, received {len(rows)}")
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    if len(by_id) != 100 or any(len(values) != 4 for values in by_id.values()):
        raise ValueError("K=4 groups are incomplete")
    distinct_reward_counts: Counter[int] = Counter()
    distinct_response_counts: Counter[int] = Counter()
    variances: list[float] = []
    mixed_groups = 0
    top_mixed_correct = 0.0
    top_all_correct = 0.0
    correct_wins = 0.0
    pairs = 0
    groups_with_two_correct_varying = 0
    pass1 = 0
    pass4 = 0
    for values in by_id.values():
        ordered = sorted(values, key=lambda row: int(row["rollout_index"]))
        rewards = [float(row["reward"]) for row in ordered]
        distinct_reward_counts[len(set(round(value, 8) for value in rewards))] += 1
        distinct_response_counts[len(set(row["response"].strip() for row in ordered))] += 1
        variances.append(float(np.var(rewards)))
        correct = [row for row in ordered if row["score"]["correct"]]
        incorrect = [row for row in ordered if not row["score"]["correct"]]
        pass1 += int(bool(ordered[0]["score"]["correct"]))
        pass4 += int(bool(correct))
        maximum = max(rewards)
        top = [row for row in ordered if float(row["reward"]) == maximum]
        top_all_correct += sum(bool(row["score"]["correct"]) for row in top) / len(top)
        if correct and incorrect:
            mixed_groups += 1
            top_mixed_correct += sum(bool(row["score"]["correct"]) for row in top) / len(top)
            for correct_row in correct:
                for incorrect_row in incorrect:
                    cr = float(correct_row["reward"])
                    ir = float(incorrect_row["reward"])
                    correct_wins += float(cr > ir) + 0.5 * float(cr == ir)
                    pairs += 1
        if len(correct) >= 2 and len({round(float(row["reward"]), 8) for row in correct}) >= 2:
            groups_with_two_correct_varying += 1
    correct_rows = [row for row in rows if row["score"]["correct"]]
    incorrect_rows = [row for row in rows if not row["score"]["correct"]]
    all_accuracy = sum(bool(row["score"]["correct"]) for row in rows) / len(rows)
    return {
        "n": len(rows),
        "accuracy": accuracy(rows),
        "pass_at_1": pass1 / len(by_id),
        "pass_at_4": pass4 / len(by_id),
        "reward": {
            "mean": mean(float(row["reward"]) for row in rows),
            "std": float(np.std([float(row["reward"]) for row in rows])),
            "min": min(float(row["reward"]) for row in rows),
            "max": max(float(row["reward"]) for row in rows),
        },
        "p_own": {
            "mean": mean(float(row["p_own"]) for row in rows),
            "fooling_sampled": mean(float(row.get("judge_token") == "1") for row in rows),
            "fooling_argmax": mean(float(float(row["p_own"]) > 0.5) for row in rows),
        },
        "group_signal": {
            "n_groups": len(by_id),
            "complete_k4_groups": sum(len(values) == 4 for values in by_id.values()),
            "distinct_reward_count_distribution": dict(sorted(distinct_reward_counts.items())),
            "distinct_response_count_distribution": dict(sorted(distinct_response_counts.items())),
            "groups_with_at_least_two_distinct_rewards": sum(
                count for distinct, count in distinct_reward_counts.items() if distinct >= 2
            ),
            "groups_with_at_least_two_distinct_responses": sum(
                count for distinct, count in distinct_response_counts.items() if distinct >= 2
            ),
            "mean_within_group_reward_variance": mean(variances),
            "median_within_group_reward_variance": float(np.median(variances)),
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
            "top_reward_accuracy_all_groups": top_all_correct / len(by_id),
            "random_candidate_expected_accuracy": all_accuracy,
            "top_reward_correct_rate_mixed_groups": top_mixed_correct / mixed_groups if mixed_groups else None,
            "correct_beats_incorrect_pairwise_rate": correct_wins / pairs if pairs else None,
            "n_correct_incorrect_pairs": pairs,
            "correct_only_reward_std": float(np.std([float(row["reward"]) for row in correct_rows])) if correct_rows else None,
            "groups_with_two_correct_and_reward_variation": groups_with_two_correct_varying,
        },
        "feature_audit": feature_audit(rows, within_task=True),
        "lexical_audit": lexical_audit(rows, within_task=True),
        "correct_only_feature_audit": feature_audit(correct_rows, within_task=True) if correct_rows else {},
        "correct_only_lexical_audit": lexical_audit(correct_rows, within_task=True),
        "prediction_counts": accuracy(rows)["prediction_counts"],
        "protocol_audit": protocol_audit(rows),
    }


def go_no_go(
    baseline: dict[str, Any],
    probe: dict[str, Any],
    shortcut: dict[str, Any] | None,
    gates: dict[str, Any],
    repeatability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    own = baseline["groups"]["B_own"]
    plain = baseline["groups"]["A0_plain"]
    correlations = {
        name: values.get("within_task_correlation_with_reward")
        for name, values in probe["feature_audit"].items()
    }
    finite = [abs(value) for value in correlations.values() if value is not None and math.isfinite(value)]
    lexical_max = probe.get("lexical_audit", {}).get("maximum_absolute_correlation")
    if lexical_max is not None and math.isfinite(float(lexical_max)):
        finite.append(abs(float(lexical_max)))
    audits = [baseline["protocol_audit"], probe["protocol_audit"]]
    protocol_ok = all(
        audit["all_logprobs_finite"]
        and audit["single_token_output_rate"] >= float(gates["minimum_single_binary_token_rate"])
        and audit["binary_label_mass"]["min"] > 0
        and audit["models"] == [str(gates["expected_model"])]
        and audit["providers"] == [str(gates["expected_provider"])]
        and audit["message_roles"] == [("user",)]
        and audit["system_prompt_present"] == [False]
        for audit in audits
    )
    repeatability_ok = repeatability is not None and (
        float(repeatability["reward_reliability"]["icc_1_1"])
        >= float(gates["minimum_reward_repeatability_icc"])
    )
    sample_calibration_gap = (
        abs(
            float(repeatability["decision_behavior"]["mean_emitted_own"])
            - float(repeatability["decision_behavior"]["mean_p_own"])
        )
        if repeatability is not None
        else None
    )
    sample_calibrated = sample_calibration_gap is not None and (
        sample_calibration_gap <= float(gates["maximum_sample_probability_calibration_gap"])
    )
    correctness_auc = probe["correctness_relation"]["reward_auc_for_correctness"]
    reward_not_only_correctness = (
        probe["correctness_relation"]["correct_only_reward_std"] is not None
        and probe["correctness_relation"]["correct_only_reward_std"]
        >= float(gates["minimum_correct_only_reward_std"])
        and (correctness_auc is None or correctness_auc <= float(gates["maximum_correctness_auc"]))
    )
    paired_shortcut_ok = shortcut is not None and (
        float(shortcut["maximum_absolute_mean_delta_p_own"])
        <= float(gates["maximum_absolute_paired_shortcut_effect"])
    )
    checks = {
        "B_accuracy_exceeds_A0_plain": own["accuracy"]["accuracy"] - plain["accuracy"]["accuracy"]
        >= float(gates["minimum_b_minus_a_accuracy"]),
        "paired_S_B_minus_S_A_positive": plain["paired_gap"]["lcb_95_one_sided"] > 0,
        "plain_attribution_auc": plain["attribution_auc"] >= float(gates["minimum_attribution_auc"]),
        "logprob_protocol_coherent": protocol_ok,
        "continuous_reward_repeatable": repeatability_ok,
        "emitted_sample_calibrated_to_p_own": sample_calibrated,
        "reward_varies_within_K4_groups": probe["group_signal"]["groups_with_at_least_two_distinct_rewards"]
        >= int(gates["minimum_groups_with_reward_variation"]),
        "reward_not_only_correctness": reward_not_only_correctness,
        "no_dominant_correlational_surface_shortcut": (max(finite) if finite else 0.0)
        <= float(gates["maximum_absolute_within_task_surface_correlation"]),
        "no_dominant_paired_surface_shortcut": paired_shortcut_ok,
        "top_reward_not_less_accurate_than_random": (
            probe["correctness_relation"]["top_reward_accuracy_all_groups"]
            >= probe["correctness_relation"]["random_candidate_expected_accuracy"]
        ),
    }
    if repeatability is None:
        decision = "PENDING-REPEATABILITY-AUDIT"
    elif shortcut is None:
        decision = "PENDING-SHORTCUT-AUDIT"
    else:
        decision = "GO" if all(checks.values()) else "NO-GO"
    return {
        "decision": decision,
        "checks": checks,
        "n_passed": sum(checks.values()),
        "n_checks": len(checks),
        "thresholds": gates,
        "maximum_observed_absolute_within_task_surface_correlation": max(finite) if finite else None,
        "paired_shortcut_audit_present": shortcut is not None,
        "repeatability_audit_present": repeatability is not None,
        "sample_probability_calibration_gap": sample_calibration_gap,
        "non_gating_diagnostics": {
            "baseline_emitted_argmax_consistency": baseline["protocol_audit"]["emitted_argmax_consistency"],
            "probe_emitted_argmax_consistency": probe["protocol_audit"]["emitted_argmax_consistency"],
            "interpretation": (
                "The endpoint emits a stochastic sample; argmax fooling is derived from logp_1-logp_0. "
                "Per-call emitted/argmax equality is reported but is not a coherence requirement when the "
                "sample rate is calibrated and the continuous reward is repeatable."
            ),
        },
    }


def save_plots(run_dir: Path, baseline_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "figures"
    plot_dir.mkdir(exist_ok=True)
    files: list[str] = []
    groups = {
        "B own": [row for row in baseline_rows if row["source"] == "B_own"],
        "A0 plain": [row for row in baseline_rows if row["source"] == "A0" and row["variant"] == "plain"],
        "A0 imitation": [row for row in baseline_rows if row["source"] == "A0" and row["variant"] == "imitation"],
    }
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (label, values), color in zip(groups.items(), colors):
        ax.hist([float(row["p_own"]) for row in values], bins=np.linspace(0, 1, 21), density=True,
                histtype="step", linewidth=2.2, color=color, label=label)
    ax.set(xlabel=r"Self-attribution score $p_{own}$", ylabel="Density", title="GSM8K self-attribution audit")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = plot_dir / f"attribution_distributions.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(str(path.relative_to(run_dir)))
    plt.close(fig)

    correct = [float(row["reward"]) for row in probe_rows if row["score"]["correct"]]
    incorrect = [float(row["reward"]) for row in probe_rows if not row["score"]["correct"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    if correct and incorrect:
        parts = ax.violinplot([incorrect, correct], positions=[0, 1], showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], ("#D55E00", "#009E73")):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
        parts["cmeans"].set_color("#222222")
    ax.set_xticks([0, 1], [f"Incorrect\n(n={len(incorrect)})", f"Correct\n(n={len(correct)})"])
    ax.set(ylabel=r"Reward $\log P_B(1\mid x,y)$", title="Reward versus GSM8K exact match")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = plot_dir / f"reward_by_correctness.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(str(path.relative_to(run_dir)))
    plt.close(fig)
    manifest = {
        "files": files,
        "reward": "absolute log-probability log P_B(label=1 | task, candidate)",
        "external_quality": "normalized exact match of the final Answer: X line",
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
    root = Path(config["project"]["root"]).resolve()
    split_dir = Path(config["paths"]["split_dir"])
    if not split_dir.is_absolute():
        split_dir = root / split_dir
    examples = {row["id"]: row for row in read_jsonl(split_dir / "reward_audit.jsonl")}
    for row in baseline_rows + probe_rows:
        example = examples[row["id"]]
        row["score"] = score_response(row["response"], example["answer_value"])
        row["features"] = response_features(row["response"], example["question"])
    baseline = baseline_summary(baseline_rows, float(config["gates"]["paired_lcb_z"]))
    probe = probe_summary(probe_rows)
    replicated_shortcut_path = run_dir / "shortcut_replicated_audit.json"
    shortcut_path = (
        replicated_shortcut_path
        if replicated_shortcut_path.exists()
        else run_dir / "shortcut_audit.json"
    )
    shortcut = json.loads(shortcut_path.read_text(encoding="utf-8")) if shortcut_path.exists() else None
    repeatability_path = run_dir / "judge_repeatability_audit.json"
    repeatability = (
        json.loads(repeatability_path.read_text(encoding="utf-8"))
        if repeatability_path.exists()
        else None
    )
    decision = go_no_go(baseline, probe, shortcut, config["gates"], repeatability)
    plots = save_plots(run_dir, baseline_rows, probe_rows)
    cost_components = {
        "B_own_generation": float(baseline["B_own_generation_usage"]["cost"]),
        "baseline_judgments": float(baseline["protocol_audit"]["judge_usage"]["cost"]),
        "probe_judgments": float(probe["protocol_audit"]["judge_usage"]["cost"]),
        "shortcut_judgments": float((shortcut or {}).get("judge_usage", {}).get("cost", 0.0)),
        "repeatability_judgments": float((repeatability or {}).get("usage", {}).get("cost", 0.0)),
    }
    temperature_path = run_dir / "endpoint_temperature_audit.json"
    temperature_audit = (
        json.loads(temperature_path.read_text(encoding="utf-8"))
        if temperature_path.exists()
        else None
    )
    temperature_cost = sum(
        float((row.get("usage") or {}).get("cost", 0) or 0)
        for row in (temperature_audit or {}).get("records", [])
        if row.get("status") == "ok"
    )
    cost_components["endpoint_temperature_audit"] = temperature_cost
    summary = {
        "status": "complete",
        "baseline": baseline,
        "probe": probe,
        "shortcut_audit": shortcut,
        "judge_repeatability_audit": repeatability,
        "endpoint_temperature_audit": temperature_audit,
        "go_no_go": decision,
        "costs": {**cost_components, "total_openrouter_cost": sum(cost_components.values())},
        "plots": plots,
    }
    write_json(run_dir / "phase0_summary.json", summary)
    write_json(run_dir / "go_no_go.json", decision)
    write_json(run_dir / "status.json", {"status": "complete", "decision": decision["decision"]})
    print(json.dumps({
        "baseline_accuracy": {name: value["accuracy"]["accuracy"] for name, value in baseline["groups"].items()},
        "baseline_p_own": {name: value["mean_p_own"] for name, value in baseline["groups"].items()},
        "probe_pass_at_1": probe["pass_at_1"],
        "probe_pass_at_4": probe["pass_at_4"],
        "go_no_go": decision,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
