#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def auc(scores_positive: list[float], scores_negative: list[float]) -> float | None:
    if not scores_positive or not scores_negative:
        return None
    wins = 0.0
    for positive in scores_positive:
        for negative in scores_negative:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(scores_positive) * len(scores_negative))


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "min": quantile(values, 0.0),
        "q10": quantile(values, 0.1),
        "q25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "q75": quantile(values, 0.75),
        "q90": quantile(values, 0.9),
        "max": quantile(values, 1.0),
        "mean": mean(values),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stratified_bootstrap_ci(
    own: list[dict[str, Any]],
    non_own: list[dict[str, Any]],
    statistic: Any,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Percentile CI while preserving the planned 50/50 class balance."""
    point_estimate = statistic(own, non_own)
    if not own or not non_own or point_estimate is None:
        return {
            "point_estimate": point_estimate,
            "lower": None,
            "upper": None,
            "replicates_requested": replicates,
            "replicates_valid": 0,
        }
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        sampled_own = [own[rng.randrange(len(own))] for _ in own]
        sampled_non_own = [non_own[rng.randrange(len(non_own))] for _ in non_own]
        value = statistic(sampled_own, sampled_non_own)
        if value is not None:
            samples.append(float(value))
    return {
        "point_estimate": float(point_estimate),
        "lower": quantile(samples, 0.025),
        "upper": quantile(samples, 0.975),
        "replicates_requested": replicates,
        "replicates_valid": len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")

    rows = load_jsonl(args.run_dir / "judgments.jsonl")
    valid = [row for row in rows if row["judge"]["parsed"].get("ok")]
    own = [row for row in valid if int(row["ground_truth"]) == 1]
    non_own = [row for row in valid if int(row["ground_truth"]) == 0]

    def p1(row: dict[str, Any]) -> float:
        return float(row["judge"]["parsed"]["p1"])

    def emitted(row: dict[str, Any]) -> int | None:
        value = row.get("prediction")
        return None if value is None else int(value)

    def argmax(row: dict[str, Any]) -> int:
        parsed = row["judge"]["parsed"]
        return int(float(parsed["logp_1"]) > float(parsed["logp_0"]))

    def accuracy(rows_subset: list[dict[str, Any]], predictor: Any) -> float | None:
        values = [
            float(prediction == int(row["ground_truth"]))
            for row in rows_subset
            if (prediction := predictor(row)) is not None
        ]
        return mean(values)

    def confusion(rows_subset: list[dict[str, Any]], predictor: Any) -> dict[str, int]:
        counts = {"true_negative": 0, "false_positive": 0, "false_negative": 0, "true_positive": 0}
        for row in rows_subset:
            prediction = predictor(row)
            if prediction is None:
                continue
            truth = int(row["ground_truth"])
            if truth == 0 and prediction == 0:
                counts["true_negative"] += 1
            elif truth == 0 and prediction == 1:
                counts["false_positive"] += 1
            elif truth == 1 and prediction == 0:
                counts["false_negative"] += 1
            else:
                counts["true_positive"] += 1
        counts["n_scored"] = sum(counts.values())
        return counts

    emitted_valid = [row for row in valid if emitted(row) is not None]
    masses = [
        float(row["judge"]["parsed"]["binary_label_mass"])
        for row in valid
        if row["judge"]["parsed"].get("binary_label_mass") is not None
    ]
    consistencies = [float(emitted(row) == argmax(row)) for row in emitted_valid]
    own_scores = [p1(row) for row in own]
    non_own_scores = [p1(row) for row in non_own]
    own_gaps = [
        float(row["judge"]["parsed"]["logp_1"])
        - float(row["judge"]["parsed"]["logp_0"])
        for row in own
    ]
    non_own_gaps = [
        float(row["judge"]["parsed"]["logp_1"])
        - float(row["judge"]["parsed"]["logp_0"])
        for row in non_own
    ]

    metadata = json.loads((args.run_dir / "summary.json").read_text())
    source_dirs = {
        "B_own": Path(metadata["source_runs"]["b_own"]),
        "plain": Path(metadata["source_runs"]["plain"]),
        "imitation": Path(metadata["source_runs"]["imitation"]),
    }
    source_responses = {
        source: {int(row["index"]): row for row in load_jsonl(path / "responses.jsonl")}
        for source, path in source_dirs.items()
    }

    def candidate_length(row: dict[str, Any]) -> int:
        source = str(row["candidate_source_key"])
        return len(str(source_responses[source][int(row["index"])]["text"]))

    own_lengths = [float(candidate_length(row)) for row in own]
    non_own_lengths = [float(candidate_length(row)) for row in non_own]

    def summarize_quality(rows_subset: list[dict[str, Any]]) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for passed in (False, True):
            group = [
                row
                for row in rows_subset
                if bool((row.get("candidate_unit_tests") or {}).get("passed")) == passed
            ]
            groups[str(passed).lower()] = {
                "n": len(group),
                "acceptance_emitted": mean([
                    float(emitted(row) == 1) for row in group
                ]),
                "mean_p_own": mean([p1(row) for row in group]),
            }
        return groups

    s_b = mean(own_scores)
    s_a = mean(non_own_scores)

    def gap_statistic(own_sample: list[dict[str, Any]], non_own_sample: list[dict[str, Any]]) -> float | None:
        own_mean = mean([p1(row) for row in own_sample])
        non_own_mean = mean([p1(row) for row in non_own_sample])
        return None if own_mean is None or non_own_mean is None else own_mean - non_own_mean

    def emitted_accuracy_statistic(
        own_sample: list[dict[str, Any]], non_own_sample: list[dict[str, Any]]
    ) -> float | None:
        return accuracy(own_sample + non_own_sample, emitted)

    def argmax_accuracy_statistic(
        own_sample: list[dict[str, Any]], non_own_sample: list[dict[str, Any]]
    ) -> float | None:
        return accuracy(own_sample + non_own_sample, argmax)

    def auc_statistic(own_sample: list[dict[str, Any]], non_own_sample: list[dict[str, Any]]) -> float | None:
        return auc([p1(row) for row in own_sample], [p1(row) for row in non_own_sample])

    bootstrap = {
        "gap_S_B_minus_S_A": stratified_bootstrap_ci(
            own, non_own, gap_statistic,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        ),
        "attribution_accuracy_emitted": stratified_bootstrap_ci(
            own, non_own, emitted_accuracy_statistic,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 1,
        ),
        "attribution_accuracy_argmax": stratified_bootstrap_ci(
            own, non_own, argmax_accuracy_statistic,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 2,
        ),
        "auc_own_vs_non_own": stratified_bootstrap_ci(
            own, non_own, auc_statistic,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 3,
        ),
    }
    own_correct = [
        row for row in own if bool((row.get("candidate_unit_tests") or {}).get("passed"))
    ]
    non_own_correct = [
        row for row in non_own if bool((row.get("candidate_unit_tests") or {}).get("passed"))
    ]
    metrics = {
        "n": len(rows),
        "n_valid_logprobs": len(valid),
        "n_own": len(own),
        "n_non_own": len(non_own),
        "pass_at_1": {
            "own": mean([
                float(row["candidate_unit_tests"]["passed"])
                for row in own
                if row.get("candidate_unit_tests")
            ]),
            "non_own": mean([
                float(row["candidate_unit_tests"]["passed"])
                for row in non_own
                if row.get("candidate_unit_tests")
            ]),
        },
        "S_B_mean_p_own_on_own": s_b,
        "S_A_mean_p_own_on_non_own": s_a,
        "gap_S_B_minus_S_A": None if s_b is None or s_a is None else s_b - s_a,
        "self_acceptance_emitted": mean([float(emitted(row) == 1) for row in own]),
        "self_acceptance_argmax": mean([float(argmax(row) == 1) for row in own]),
        "fooling_emitted": mean([float(emitted(row) == 1) for row in non_own]),
        "fooling_argmax": mean([float(argmax(row) == 1) for row in non_own]),
        "attribution_accuracy_emitted": mean([
            float(emitted(row) == int(row["ground_truth"]))
            for row in emitted_valid
        ]),
        "attribution_accuracy_argmax": mean([
            float(argmax(row) == int(row["ground_truth"]))
            for row in valid
        ]),
        "auc_own_vs_non_own": auc(own_scores, non_own_scores),
        "confusion_matrix": {
            "emitted": confusion(valid, emitted),
            "argmax": confusion(valid, argmax),
        },
        "bootstrap_ci_95": bootstrap,
        "correct_only_control": {
            "n_own": len(own_correct),
            "n_non_own": len(non_own_correct),
            "S_B": mean([p1(row) for row in own_correct]),
            "S_A": mean([p1(row) for row in non_own_correct]),
            "auc_own_vs_non_own": auc(
                [p1(row) for row in own_correct],
                [p1(row) for row in non_own_correct],
            ),
            "self_acceptance_emitted": mean([
                float(emitted(row) == 1) for row in own_correct
            ]),
            "fooling_emitted": mean([
                float(emitted(row) == 1) for row in non_own_correct
            ]),
        },
        "p_own_distribution": {
            "own": distribution(own_scores),
            "non_own": distribution(non_own_scores),
        },
        "logit_gap_distribution": {
            "own": distribution(own_gaps),
            "non_own": distribution(non_own_gaps),
        },
        "candidate_length_chars": {
            "own": distribution(own_lengths),
            "non_own": distribution(non_own_lengths),
            "correlation_with_p_own": {
                "own": pearson(own_lengths, own_scores),
                "non_own": pearson(non_own_lengths, non_own_scores),
                "overall": pearson(own_lengths + non_own_lengths, own_scores + non_own_scores),
            },
        },
        "by_candidate_correctness": {
            "own": summarize_quality(own),
            "non_own": summarize_quality(non_own),
        },
        "saturation": {
            "own_p_below_1e-6": mean([float(value < 1e-6) for value in own_scores]),
            "own_p_above_1_minus_1e-6": mean([
                float(value > 1.0 - 1e-6) for value in own_scores
            ]),
            "non_own_p_below_1e-6": mean([
                float(value < 1e-6) for value in non_own_scores
            ]),
            "non_own_p_above_1_minus_1e-6": mean([
                float(value > 1.0 - 1e-6) for value in non_own_scores
            ]),
        },
        "emitted_argmax_consistency": mean(consistencies),
        "binary_label_mass": {
            "min": min(masses) if masses else None,
            "mean": mean(masses),
            "max": max(masses) if masses else None,
        },
        "raw_models": sorted({str(row["judge"].get("raw_model")) for row in rows}),
        "raw_providers": sorted({str(row["judge"].get("raw_provider")) for row in rows}),
        "total_api_cost": sum(
            float((row["judge"].get("usage") or {}).get("cost") or 0.0)
            for row in rows
        ),
    }

    out_path = args.run_dir / "audit_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
