from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml


PARAMETER_COLUMNS = [
    "learning_rate",
    "beta",
    "temperature",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "scale_rewards",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate auditable GRPO HPO results")
    parser.add_argument("--study-dir", required=True)
    args = parser.parse_args()
    outputs = analyze_study(Path(args.study_dir))
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


def analyze_study(study_dir: Path) -> dict[str, str | int]:
    study_dir = study_dir.expanduser().resolve()
    rows = [
        summarize_trial(path)
        for path in sorted((study_dir / "trials").glob("hpo_logp_trial_*_seed*"))
        if (path / "config.yaml").exists()
    ]
    rows.sort(
        key=lambda row: (
            row.get("objective_value") is not None,
            float(row.get("objective_value") or -math.inf),
        ),
        reverse=True,
    )
    analysis_dir = study_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    json_path = analysis_dir / "trials.json"
    csv_path = analysis_dir / "trials.csv"
    markdown_path = analysis_dir / "ranking.md"
    protocol_path = analysis_dir / "protocol_audit.json"
    cost_path = analysis_dir / "cost_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    markdown_path.write_text(render_markdown(rows), encoding="utf-8")
    protocol_path.write_text(
        json.dumps(audit_protocol(study_dir), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cost_summary = {
        "schema_version": 1,
        "trial_count": len(rows),
        "complete_trial_cost": sum(
            float(row.get("total_openrouter_cost") or 0.0)
            for row in rows
            if row["status"] == "complete"
        ),
        "incomplete_or_failed_trial_cost": sum(
            float(row.get("total_openrouter_cost") or 0.0)
            for row in rows
            if row["status"] != "complete"
        ),
    }
    cost_summary["total_openrouter_cost"] = (
        cost_summary["complete_trial_cost"]
        + cost_summary["incomplete_or_failed_trial_cost"]
    )
    cost_path.write_text(
        json.dumps(cost_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    outputs: dict[str, str | int] = {
        "trial_count": len(rows),
        "complete_count": sum(row["status"] == "complete" for row in rows),
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "protocol_audit": str(protocol_path),
        "cost_summary": str(cost_path),
    }
    (analysis_dir / "manifest.json").write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return outputs


def audit_protocol(study_dir: Path) -> dict[str, Any]:
    run_dirs = [
        path
        for path in sorted((study_dir / "trials").glob("hpo_logp_trial_*_seed*"))
        if (path / "objective.json").exists()
    ]
    if not run_dirs:
        return {"status": "no_complete_trials", "consistent": False}
    reference_dir = run_dirs[0]
    reference_config = read_yaml(reference_dir / "config.yaml")
    required_paths = [
        ("project", "root"),
        ("paths", "experiments"),
        ("paths", "checkpoints"),
        ("dataset", "index_manifest"),
        ("trial", "run_name"),
        ("actor", "lora", "r"),
        ("actor", "lora", "dropout"),
        ("discriminator", "model_id"),
        ("reward", "value"),
        ("reward", "max_reward_calls"),
        ("training", "learning_rate"),
        ("training", "beta"),
        ("training", "temperature"),
        ("training", "scale_rewards"),
    ]
    missing = [".".join(path) for path in required_paths if not has_nested(reference_config, path)]
    if missing:
        return {
            "status": "insufficient_protocol_config",
            "consistent": False,
            "reference_run": reference_dir.name,
            "missing_fields": missing,
        }
    reference_normalized = normalized_protocol_config(reference_config)
    reference_training_examples = training_prompt_manifest(reference_dir)
    reference_validation = validation_manifest(reference_dir)
    reference_baseline_generations = baseline_generation_manifest(reference_dir)
    expected_model = str(reference_config["discriminator"]["model_id"])
    expected_reward = str(reference_config["reward"]["value"])
    expected_calls = int(reference_config["reward"]["max_reward_calls"])
    rows = []
    baseline_p_own_means: list[float] = []
    for run_dir in run_dirs:
        config = read_yaml(run_dir / "config.yaml")
        summary = read_json_if_present(run_dir / "summary.json") or {}
        reward_stats = summary.get("reward_statistics") or {}
        audit = summary.get("reward_audit") or {}
        groups = reward_stats.get("by_task") or {}
        evaluation = read_json_if_present(run_dir / "hpo_evaluation_summary.json") or {}
        baseline_state = next(
            (
                state
                for state in evaluation.get("validation", [])
                if int(state.get("step", -1)) == 0
            ),
            None,
        )
        if baseline_state and baseline_state.get("p_own_mean") is not None:
            baseline_p_own_means.append(float(baseline_state["p_own_mean"]))
        raw_models = set((reward_stats.get("raw_models") or {}).keys())
        reward_values = set((reward_stats.get("reward_values") or {}).keys())
        checks = {
            "normalized_config_matches": normalized_protocol_config(config)
            == reference_normalized,
            "training_prompts_match": training_prompt_manifest(run_dir)
            == reference_training_examples,
            "validation_manifest_matches": validation_manifest(run_dir)
            == reference_validation,
            "baseline_generator_outputs_match": baseline_generation_manifest(run_dir)
            == reference_baseline_generations,
            "logical_reward_calls_match": audit.get("logical_reward_calls")
            == expected_calls,
            "all_training_rewards_succeeded": reward_stats.get("n_ok") == expected_calls,
            "discriminator_route_matches": raw_models == {expected_model},
            "reward_value_matches": reward_values == {expected_reward},
            "all_groups_have_variance": len(groups) == 100
            and all(float(group.get("std", 0.0)) > 0 for group in groups.values()),
            "all_groups_have_four_unique_candidates": len(groups) == 100
            and all(int(group.get("unique_candidates", 0)) == 4 for group in groups.values()),
        }
        rows.append(
            {
                "run_id": run_dir.name,
                "checks": checks,
                "consistent": all(checks.values()),
            }
        )
    manifest_path = Path(reference_config["project"]["root"]) / str(
        reference_config["dataset"]["index_manifest"]
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "complete_trial_count": len(rows),
        "reference_run": reference_dir.name,
        "expected_discriminator_model": expected_model,
        "expected_reward": expected_reward,
        "expected_training_reward_calls": expected_calls,
        "training_index_manifest": str(manifest_path),
        "training_index_manifest_sha256": sha256_if_present(manifest_path),
        "training_prompt_manifest_sha256": hash_json(reference_training_examples),
        "validation_manifest_sha256": hash_json(reference_validation),
        "baseline_generation_manifest_sha256": hash_json(
            reference_baseline_generations
        ),
        "baseline_discriminator_score_variation": distribution_summary(
            baseline_p_own_means
        ),
        "normalized_protocol_config_sha256": hash_json(reference_normalized),
        "runs": rows,
        "consistent": all(row["consistent"] for row in rows),
        "note": (
            "The normalized config excludes only run identifiers, output paths, the seven "
            "registered Optuna parameters, and HPO provenance. Generator task statements and "
            "their order are compared directly across every complete trial. Step-0 Actor "
            "generations are compared byte-for-byte; their Discriminator scores are reported "
            "separately because a remote judge can vary despite a fixed model identifier."
        ),
    }


def normalized_protocol_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["trial"]["run_name"] = "<RUN_ID>"
    value["paths"]["experiments"] = "<EXPERIMENTS>"
    value["paths"]["checkpoints"] = "<CHECKPOINTS>"
    for key in ["learning_rate", "beta", "temperature", "scale_rewards"]:
        value["training"][key] = f"<OPTUNA:{key}>"
    value["actor"]["lora"]["r"] = "<OPTUNA:lora_r>"
    value["actor"]["lora"]["dropout"] = "<OPTUNA:lora_dropout>"
    value["hpo"] = "<HPO_PROVENANCE>"
    return value


def has_nested(value: dict[str, Any], path: tuple[str, ...]) -> bool:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def training_prompt_manifest(run_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "task_id": str(row["task_id"]),
            "dataset_index": int(row["dataset_index"]),
            "task_statement": str(row["task_statement"]),
        }
        for row in read_jsonl(run_dir / "train_examples.jsonl")
    ]


def validation_manifest(run_dir: Path) -> dict[str, Any] | None:
    evaluation = read_json_if_present(run_dir / "hpo_evaluation_summary.json")
    if not evaluation:
        return None
    return (evaluation.get("split_manifest") or {}).get("validation")


def baseline_generation_manifest(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "evaluations" / "validation" / "step_0000" / "rows.jsonl"
    return [
        {
            "task_id": str(row["task_id"]),
            "dataset_index": int(row["dataset_index"]),
            "candidate_raw": str(row["candidate_raw"]),
        }
        for row in read_jsonl(path)
    ]


def distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_if_present(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def summarize_trial(run_dir: Path) -> dict[str, Any]:
    config = read_yaml(run_dir / "config.yaml")
    hpo = config.get("hpo") or {}
    params = dict(hpo.get("parameters") or {})
    row: dict[str, Any] = {
        "run_id": run_dir.name,
        "trial_number": hpo.get("trial_number"),
        "seed": int(config["project"]["seed"]),
        "source": hpo.get("source"),
        "status": "incomplete",
        **{name: params.get(name) for name in PARAMETER_COLUMNS},
    }

    training_summary = read_json_if_present(run_dir / "summary.json")
    training_logged_cost = logged_cost([run_dir / "reward_calls.jsonl"])
    row["training_logged_cost"] = training_logged_cost
    if training_summary:
        row["training_status"] = training_summary.get("status")
        reward_stats = training_summary.get("reward_statistics") or {}
        audit = training_summary.get("reward_audit") or {}
        usage = reward_stats.get("usage") or {}
        groups = list((reward_stats.get("by_task") or {}).values())
        row.update(
            {
                "training_reward_calls": audit.get("logical_reward_calls"),
                "training_http_requests": audit.get("http_requests"),
                "training_retries": _difference(
                    audit.get("http_requests"), audit.get("logical_reward_calls")
                ),
                "training_cost": usage.get("cost"),
                "training_p_own_mean": (reward_stats.get("p_own") or {}).get("mean"),
                "training_reward_mean": (reward_stats.get("reward") or {}).get("mean"),
                "nonzero_reward_groups": sum(float(group.get("std", 0.0)) > 0 for group in groups),
                "group_count": len(groups),
                "minimum_unique_candidates": min(
                    (int(group.get("unique_candidates", 0)) for group in groups), default=None
                ),
            }
        )
        training_log = reward_log_metrics(run_dir / "reward_calls.jsonl")
        row.update(
            {
                "training_instrumented_reward_calls": training_log.get(
                    "instrumented_token_count"
                ),
                "training_emitted_logprob_token_mismatches": training_log.get(
                    "emitted_logprob_token_mismatches"
                ),
                "training_emitted_argmax_mismatches": training_log.get(
                    "emitted_argmax_mismatches"
                ),
                "training_binary_argmax_ties": training_log.get("ties"),
                "training_binary_label_mass_min": training_log.get(
                    "binary_label_mass_min"
                ),
            }
        )

    evaluation = read_json_if_present(run_dir / "hpo_evaluation_summary.json")
    objective = read_json_if_present(run_dir / "objective.json")
    evaluation_logged_cost = logged_cost(
        sorted((run_dir / "evaluations" / "validation").glob("step_*/reward_calls.jsonl"))
    )
    row["evaluation_logged_cost"] = evaluation_logged_cost
    if evaluation and objective:
        selected_step = objective.get("selected_step")
        states = {int(state["step"]): state for state in evaluation["validation"]}
        baseline = states[0]
        selected = states.get(int(selected_step)) if selected_step is not None else None
        baseline_log = evaluation_log_metrics(run_dir, 0)
        selected_log = (
            evaluation_log_metrics(run_dir, int(selected_step))
            if selected_step is not None
            else {}
        )
        comparison = next(
            (
                item
                for item in objective.get("comparisons", [])
                if item.get("step") == selected_step
            ),
            None,
        )
        eval_cost = sum(
            float((state.get("reward_usage") or {}).get("cost", 0.0) or 0.0)
            for state in states.values()
        )
        row.update(
            {
                "status": "complete",
                "objective_value": objective.get("objective_value"),
                "selected_step": selected_step,
                "quality_feasible": objective.get("quality_feasible"),
                "p_own_baseline": baseline.get("p_own_mean"),
                "p_own_selected": selected.get("p_own_mean") if selected else None,
                "p_own_delta": comparison.get("p_own_delta_mean") if comparison else None,
                "p_own_lcb": comparison.get("p_own_delta_lcb") if comparison else None,
                "hard_fooling_baseline": baseline.get("fooling_rate"),
                "hard_fooling_selected": selected.get("fooling_rate") if selected else None,
                "hard_fooling_delta": _difference(
                    selected.get("fooling_rate") if selected else None,
                    baseline.get("fooling_rate"),
                ),
                "argmax_fooling_baseline": baseline.get(
                    "argmax_fooling_rate", baseline_log.get("argmax_fooling_rate")
                ),
                "argmax_fooling_selected": (
                    selected.get(
                        "argmax_fooling_rate", selected_log.get("argmax_fooling_rate")
                    )
                    if selected
                    else None
                ),
                "argmax_fooling_delta": _difference(
                    (
                        selected.get(
                            "argmax_fooling_rate", selected_log.get("argmax_fooling_rate")
                        )
                        if selected
                        else None
                    ),
                    baseline.get(
                        "argmax_fooling_rate", baseline_log.get("argmax_fooling_rate")
                    ),
                ),
                "pass_at_1_baseline": baseline.get("pass@1"),
                "pass_at_1_selected": selected.get("pass@1") if selected else None,
                "pass_at_1_delta": comparison.get("pass_at_1_delta") if comparison else None,
                "net_passed_change": comparison.get("net_passed_change") if comparison else None,
                "evaluation_cost": eval_cost,
                "logprob_contradictions": sum(
                    int((state.get("logprob_consistency") or {}).get("contradictions", 0))
                    for state in states.values()
                ),
                "selected_binary_label_mass_min": selected_log.get(
                    "binary_label_mass_min"
                ),
                "selected_instrumented_reward_calls": selected_log.get(
                    "instrumented_token_count"
                ),
                "selected_emitted_logprob_token_mismatches": selected_log.get(
                    "emitted_logprob_token_mismatches"
                ),
                "selected_emitted_argmax_mismatches": selected_log.get(
                    "emitted_argmax_mismatches"
                ),
                "selected_binary_argmax_ties": selected_log.get("ties"),
            }
        )

    trainer_rows = read_jsonl(run_dir / "trainer_log_history.jsonl")
    step_rows = [item for item in trainer_rows if "loss" in item]
    if step_rows:
        for key, output_name in [
            ("loss", "loss_mean"),
            ("grad_norm", "gradient_norm_mean"),
            ("kl", "kl_mean"),
            ("entropy", "entropy_mean"),
            ("clip_ratio/region_mean", "ppo_clip_ratio_mean"),
            ("completions/clipped_ratio", "completion_truncation_mean"),
            ("completions/mean_length", "completion_length_mean"),
        ]:
            values = [float(item[key]) for item in step_rows if item.get(key) is not None]
            row[output_name] = statistics.mean(values) if values else None
            row[output_name.replace("_mean", "_max")] = max(values) if values else None

    row["total_openrouter_cost"] = _sum_optional(
        (
            training_logged_cost
            if training_logged_cost is not None
            else row.get("training_cost")
        ),
        (
            evaluation_logged_cost
            if evaluation_logged_cost is not None
            else row.get("evaluation_cost")
        ),
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    complete = [row for row in rows if row["status"] == "complete"]
    lines = [
        r"# GRPO log-\(p_{\rm own}\) HPO ranking",
        "",
        "The ranking uses the pre-registered paired validation lower confidence bound. "
        "Hard fooling and pass@1 are reported independently.",
        "",
        "| Trial | Seed | LCB | Δ p_own | Δ sampled fooling | Δ argmax fooling | Δ pass@1 | Step | LR | β | Temp. | Rank | Dropout | Scaling | Cost |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in complete:
        lines.append(
            "| {trial_number} | {seed} | {objective_value:+.4f} | {p_own_delta:+.4f} | "
            "{hard_fooling_delta:+.4f} | {argmax_fooling_delta:+.4f} | "
            "{pass_at_1_delta:+.4f} | {selected_step} | "
            "{learning_rate:.0e} | {beta:.0e} | {temperature:.1f} | {lora_r} | "
            "{lora_dropout:.2f} | {scale_rewards} | ${total_openrouter_cost:.4f} |".format(
                **row
            )
        )
    incomplete = [row for row in rows if row["status"] != "complete"]
    if incomplete:
        lines.extend(
            [
                "",
                "Incomplete runs: "
                + ", ".join(str(row.get("trial_number")) for row in incomplete)
                + ".",
            ]
        )
    lines.extend(
        [
            "",
            "Every trial directory retains the complete configuration, request log, trainer log, "
            "checkpoint evaluation, usage and cost records.",
            "",
        ]
    )
    return "\n".join(lines)


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}")
    return value


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def logged_cost(paths: list[Path]) -> float | None:
    costs = [
        float((row.get("usage") or {}).get("cost"))
        for path in paths
        for row in read_jsonl(path)
        if (row.get("usage") or {}).get("cost") is not None
    ]
    return sum(costs) if costs else None


def evaluation_log_metrics(run_dir: Path, step: int) -> dict[str, float | int | None]:
    path = (
        run_dir
        / "evaluations"
        / "validation"
        / f"step_{step:04d}"
        / "reward_calls.jsonl"
    )
    return reward_log_metrics(path)


def reward_log_metrics(path: Path) -> dict[str, float | int | None]:
    rows = [row for row in read_jsonl(path) if row.get("status") == "ok"]
    if not rows:
        return {}
    masses = [float(row["binary_label_mass"]) for row in rows]
    instrumented = [
        row
        for row in rows
        if row.get("judge_token") is not None
        and row.get("logprob_first_token") is not None
    ]
    return {
        "n": len(rows),
        "sampled_fooling_rate": statistics.mean(
            str(row["judge_token"]) == "1" for row in rows
        ),
        "argmax_fooling_rate": statistics.mean(float(row["delta"]) > 0 for row in rows),
        "contradictions": sum(
            (float(row["delta"]) > 0 and str(row["judge_token"]) != "1")
            or (float(row["delta"]) < 0 and str(row["judge_token"]) != "0")
            for row in rows
        ),
        "ties": sum(float(row["delta"]) == 0 for row in rows),
        "binary_label_mass_min": min(masses),
        "instrumented_token_count": len(instrumented),
        "emitted_logprob_token_mismatches": (
            sum(
                str(row["judge_token"]).strip()
                != str(row["logprob_first_token"]).strip()
                for row in instrumented
            )
            if instrumented
            else None
        ),
        "emitted_argmax_mismatches": (
            sum(
                (float(row["delta"]) > 0 and str(row["judge_token"]).strip() != "1")
                or (float(row["delta"]) < 0 and str(row["judge_token"]).strip() != "0")
                for row in instrumented
            )
            if instrumented
            else None
        ),
    }


def _difference(left: Any, right: Any) -> float | int | None:
    if left is None or right is None:
        return None
    return left - right


def _sum_optional(*values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) if present else None


if __name__ == "__main__":
    main()
