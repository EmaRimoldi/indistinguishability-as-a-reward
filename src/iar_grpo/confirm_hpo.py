from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .hpo import read_yaml, write_json
from .reward import ATTRIBUTION_PROMPT_VERSION


Z_ONE_SIDED_95 = 1.6448536269514722


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and collect multi-seed HPO confirmation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--study-dir", required=True)
    prepare.add_argument("--candidate-trials", required=True)
    prepare.add_argument("--seeds", default="17,23,42")
    prepare.add_argument("--batch-id")
    collect = subparsers.add_parser("collect")
    collect.add_argument("--batch-dir", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        batch = prepare_confirmation(
            Path(args.study_dir),
            parse_integers(args.candidate_trials),
            parse_integers(args.seeds),
            args.batch_id,
        )
        print(batch)
    else:
        summary = collect_confirmation(Path(args.batch_dir))
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def prepare_confirmation(
    study_dir: Path,
    candidate_trials: list[int],
    seeds: list[int],
    batch_id: str | None,
) -> Path:
    study_dir = study_dir.expanduser().resolve()
    if not candidate_trials or not seeds:
        raise ValueError("At least one candidate and one seed are required")
    if len(set(candidate_trials)) != len(candidate_trials) or len(set(seeds)) != len(seeds):
        raise ValueError("Candidates and seeds must be unique")
    batch_id = batch_id or datetime.now(timezone.utc).strftime("confirm_%Y%m%dT%H%M%SZ")
    batch_dir = study_dir / "confirmations" / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    specs_dir = batch_dir / "specs"
    configs_dir = batch_dir / "configs"
    results_dir = batch_dir / "results"
    specs_dir.mkdir()
    configs_dir.mkdir()
    results_dir.mkdir()

    study_name = json.loads((study_dir / "study_manifest.json").read_text())["study_name"]
    candidates: list[dict[str, Any]] = []
    spec_paths: list[Path] = []
    for trial_number in candidate_trials:
        source_config_path = study_dir / "configs" / f"trial_{trial_number:04d}.yaml"
        source_config = read_yaml(source_config_path)
        source_seed = int(source_config["project"]["seed"])
        source_run_id = str(source_config["trial"]["run_name"])
        source_run_dir = (
            Path(source_config["project"]["root"])
            / source_config["paths"]["experiments"]
            / source_run_id
        ).resolve()
        objective_path = source_run_dir / "objective.json"
        if not objective_path.exists():
            raise RuntimeError(f"Candidate trial {trial_number} is not complete")
        objective = json.loads(objective_path.read_text(encoding="utf-8"))
        fixed_step = int(objective["selected_step"])
        params = dict((source_config.get("hpo") or {})["parameters"])
        candidate: dict[str, Any] = {
            "trial_number": trial_number,
            "parameters": params,
            "fixed_confirmation_step": fixed_step,
            "source_seed": source_seed,
            "source_run_id": source_run_id,
            "source_run_dir": str(source_run_dir),
            "runs": [],
        }
        for seed in seeds:
            config = copy.deepcopy(source_config)
            run_id = f"confirm_trial{trial_number:04d}_seed{seed}_{batch_id}"
            relative_runs = f"experiments/{study_name}/confirmations/trial_{trial_number:04d}/runs"
            relative_checkpoints = (
                f"models/checkpoints/{study_name}/confirmations/trial_{trial_number:04d}/{batch_id}"
            )
            config["project"]["seed"] = seed
            config["trial"]["run_name"] = run_id
            config["paths"]["experiments"] = relative_runs
            config["paths"]["checkpoints"] = relative_checkpoints
            config["hpo"].update(
                {
                    "confirmation_of_trial": trial_number,
                    "confirmation_seed": seed,
                    "confirmation_batch_id": batch_id,
                    "fixed_confirmation_step": fixed_step,
                }
            )
            config_path = configs_dir / f"trial_{trial_number:04d}_seed{seed}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run_dir = (Path(config["project"]["root"]) / relative_runs / run_id).resolve()
            result_path = results_dir / f"trial_{trial_number:04d}_seed{seed}.json"
            spec = {
                "schema_version": 1,
                "study_name": study_name,
                "trial_number": trial_number,
                "seed": seed,
                "source": f"confirmation:trial_{trial_number:04d}",
                "parameters": params,
                "run_id": run_id,
                "config_path": str(config_path.resolve()),
                "result_path": str(result_path.resolve()),
                "experiment_dir": str(run_dir),
            }
            spec_path = specs_dir / f"trial_{trial_number:04d}_seed{seed}.json"
            write_json(spec_path, spec)
            spec_paths.append(spec_path)
            candidate["runs"].append(
                {"seed": seed, "run_id": run_id, "run_dir": str(run_dir), "reused": False, "result_path": str(result_path)}
            )
        candidates.append(candidate)

    spec_paths.sort(key=confirmation_spec_sort_key)
    launch_order = [
        {
            "trial_number": int(spec["trial_number"]),
            "seed": int(spec["seed"]),
            "spec_path": str(path.resolve()),
        }
        for path in spec_paths
        for spec in [json.loads(path.read_text(encoding="utf-8"))]
    ]
    (batch_dir / "trial_specs.txt").write_text(
        "".join(f"{path.resolve()}\n" for path in spec_paths), encoding="utf-8"
    )
    write_json(
        batch_dir / "batch_manifest.json",
        {
            "schema_version": 1,
            "batch_id": batch_id,
            "created_at": now(),
            "study_name": study_name,
            "candidate_trials": candidate_trials,
            "seeds": seeds,
            "new_run_count": len(spec_paths),
            "launch_order": launch_order,
            "candidates": candidates,
            "robustness_rule": {
                "pooled_task_level_p_own_delta_lcb_positive": True,
                "each_seed_p_own_delta_positive": True,
                "each_seed_pass_at_1_drop_at_least": -0.02,
                "mean_sampled_fooling_delta_positive": True,
                "mean_argmax_fooling_delta_positive": True,
            },
            "full_training_launched": False,
        },
    )
    return batch_dir.resolve()


def confirmation_spec_sort_key(path: Path) -> tuple[int, int]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    return int(spec["seed"]), int(spec["trial_number"])


def collect_confirmation(batch_dir: Path) -> dict[str, Any]:
    batch_dir = batch_dir.expanduser().resolve()
    manifest = json.loads((batch_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    missing = []
    failed = []
    for candidate in manifest["candidates"]:
        for run in candidate["runs"]:
            if run["reused"]:
                continue
            result_path = Path(run["result_path"])
            if not result_path.exists():
                missing.append(str(result_path))
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "complete":
                failed.append({"path": str(result_path), "result": result})
    if missing or failed:
        raise RuntimeError(f"Confirmation is incomplete; missing={missing}, failed={len(failed)}")

    candidates = []
    for candidate in manifest["candidates"]:
        fixed_step = int(candidate["fixed_confirmation_step"])
        seed_rows = [
            metrics_for_run(
                Path(run["run_dir"]),
                int(run["seed"]),
                fixed_step,
                reused=bool(run["reused"]),
            )
            for run in candidate["runs"]
        ]
        aggregate = aggregate_candidate(seed_rows)
        candidates.append(
            {
                "trial_number": candidate["trial_number"],
                "parameters": candidate["parameters"],
                "fixed_confirmation_step": fixed_step,
                "seeds": seed_rows,
                **aggregate,
            }
        )
    candidates.sort(key=lambda row: float(row["pooled_p_own_delta_lcb"]), reverse=True)
    confirmation_providers = {
        provider
        for candidate in candidates
        for provider in candidate["observed_providers"]
    }
    provider_consistent_across_candidates = len(confirmation_providers) == 1
    baseline_generation_hashes = {
        str(seed["baseline_generation_manifest_sha256"])
        for candidate in candidates
        for seed in candidate["seeds"]
    }
    baseline_generation_consistent_across_candidates = (
        len(baseline_generation_hashes) == 1
    )
    if not (
        provider_consistent_across_candidates
        and baseline_generation_consistent_across_candidates
    ):
        for candidate in candidates:
            candidate["protocol_ok"] = False
            candidate["robust"] = False
    candidate_comparisons = [
        compare_candidates(left, right)
        for left, right in combinations(candidates, 2)
    ]
    summary = {
        "schema_version": 1,
        "created_at": now(),
        "study_name": manifest["study_name"],
        "batch_id": manifest["batch_id"],
        "candidates": candidates,
        "observed_providers": sorted(confirmation_providers),
        "provider_consistent_across_candidates": provider_consistent_across_candidates,
        "baseline_generation_manifest_sha256": (
            next(iter(baseline_generation_hashes))
            if baseline_generation_consistent_across_candidates
            else None
        ),
        "baseline_generation_consistent_across_candidates": (
            baseline_generation_consistent_across_candidates
        ),
        "robust_candidates": [row["trial_number"] for row in candidates if row["robust"]],
        "candidate_comparisons": candidate_comparisons,
        "full_training_launched": False,
    }
    write_json(batch_dir / "confirmation_summary.json", summary)
    (batch_dir / "confirmation_summary.md").write_text(render_summary(summary), encoding="utf-8")
    manifest.update({"status": "collected", "collected_at": now()})
    write_json(batch_dir / "batch_manifest.json", manifest)
    return summary


def metrics_for_run(
    run_dir: Path, seed: int, fixed_step: int, *, reused: bool = False
) -> dict[str, Any]:
    evaluation = json.loads((run_dir / "hpo_evaluation_summary.json").read_text(encoding="utf-8"))
    training = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    objective = evaluation["objective"]
    comparison = next(item for item in objective["comparisons"] if int(item["step"]) == fixed_step)
    states = {int(item["step"]): item for item in evaluation["validation"]}
    baseline = states[0]
    selected = states[fixed_step]
    baseline_rows = read_rows(run_dir, 0)
    selected_rows = read_rows(run_dir, fixed_step)
    baseline_by_id = {str(row["task_id"]): row for row in baseline_rows}
    selected_by_id = {str(row["task_id"]): row for row in selected_rows}
    if baseline_by_id.keys() != selected_by_id.keys():
        raise RuntimeError(f"Mismatched validation rows in {run_dir}")
    task_deltas = {
        task_id: float(selected_by_id[task_id]["p_own"]) - float(baseline_by_id[task_id]["p_own"])
        for task_id in baseline_by_id
    }
    sampled_delta = float(selected["fooling_rate"]) - float(baseline["fooling_rate"])
    argmax_baseline = statistics.mean(float(row["delta"]) > 0 for row in baseline_rows)
    argmax_selected = statistics.mean(float(row["delta"]) > 0 for row in selected_rows)
    training_cost = float(
        (((training.get("reward_statistics") or {}).get("usage") or {}).get("cost", 0.0))
        or 0.0
    )
    evaluation_cost = sum(
        float(((state.get("reward_usage") or {}).get("cost", 0.0)) or 0.0)
        for state in states.values()
    )
    reward_stats = training.get("reward_statistics") or {}
    reward_audit = training.get("reward_audit") or {}
    expected_calls = int(config["reward"]["max_reward_calls"])
    expected_evaluation_calls = (
        int(config["evaluation"]["validation"]["count"])
        * int(config["evaluation"]["validation"]["num_generations"])
    )
    expected_model = str(config["discriminator"]["model_id"])
    expected_reward = str(config["reward"]["value"])
    expected_groups = int(config["trial"]["prompt_count"])
    expected_generations = int(config["training"]["num_generations"])
    groups = reward_stats.get("by_task") or {}
    reward_logs = [reward_log_signature(run_dir / "reward_calls.jsonl")]
    reward_logs.extend(
        reward_log_signature(
            run_dir
            / "evaluations"
            / "validation"
            / f"step_{step:04d}"
            / "reward_calls.jsonl"
        )
        for step in sorted(states)
    )
    all_providers = set().union(*(entry["providers"] for entry in reward_logs))
    all_raw_models = set().union(*(entry["raw_models"] for entry in reward_logs))
    all_reward_values = set().union(*(entry["reward_values"] for entry in reward_logs))
    all_prompt_versions = set().union(*(entry["prompt_versions"] for entry in reward_logs))
    all_prompt_hashes = set().union(*(entry["prompt_hashes"] for entry in reward_logs))
    all_system_hashes = set().union(*(entry["system_hashes"] for entry in reward_logs))
    protocol_checks = {
        "reward_is_log_p_own": expected_reward == "log_p_own",
        "logical_reward_calls_match": reward_audit.get("logical_reward_calls")
        == expected_calls,
        "all_training_rewards_succeeded": reward_stats.get("n_ok") == expected_calls,
        "k_is_four": expected_generations == 4
        and int(training.get("num_generations", -1)) == 4,
        "training_group_count_matches": len(groups) == expected_groups,
        "all_training_groups_have_variance": len(groups) == expected_groups
        and all(float(group.get("std", 0.0)) > 0 for group in groups.values()),
        "all_training_groups_have_four_unique_candidates": len(groups)
        == expected_groups
        and all(
            int(group.get("unique_candidates", 0)) == expected_generations
            for group in groups.values()
        ),
        "training_reward_log_complete": reward_logs[0]["n_ok"] == expected_calls,
        "all_validation_reward_logs_complete": all(
            entry["n_ok"] == expected_evaluation_calls for entry in reward_logs[1:]
        ),
        "discriminator_route_matches": all_raw_models == {expected_model},
        "reward_value_matches": all_reward_values == {expected_reward},
        "judge_prompt_version_matches": all_prompt_versions
        == {ATTRIBUTION_PROMPT_VERSION},
        "single_judge_prompt_template_hash": len(all_prompt_hashes) == 1,
        "single_judge_system_prompt_hash": len(all_system_hashes) == 1,
        "single_exposed_provider": len(all_providers) == 1,
    }
    observed_providers = sorted(all_providers)
    return {
        "seed": seed,
        "run_id": run_dir.name,
        "reused": reused,
        "fixed_step": fixed_step,
        "p_own_delta": comparison["p_own_delta_mean"],
        "p_own_delta_lcb": comparison["p_own_delta_lcb"],
        "pass_at_1_delta": comparison["pass_at_1_delta"],
        "net_passed_change": comparison["net_passed_change"],
        "quality_feasible": comparison["quality_feasible"],
        "sampled_fooling_delta": sampled_delta,
        "argmax_fooling_delta": argmax_selected - argmax_baseline,
        "training_cost": training_cost,
        "evaluation_cost": evaluation_cost,
        "total_openrouter_cost": training_cost + evaluation_cost,
        "training_prompt_manifest_sha256": training_prompt_hash(run_dir),
        "baseline_generation_manifest_sha256": generation_manifest_hash(
            baseline_rows
        ),
        "protocol_checks": protocol_checks,
        "protocol_ok": all(protocol_checks.values()),
        "observed_providers": observed_providers,
        "task_p_own_deltas": task_deltas,
    }


def reward_log_signature(path: Path) -> dict[str, Any]:
    rows = [row for row in read_jsonl(path) if row.get("status") == "ok"]

    def values(field: str) -> set[str]:
        return {
            str(row[field])
            for row in rows
            if row.get(field) is not None and str(row[field]) != "None"
        }

    return {
        "n_ok": len(rows),
        "providers": values("raw_provider"),
        "raw_models": values("raw_model"),
        "reward_values": values("reward_value"),
        "prompt_versions": values("judge_prompt_template_version"),
        "prompt_hashes": values("judge_prompt_template_sha256"),
        "system_hashes": values("judge_system_prompt_sha256"),
    }


def aggregate_candidate(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = set(seed_rows[0]["task_p_own_deltas"])
    if any(set(row["task_p_own_deltas"]) != task_ids for row in seed_rows):
        raise ValueError("Confirmation seeds do not use identical validation tasks")
    task_means = [
        statistics.mean(row["task_p_own_deltas"][task_id] for row in seed_rows)
        for task_id in sorted(task_ids, key=int)
    ]
    mean_delta = statistics.mean(task_means)
    se = statistics.stdev(task_means) / math.sqrt(len(task_means)) if len(task_means) > 1 else 0.0
    pooled_lcb = mean_delta - Z_ONE_SIDED_95 * se
    sampled_mean = statistics.mean(float(row["sampled_fooling_delta"]) for row in seed_rows)
    argmax_mean = statistics.mean(float(row["argmax_fooling_delta"]) for row in seed_rows)
    training_prompt_hashes = {
        str(row["training_prompt_manifest_sha256"]) for row in seed_rows
    }
    baseline_generation_hashes = {
        str(row["baseline_generation_manifest_sha256"]) for row in seed_rows
    }
    observed_providers = {
        provider for row in seed_rows for provider in row["observed_providers"]
    }
    protocol_ok = all(bool(row["protocol_ok"]) for row in seed_rows) and len(
        training_prompt_hashes
    ) == 1 and len(baseline_generation_hashes) == 1 and len(observed_providers) == 1
    robust = (
        pooled_lcb > 0
        and all(float(row["p_own_delta"]) > 0 for row in seed_rows)
        and all(bool(row["quality_feasible"]) for row in seed_rows)
        and sampled_mean > 0
        and argmax_mean > 0
        and protocol_ok
    )
    return {
        "pooled_p_own_delta_mean": mean_delta,
        "pooled_p_own_delta_standard_error": se,
        "pooled_p_own_delta_lcb": pooled_lcb,
        "mean_sampled_fooling_delta": sampled_mean,
        "mean_argmax_fooling_delta": argmax_mean,
        "total_openrouter_cost": sum(
            float(row.get("total_openrouter_cost", 0.0)) for row in seed_rows
        ),
        "incremental_confirmation_cost": sum(
            float(row.get("total_openrouter_cost", 0.0))
            for row in seed_rows
            if not bool(row.get("reused"))
        ),
        "reused_source_cost": sum(
            float(row.get("total_openrouter_cost", 0.0))
            for row in seed_rows
            if bool(row.get("reused"))
        ),
        "all_seed_p_own_deltas_positive": all(float(row["p_own_delta"]) > 0 for row in seed_rows),
        "all_seed_quality_feasible": all(bool(row["quality_feasible"]) for row in seed_rows),
        "training_prompt_manifest_sha256": next(iter(training_prompt_hashes)),
        "baseline_generation_manifest_sha256": next(
            iter(baseline_generation_hashes)
        ),
        "observed_providers": sorted(observed_providers),
        "protocol_ok": protocol_ok,
        "robust": robust,
    }


def compare_candidates(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    left_by_seed = {int(row["seed"]): row for row in left["seeds"]}
    right_by_seed = {int(row["seed"]): row for row in right["seeds"]}
    if left_by_seed.keys() != right_by_seed.keys():
        raise ValueError("Candidates do not use identical confirmation seeds")
    task_differences_by_seed: dict[int, dict[str, float]] = {}
    per_seed = []
    for seed in sorted(left_by_seed):
        left_tasks = left_by_seed[seed]["task_p_own_deltas"]
        right_tasks = right_by_seed[seed]["task_p_own_deltas"]
        if left_tasks.keys() != right_tasks.keys():
            raise ValueError("Candidates do not use identical validation tasks")
        differences = {
            task_id: float(left_tasks[task_id]) - float(right_tasks[task_id])
            for task_id in left_tasks
        }
        task_differences_by_seed[seed] = differences
        per_seed.append(
            {"seed": seed, "p_own_delta_difference": statistics.mean(differences.values())}
        )
    task_ids = set(next(iter(task_differences_by_seed.values())))
    task_means = [
        statistics.mean(
            task_differences_by_seed[seed][task_id] for seed in sorted(left_by_seed)
        )
        for task_id in sorted(task_ids, key=int)
    ]
    mean_difference = statistics.mean(task_means)
    standard_error = (
        statistics.stdev(task_means) / math.sqrt(len(task_means))
        if len(task_means) > 1
        else 0.0
    )
    lcb = mean_difference - Z_ONE_SIDED_95 * standard_error
    return {
        "left_trial": int(left["trial_number"]),
        "right_trial": int(right["trial_number"]),
        "interpretation": "left_minus_right",
        "p_own_delta_difference_mean": mean_difference,
        "p_own_delta_difference_standard_error": standard_error,
        "p_own_delta_difference_lcb": lcb,
        "per_seed": per_seed,
        "left_superiority_supported": lcb > 0,
    }


def training_prompt_hash(run_dir: Path) -> str:
    rows = [
        {
            "task_id": str(row["task_id"]),
            "dataset_index": int(row["dataset_index"]),
            "task_statement": str(row["task_statement"]),
        }
        for row in read_jsonl(run_dir / "train_examples.jsonl")
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generation_manifest_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [
            {
                "task_id": str(row["task_id"]),
                "rollout_index": int(row["rollout_index"]),
                "candidate_raw": str(row["candidate_raw"]),
            }
            for row in rows
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_rows(run_dir: Path, step: int) -> list[dict[str, Any]]:
    path = run_dir / "evaluations" / "validation" / f"step_{step:04d}" / "rows.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Multi-seed confirmation",
        "",
        "| Source trial | Fixed step | Pooled Δ p_own | Pooled LCB | Mean Δ sampled fooling | Mean Δ argmax fooling | Incremental cost | Protocol | Robust |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in summary["candidates"]:
        lines.append(
            f"| {row['trial_number']} | {row['fixed_confirmation_step']} | "
            f"{row['pooled_p_own_delta_mean']:+.4f} | {row['pooled_p_own_delta_lcb']:+.4f} | "
            f"{row['mean_sampled_fooling_delta']:+.4f} | {row['mean_argmax_fooling_delta']:+.4f} | "
            f"${row['incremental_confirmation_cost']:.4f} | "
            f"{'yes' if row['protocol_ok'] else 'no'} | "
            f"{'yes' if row['robust'] else 'no'} |"
        )
    if summary.get("candidate_comparisons"):
        lines.extend(
            [
                "",
                "## Direct paired candidate comparison",
                "",
                "| Left - right | Mean difference | One-sided 95% LCB | Superiority supported |",
                "|---|---:|---:|:---:|",
            ]
        )
        for row in summary["candidate_comparisons"]:
            lines.append(
                f"| trial {row['left_trial']} - trial {row['right_trial']} | "
                f"{row['p_own_delta_difference_mean']:+.4f} | "
                f"{row['p_own_delta_difference_lcb']:+.4f} | "
                f"{'yes' if row['left_superiority_supported'] else 'no'} |"
            )
    lines.extend(["", "No full training is launched by this confirmation workflow.", ""])
    return "\n".join(lines)


def parse_integers(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


if __name__ == "__main__":
    main()
