from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from iar_phase0.parameters import write_parameter_artifact

from .preference_data import resolve
from .teacher_data import read_jsonl, sha256_text, write_json, write_jsonl


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def adapter_map(
    config: dict[str, Any], root: Path, domain: str, method: str, run_id: str
) -> dict[int, Path | None]:
    domain_cfg = config["domains"][domain]
    checkpoint_dir = resolve(root, config["paths"]["checkpoints"]) / domain / method / run_id
    epoch_step = int(domain_cfg["steps_per_epoch"])
    final_step = int(domain_cfg["max_steps"])
    if final_step % epoch_step:
        raise ValueError(
            f"max_steps ({final_step}) is not an integer number of epochs "
            f"for steps_per_epoch={epoch_step}"
        )
    adapters: dict[int, Path | None] = {0: None}
    for step in range(epoch_step, final_step + 1, epoch_step):
        adapters[step] = (
            checkpoint_dir / "final_adapter"
            if step == final_step
            else checkpoint_dir / f"checkpoint-{step}"
        )
    for step, path in adapters.items():
        if step and (path is None or not (path / "adapter_model.safetensors").exists()):
            raise FileNotFoundError(f"Missing adapter for {domain}/{method} step {step}: {path}")
    return adapters


def load_splits(root: Path, domain: str) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    if domain == "mbpp":
        from iar_grpo.config import GrpoTrialConfig
        from iar_grpo.evaluate_pilot import select_examples
        from iar_phase0.config import load_config
        from iar_phase0.data import load_mbpp

        path = root / "configs" / "grpo_primary_trial17.yaml"
        raw = load_yaml(path)
        wrapper = GrpoTrialConfig(raw=raw, path=path)
        phase0 = load_config(resolve(root, raw["dataset"]["phase0_config"]))
        dataset = load_mbpp(phase0)
        return wrapper, {
            "validation": select_examples(dataset, raw["evaluation"]["validation"]),
            "test": select_examples(dataset, raw["evaluation"]["final_test"]),
        }
    if domain == "commonsenseqa":
        from iar_cqa.train import resolve as cqa_resolve
        from iar_cqa.train import select_balanced

        raw = load_yaml(root / "configs" / "cqa_grpo_full_log_p_own_v2.yaml")
        split_dir = cqa_resolve(root, raw["paths"]["split_dir"])
        validation = select_balanced(
            read_jsonl(split_dir / "validation.jsonl"),
            count=int(raw["evaluation"]["validation_count"]),
            seed=int(raw["project"]["seed"]) + 1_000_003,
        )
        test = read_jsonl(split_dir / "test.jsonl")
        return raw, {"validation": validation, "test": test}
    if domain == "gsm8k":
        from iar_gsm8k.train import frozen_subset

        raw = load_yaml(root / "configs" / "gsm8k_grpo_official_log_p_own_v2.yaml")
        split_dir = resolve(root, raw["paths"]["split_dir"])
        validation = frozen_subset(
            read_jsonl(split_dir / str(raw["dataset"]["validation_file"])),
            count=int(raw["dataset"]["validation_count"]),
        )
        test = read_jsonl(split_dir / str(raw["dataset"]["final_test_file"]))
        return raw, {"validation": validation, "test": test}
    raise ValueError(domain)


def example_ids(domain: str, examples: list[Any]) -> list[str]:
    if domain == "mbpp":
        return [str(example.task_id) for example in examples]
    return [str(example["id"]) for example in examples]


def generate_rows(
    *,
    domain: str,
    base_config: Any,
    root: Path,
    examples: list[Any],
    adapter: Path | None,
    step: int,
) -> list[dict[str, Any]]:
    if domain == "mbpp":
        from iar_grpo.evaluate_pilot import attach_unit_tests, generate_candidates

        rows = generate_candidates(
            config=base_config,
            examples=examples,
            adapter_dir=adapter,
            num_generations=1,
            temperature=0.0,
            top_p=1.0,
            seed=int(base_config.raw["project"]["seed"]),
        )
        attach_unit_tests(
            rows,
            examples,
            int(base_config.raw["evaluation"]["unit_test_timeout_seconds"]),
        )
        for row in rows:
            row["step"] = step
        return rows
    if domain == "commonsenseqa":
        from iar_cqa.evaluate_short import generate

        return generate(base_config, root, examples, adapter, step)
    if domain == "gsm8k":
        from iar_gsm8k.evaluate_short import generate

        return generate(base_config, root, examples, adapter, step)
    raise ValueError(domain)


def row_id(domain: str, row: dict[str, Any]) -> str:
    return str(row["task_id"] if domain == "mbpp" else row["id"])


def is_correct(domain: str, row: dict[str, Any]) -> bool:
    if domain == "mbpp":
        return bool(row["unit_tests"]["passed"])
    return bool(row["score"]["correct"])


def summarize(domain: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(is_correct(domain, row) for row in rows)
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows)}


def paired(domain: str, baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_by_id = {row_id(domain, row): row for row in baseline}
    candidate_by_id = {row_id(domain, row): row for row in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("Paired evaluation IDs differ")
    transitions = {"wrong_to_correct": 0, "correct_to_wrong": 0, "stable_correct": 0, "stable_wrong": 0}
    for item_id in baseline_by_id:
        before = is_correct(domain, baseline_by_id[item_id])
        after = is_correct(domain, candidate_by_id[item_id])
        if not before and after:
            transitions["wrong_to_correct"] += 1
        elif before and not after:
            transitions["correct_to_wrong"] += 1
        elif before:
            transitions["stable_correct"] += 1
        else:
            transitions["stable_wrong"] += 1
    n = len(baseline_by_id)
    return {
        "n": n,
        "accuracy_delta": (
            transitions["wrong_to_correct"] - transitions["correct_to_wrong"]
        )
        / n,
        "net_correct_change": transitions["wrong_to_correct"] - transitions["correct_to_wrong"],
        "transitions": transitions,
    }


def evaluate(
    config: dict[str, Any], root: Path, method: str, domain: str, run_id: str
) -> dict[str, Any]:
    run_dir = resolve(root, config["paths"]["experiments"]) / domain / method / run_id
    training = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if training.get("status") != "complete":
        raise RuntimeError(f"Training is not complete: {run_dir}")
    base_config, splits = load_splits(root, domain)
    adapters = adapter_map(config, root, domain, method, run_id)
    split_manifest = {
        name: {
            "n": len(examples),
            "ordered_ids": example_ids(domain, examples),
            "ids_sha256": sha256_text(json.dumps(example_ids(domain, examples))),
        }
        for name, examples in splits.items()
    }
    write_parameter_artifact(run_dir, "task_evaluation_split_manifest.json", split_manifest)
    all_summaries: dict[str, list[dict[str, Any]]] = {}
    comparisons: dict[str, list[dict[str, Any]]] = {}
    for split_name, examples in splits.items():
        rows_by_step: dict[int, list[dict[str, Any]]] = {}
        summaries: list[dict[str, Any]] = []
        for step, adapter in adapters.items():
            state_dir = run_dir / "evaluations" / "task_performance" / split_name / f"step_{step:04d}"
            rows_path = state_dir / "rows.jsonl"
            summary_path = state_dir / "summary.json"
            if rows_path.exists() and summary_path.exists():
                rows = read_jsonl(rows_path)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                state_dir.mkdir(parents=True, exist_ok=True)
                rows = generate_rows(
                    domain=domain,
                    base_config=base_config,
                    root=root,
                    examples=examples,
                    adapter=adapter,
                    step=step,
                )
                summary = {"split": split_name, "step": step, **summarize(domain, rows)}
                write_jsonl(rows_path, rows)
                write_json(summary_path, summary)
            rows_by_step[step] = rows
            summaries.append(summary)
        baseline = rows_by_step[0]
        all_summaries[split_name] = summaries
        comparisons[split_name] = [
            {"step": step, **paired(domain, baseline, rows_by_step[step])}
            for step in adapters
            if step != 0
        ]
    result = {
        "status": "complete",
        "domain": domain,
        "method": method,
        "run_id": run_id,
        "primary_comparison_checkpoint": int(config["domains"][domain]["max_steps"]),
        "checkpoint_policy": (
            "fixed final-epoch endpoint; intermediate epochs are diagnostic and are not "
            "selected on test"
        ),
        "summaries": all_summaries,
        "paired_comparisons": comparisons,
    }
    write_json(run_dir / "task_performance_evaluation_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SFT/DPO adapters on frozen task splits")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("sft", "sft_matched", "dpo"), required=True)
    parser.add_argument("--domain", choices=("mbpp", "commonsenseqa", "gsm8k"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config.resolve())
    root = Path(config["project"]["root"]).resolve()
    print(json.dumps(evaluate(config, root, args.method, args.domain, args.run_id), indent=2))


if __name__ == "__main__":
    main()
