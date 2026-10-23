from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iar_phase0.data import write_jsonl
from iar_phase0.env import load_env_file
from iar_phase0.evaluate import run_mbpp_tests
from iar_phase0.generator import strip_code_fences
from iar_phase0.parameters import initialize_parameter_record
from iar_phase0.config import load_config as load_phase0_config
from iar_phase0.data import load_mbpp, materialize_split

from .config import load_grpo_config
from .data import build_trial_dataset
from .reward import LiveOpenRouterReward, settings_from_config


def summarize_probe_groups(
    reward_rows: list[dict[str, Any]],
    *,
    expected_rollouts: int,
    completion_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit whether stochastic rollouts expose a usable within-group signal."""

    def reward_value(row: dict[str, Any]) -> float:
        # Historical p_own pilots predate the explicit transformed-reward
        # field. For those records the training reward was p_own itself.
        return float(row["reward"] if row.get("reward") is not None else row["p_own"])

    def argmax_accept(row: dict[str, Any]) -> int:
        if row.get("logp_1") is not None and row.get("logp_0") is not None:
            return int(float(row["logp_1"]) > float(row["logp_0"]))
        # p_own is the binary-normalized probability in legacy audit rows.
        return int(float(row["p_own"]) > 0.5)

    completions_by_key = {
        (str(row["task_id"]), int(row["rollout_index"])): row
        for row in (completion_rows or [])
    }
    ok_rows: list[dict[str, Any]] = []
    for source_row in reward_rows:
        if source_row.get("status") != "ok":
            continue
        row = dict(source_row)
        completion = completions_by_key.get(
            (str(row["task_id"]), int(row["rollout_index"]))
        )
        if completion is not None:
            row["unit_tests"] = completion.get("unit_tests")
            row["completion_tokens"] = completion.get("completion_tokens")
        ok_rows.append(row)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        by_task[str(row["task_id"])].append(row)

    groups: list[dict[str, Any]] = []
    for task_id in sorted(by_task, key=lambda value: int(value) if value.isdigit() else value):
        rows = sorted(by_task[task_id], key=lambda row: int(row["rollout_index"]))
        p_own = [float(row["p_own"]) for row in rows]
        rewards = [reward_value(row) for row in rows]
        p_own_variance = statistics.pvariance(p_own) if len(p_own) > 1 else None
        reward_variance = statistics.pvariance(rewards) if len(rewards) > 1 else None
        emitted = [
            1 if str(row.get("judge_token", "")).strip().startswith("1") else 0
            for row in rows
        ]
        argmax = [argmax_accept(row) for row in rows]
        correct = [
            bool((row.get("unit_tests") or {}).get("passed")) for row in rows
        ]
        groups.append(
            {
                "task_id": task_id,
                "n_rollouts": len(rows),
                "reward": rewards,
                "reward_min": min(rewards) if rewards else None,
                "reward_max": max(rewards) if rewards else None,
                "reward_variance": reward_variance,
                "nonzero_reward_variance": bool(
                    reward_variance is not None and reward_variance > 0.0
                ),
                "p_own": p_own,
                "p_own_min": min(p_own) if p_own else None,
                "p_own_max": max(p_own) if p_own else None,
                "p_own_variance": p_own_variance,
                "nonzero_p_own_variance": bool(
                    p_own_variance is not None and p_own_variance > 0.0
                ),
                "emitted_accept_count": sum(emitted),
                "argmax_accept_count": sum(argmax),
                "all_same_emitted_label": len(set(emitted)) <= 1,
                "all_same_argmax_label": len(set(argmax)) <= 1,
                "n_correct": sum(correct),
                "any_correct": any(correct),
                "mixed_correctness": len(set(correct)) > 1,
                "distinct_candidates": len({str(row.get("candidate", "")) for row in rows}),
            }
        )

    complete_groups = [group for group in groups if group["n_rollouts"] == expected_rollouts]
    variable_reward_groups = [
        group for group in complete_groups if group["nonzero_reward_variance"]
    ]
    variable_p_own_groups = [
        group for group in complete_groups if group["nonzero_p_own_variance"]
    ]
    binary_mass = [float(row["binary_label_mass"]) for row in ok_rows]
    emitted_labels = [
        1 if str(row.get("judge_token", "")).strip().startswith("1") else 0
        for row in ok_rows
    ]
    argmax_labels = [argmax_accept(row) for row in ok_rows]
    correct_rows = [
        row for row in ok_rows if bool((row.get("unit_tests") or {}).get("passed"))
    ]
    incorrect_rows = [
        row for row in ok_rows if not bool((row.get("unit_tests") or {}).get("passed"))
    ]

    def average(rows: list[dict[str, Any]], key: str) -> float | None:
        if not rows:
            return None
        values = (
            [reward_value(row) for row in rows]
            if key == "reward"
            else [float(row[key]) for row in rows]
        )
        return statistics.mean(values)

    return {
        "n_reward_rows": len(reward_rows),
        "n_ok_reward_rows": len(ok_rows),
        "expected_rollouts_per_group": expected_rollouts,
        "n_groups": len(groups),
        "n_complete_groups": len(complete_groups),
        "n_groups_with_nonzero_reward_variance": len(variable_reward_groups),
        "n_groups_with_nonzero_p_own_variance": len(variable_p_own_groups),
        "all_groups_complete": len(complete_groups) == len(groups),
        "all_complete_groups_have_nonzero_reward_variance": (
            bool(complete_groups)
            and len(variable_reward_groups) == len(complete_groups)
        ),
        "all_complete_groups_have_nonzero_p_own_variance": (
            bool(complete_groups)
            and len(variable_p_own_groups) == len(complete_groups)
        ),
        "fooling_emitted": statistics.mean(emitted_labels) if emitted_labels else None,
        "fooling_argmax": statistics.mean(argmax_labels) if argmax_labels else None,
        "emitted_argmax_consistency": (
            statistics.mean(
                float(emitted == argmax)
                for emitted, argmax in zip(emitted_labels, argmax_labels)
            )
            if emitted_labels
            else None
        ),
        "correctness": {
            "rollout_pass_at_1": (
                len(correct_rows) / len(ok_rows) if ok_rows else None
            ),
            "groups_with_at_least_one_correct": sum(
                bool(group["any_correct"]) for group in complete_groups
            ),
            "groups_with_mixed_correctness": sum(
                bool(group["mixed_correctness"]) for group in complete_groups
            ),
            "mean_reward_correct": average(correct_rows, "reward"),
            "mean_reward_incorrect": average(incorrect_rows, "reward"),
            "mean_p_own_correct": average(correct_rows, "p_own"),
            "mean_p_own_incorrect": average(incorrect_rows, "p_own"),
        },
        "group_details": groups,
        "raw_models": dict(Counter(str(row.get("raw_model")) for row in ok_rows)),
        "raw_providers": dict(Counter(str(row.get("raw_provider")) for row in ok_rows)),
        "binary_label_mass": {
            "min": min(binary_mass) if binary_mass else None,
            "mean": statistics.mean(binary_mass) if binary_mass else None,
            "max": max(binary_mass) if binary_mass else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Coder GRPO-style rollouts and audit live reward variation without updates."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Reuse generated_rollouts.jsonl and score only candidates without a successful reward.",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    config = load_grpo_config(args.config)
    probe_cfg = config.raw.get("reward_probe") or {}
    prompt_count = int(probe_cfg.get("prompt_count", 25))
    if prompt_count <= 0 or prompt_count > config.prompt_count:
        raise ValueError("reward_probe.prompt_count must be in [1, trial.prompt_count]")

    dataset, dataset_rows = build_trial_dataset(config)
    selected_dataset = dataset.select(range(prompt_count))
    selected_rows = dataset_rows[:prompt_count]
    expected_reward_calls = prompt_count * config.num_generations

    if args.resume_dir is None:
        run_prefix = str(probe_cfg.get("run_name", "reward_probe"))
        run_id = datetime.now(timezone.utc).strftime(f"{run_prefix}_%Y%m%dT%H%M%SZ")
        out_dir = config.resolve(config.raw["paths"]["experiments"]) / "reward_probe" / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        initialize_parameter_record(
            out_dir,
            resolved_config=config.raw,
            source_config=config.path,
        )
        write_jsonl(out_dir / "probe_examples.jsonl", selected_rows)
    else:
        out_dir = args.resume_dir
        if not out_dir.is_absolute():
            out_dir = config.root / out_dir
        out_dir = out_dir.resolve()
        run_id = out_dir.name
        if not out_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {out_dir}")

    actor = config.raw["actor"]
    training = config.raw["training"]
    actor_source = config.actor_source()
    completion_path = out_dir / "generated_rollouts.jsonl"
    if args.resume_dir is None:
        cache_dir = config.resolve(config.raw["paths"]["hf_cache"])
        tokenizer = AutoTokenizer.from_pretrained(
            actor_source,
            cache_dir=str(cache_dir),
            local_files_only=bool(actor["local_files_only"]),
            trust_remote_code=False,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            actor_source,
            cache_dir=str(cache_dir),
            local_files_only=bool(actor["local_files_only"]),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )
        model.eval()
        set_seed(int(config.raw["project"]["seed"]))

        completion_rows: list[dict[str, Any]] = []
        for example in selected_dataset:
            conversations = [example["prompt"] for _ in range(config.num_generations)]
            inputs = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=int(training["max_completion_length"]),
                    do_sample=True,
                    temperature=float(training["temperature"]),
                    top_p=float(training["top_p"]),
                    top_k=0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_length = inputs["input_ids"].shape[1]
            for rollout_index, output in enumerate(outputs):
                completion_ids = output[prompt_length:]
                eos_positions = (completion_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                terminated = len(eos_positions) > 0
                if terminated:
                    completion_ids = completion_ids[: int(eos_positions[0]) + 1]
                raw_candidate = tokenizer.decode(completion_ids, skip_special_tokens=True)
                completion_rows.append(
                    {
                        "task_id": str(example["task_id"]),
                        "dataset_index": int(example["dataset_index"]),
                        "rollout_index": rollout_index,
                        "task_statement": str(example["task_statement"]),
                        "candidate_raw": raw_candidate,
                        "candidate": strip_code_fences(raw_candidate),
                        "completion_tokens": len(completion_ids),
                        "terminated_with_eos": terminated,
                    }
                )
    else:
        if not completion_path.exists():
            raise FileNotFoundError(f"Missing generated rollouts: {completion_path}")
        completion_rows = [
            json.loads(line)
            for line in completion_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(completion_rows) != expected_reward_calls:
            raise RuntimeError(
                f"Expected {expected_reward_calls} cached rollouts, found {len(completion_rows)}"
            )

    phase0_config = load_phase0_config(
        config.resolve(config.raw["dataset"]["phase0_config"])
    )
    mbpp = load_mbpp(phase0_config)
    source_examples = materialize_split(
        mbpp, split=str(config.raw["dataset"]["split"])
    )
    examples_by_index = {int(example.index): example for example in source_examples}
    unit_test_timeout = int(config.raw["evaluation"]["unit_test_timeout_seconds"])
    for row in completion_rows:
        if args.resume_dir is not None and row.get("unit_tests") is not None:
            continue
        example = examples_by_index[int(row["dataset_index"])]
        result = run_mbpp_tests(
            str(row["candidate"]),
            example.tests,
            timeout_seconds=unit_test_timeout,
            test_setup_code=example.test_setup_code,
        )
        row["unit_tests"] = {
            "passed": result.passed,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    if args.resume_dir is None:
        write_jsonl(completion_path, completion_rows)

    env_path = config.resolve(config.raw["paths"]["openrouter_env"])
    load_env_file(env_path)
    reward_log_path = out_dir / "reward_calls.jsonl"
    existing_reward_rows = (
        [
            json.loads(line)
            for line in reward_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if reward_log_path.exists()
        else []
    )
    successful_keys = {
        (str(row["task_id"]), int(row["rollout_index"]))
        for row in existing_reward_rows
        if row.get("status") == "ok"
    }
    pending_rows = [
        row
        for row in completion_rows
        if (str(row["task_id"]), int(row["rollout_index"])) not in successful_keys
    ]
    settings = replace(
        settings_from_config(config.raw),
        max_reward_calls=len(pending_rows),
        max_http_requests=len(pending_rows) * int(config.raw["reward"]["max_attempts_per_call"]),
    )
    runtime = LiveOpenRouterReward(
        settings=settings,
        log_path=reward_log_path,
        call_id_offset=max((int(row.get("call_id", 0)) for row in existing_reward_rows), default=0),
        http_request_id_offset=max(
            (int(row.get("http_request_id", 0)) for row in existing_reward_rows),
            default=0,
        ),
    )
    try:
        if pending_rows:
            asyncio.run(
                runtime.score_batch(
                    completions=[row["candidate_raw"] for row in pending_rows],
                    task_statement=[row["task_statement"] for row in pending_rows],
                    task_id=[row["task_id"] for row in pending_rows],
                    dataset_index=[row["dataset_index"] for row in pending_rows],
                    rollout_index=[int(row["rollout_index"]) for row in pending_rows],
                )
            )
        runtime.assert_complete()
        reward_rows = [
            json.loads(line)
            for line in (out_dir / "reward_calls.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        group_audit = summarize_probe_groups(
            reward_rows,
            expected_rollouts=config.num_generations,
            completion_rows=completion_rows,
        )
        if group_audit["n_ok_reward_rows"] != expected_reward_calls:
            raise RuntimeError(
                "Reward probe is incomplete: "
                f"{group_audit['n_ok_reward_rows']}/{expected_reward_calls} successful judgments"
            )
        summary = {
            "run_id": run_id,
            "status": "complete",
            "actor_model": str(actor["model_id"]),
            "actor_source": actor_source,
            "prompt_count": prompt_count,
            "num_generations": config.num_generations,
            "reward_audit": runtime.audit_summary(),
            "resume": {
                "resumed": args.resume_dir is not None,
                "successful_before_resume": len(successful_keys),
                "scored_in_session": len(pending_rows),
                "successful_total": group_audit["n_ok_reward_rows"],
            },
            "reward_statistics": runtime.reward_statistics(),
            "group_variation": group_audit,
            "termination": {
                "terminated": sum(bool(row["terminated_with_eos"]) for row in completion_rows),
                "n": len(completion_rows),
            },
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"run_dir": str(out_dir), "summary": summary}, indent=2, ensure_ascii=False))
    except Exception as exc:
        failure = {
            "run_id": run_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "reward_audit": runtime.audit_summary(),
        }
        (out_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
