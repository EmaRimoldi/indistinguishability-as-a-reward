from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import yaml
from tqdm import tqdm

from iar_phase0.env import load_env_file
from iar_phase0.openrouter import OpenRouterClient, response_text

from .data import read_jsonl, write_json, write_jsonl
from .judge import call_judge, resolve
from .prompts import ATTRIBUTION_PROMPT_VERSION, attribution_prompt, sha256_text, task_prompt, template_hashes
from .scoring import parse_answer, response_features


ANSWER_PREFIX_RE = re.compile(r"(?im)(?:#{1,6}\s*)?Answer\s*:")


def split_answer(text: str) -> tuple[list[str], str | None]:
    stripped = (text or "").strip()
    matches = list(ANSWER_PREFIX_RE.finditer(stripped))
    if not matches:
        return stripped.splitlines(), None
    start = matches[-1].start()
    reasoning = stripped[:start].rstrip().splitlines()
    answer = stripped[start:].strip()
    return reasoning, answer


def flatten_layout(text: str) -> str:
    reasoning, answer = split_answer(text)
    body = " ".join(line.strip() for line in reasoning if line.strip())
    return "\n".join(part for part in (body, answer) if part)


def compact_whitespace(text: str) -> str:
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.strip().splitlines() if line.strip())


def remove_conclusion_markers(text: str) -> str:
    reasoning, answer = split_answer(text)
    pattern = re.compile(r"(?i)\b(?:therefore|thus|hence)\b[,;:]?\s*")
    body = [pattern.sub("", line) for line in reasoning]
    return "\n".join(body + ([answer] if answer else []))


def minimize_punctuation(text: str) -> str:
    reasoning, answer = split_answer(text)
    body = [re.sub(r"(?<=[A-Za-z])[,;:](?=\s|$)", "", line) for line in reasoning]
    return "\n".join(body + ([answer] if answer else []))


def change_answer_marker(text: str) -> str:
    reasoning, answer = split_answer(text)
    if answer is None:
        return text
    changed = re.sub(r"^\s*Answer\s*:", "Final answer:", answer, flags=re.I)
    return "\n".join(reasoning + [changed])


TRANSFORMS: dict[str, Callable[[str], str]] = {
    "flatten_layout": flatten_layout,
    "compact_whitespace": compact_whitespace,
    "remove_conclusion_markers": remove_conclusion_markers,
    "minimize_punctuation": minimize_punctuation,
    "change_answer_marker": change_answer_marker,
}


def build_candidates(config: dict[str, Any], root: Path, run_dir: Path) -> list[dict[str, Any]]:
    probe = read_jsonl(run_dir / "probe_judgments.jsonl")
    correct = [row for row in probe if row["score"]["correct"]]
    random.Random(int(config["project"]["seed"]) + 700_001).shuffle(correct)
    pairs_per_transform = int(config["shortcut_audit"]["pairs_per_transform"])
    examples = {row["id"]: row for row in read_jsonl(resolve(root, config["paths"]["split_dir"]) / "reward_audit.jsonl")}
    candidates: list[dict[str, Any]] = []
    for transform_name, transform in TRANSFORMS.items():
        eligible = []
        for row in correct:
            transformed = transform(row["response"])
            if transformed.strip() != row["response"].strip():
                eligible.append((row, transformed))
            if len(eligible) == pairs_per_transform:
                break
        for pair_index, (row, transformed) in enumerate(eligible):
            original_answer = parse_answer(row["response"])
            transformed_answer = parse_answer(transformed)
            if transform_name != "change_answer_marker" and transformed_answer != original_answer:
                raise ValueError(f"{transform_name} changed the parsed answer")
            task = task_prompt(examples[row["id"]]["question"])
            for state, response in (("original", row["response"]), ("transformed", transformed)):
                candidates.append(
                    {
                        "pair_id": f"{transform_name}:{pair_index:03d}",
                        "transform": transform_name,
                        "pair_state": state,
                        "id": row["id"],
                        "source_index": row["source_index"],
                        "rollout_index": row["rollout_index"],
                        "response": response,
                        "inherited_correctness": True,
                        "original_answer": original_answer,
                        "parsed_answer": parse_answer(response),
                        "task_statement": task,
                        "features": response_features(response, examples[row["id"]]["question"]),
                    }
                )
    expected = pairs_per_transform * 2 * len(TRANSFORMS)
    if len(candidates) != expected:
        counts = Counter(row["transform"] for row in candidates)
        raise ValueError(f"Insufficient eligible shortcut pairs: {counts}; expected {expected}")
    random.Random(int(config["project"]["seed"]) + 700_003).shuffle(candidates)
    write_jsonl(run_dir / "shortcut_candidates.jsonl", candidates)
    return candidates


def judge_candidates(config: dict[str, Any], root: Path, run_dir: Path) -> None:
    path = run_dir / "shortcut_candidates.jsonl"
    candidates = read_jsonl(path) if path.exists() else build_candidates(config, root, run_dir)
    output_path = run_dir / "shortcut_judgments.jsonl"
    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {(row["pair_id"], row["pair_state"]) for row in existing}
    disc = config["discriminator"]
    transport = disc["transport"]
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    client = OpenRouterClient.from_env(
        max_attempts=int(transport["max_attempts"]),
        max_retry_wait_seconds=float(transport["max_retry_wait_seconds"]),
    )

    def score_one(candidate: dict[str, Any]) -> dict[str, Any]:
        prompt = attribution_prompt(candidate["task_statement"], candidate["response"])
        started = time.time()
        raw, parsed = call_judge(client=client, disc=disc, prompt=prompt)
        emitted = response_text(raw).strip()
        emitted_label = emitted[:1] if emitted[:1] in {"0", "1"} else None
        argmax_label = "1" if float(parsed["logp_1"]) > float(parsed["logp_0"]) else "0"
        return {
            **candidate,
            "reward": float(parsed["logp_1"]),
            "p_own": float(parsed["p1"]),
            "logp_0": float(parsed["logp_0"]),
            "logp_1": float(parsed["logp_1"]),
            "binary_label_mass": float(parsed["binary_label_mass"]),
            "judge_token": emitted_label,
            "judge_argmax_token": argmax_label,
            "emitted_argmax_consistent": emitted_label == argmax_label,
            "raw_model": raw.get("model"),
            "raw_provider": raw.get("provider"),
            "judge_usage": raw.get("usage", {}),
            "judge_prompt_version": ATTRIBUTION_PROMPT_VERSION,
            "judge_prompt_template_sha256": template_hashes()[ATTRIBUTION_PROMPT_VERSION],
            "judge_user_prompt_sha256": sha256_text(prompt),
            "judge_message_roles": ["user"],
            "judge_system_prompt_present": False,
            "raw_judge_response": raw,
            "judge_latency_seconds": time.time() - started,
        }

    pending = [row for row in candidates if (row["pair_id"], row["pair_state"]) not in done]
    concurrency = int(transport.get("max_concurrency", 1))
    progress = tqdm(total=len(candidates), initial=len(done), desc="gsm8k shortcut judge")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(score_one, row) for row in pending]
        for future in as_completed(futures):
            existing.append(future.result())
            temporary = output_path.with_suffix(".jsonl.tmp")
            write_jsonl(temporary, existing)
            temporary.replace(output_path)
            progress.update(1)
    progress.close()


def analyze(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "shortcut_judgments.jsonl")
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], {})[row["pair_state"]] = row
    incomplete = [key for key, value in by_pair.items() if set(value) != {"original", "transformed"}]
    if incomplete:
        raise ValueError(f"Incomplete shortcut pairs: {incomplete[:5]}")
    effects: dict[str, Any] = {}
    for transform_name in TRANSFORMS:
        pairs = [value for key, value in by_pair.items() if key.startswith(transform_name + ":")]
        deltas = [float(pair["transformed"]["p_own"]) - float(pair["original"]["p_own"]) for pair in pairs]
        reward_deltas = [float(pair["transformed"]["reward"]) - float(pair["original"]["reward"]) for pair in pairs]
        effects[transform_name] = {
            "n_pairs": len(pairs),
            "mean_delta_p_own": statistics.mean(deltas),
            "median_delta_p_own": statistics.median(deltas),
            "mean_delta_reward": statistics.mean(reward_deltas),
            "argmax_flip_rate": statistics.mean(
                pair["transformed"]["judge_argmax_token"] != pair["original"]["judge_argmax_token"]
                for pair in pairs
            ),
            "sampled_flip_rate": statistics.mean(
                pair["transformed"]["judge_token"] != pair["original"]["judge_token"]
                for pair in pairs
            ),
        }
    maximum = max(abs(value["mean_delta_p_own"]) for value in effects.values())
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
    result = {
        "status": "complete",
        "n_judgments": len(rows),
        "n_pairs": len(by_pair),
        "correct_only": True,
        "semantic_correctness_inherited_from_original": True,
        "effects": effects,
        "maximum_absolute_mean_delta_p_own": maximum,
        "judge_usage": {
            key: sum(float((row.get("judge_usage") or {}).get(key, 0) or 0) for row in rows)
            for key in usage_keys
        },
        "protocol": {
            "all_emitted_argmax_consistent": all(row["emitted_argmax_consistent"] for row in rows),
            "minimum_binary_label_mass": min(float(row["binary_label_mass"]) for row in rows),
            "system_prompt_present": sorted({row["judge_system_prompt_present"] for row in rows}),
        },
    }
    write_json(run_dir / "shortcut_audit.json", result)
    return result


def judge_replicates(config: dict[str, Any], root: Path, run_dir: Path) -> None:
    candidates = read_jsonl(run_dir / "shortcut_candidates.jsonl")
    repetitions = int(config["shortcut_audit"]["judgments_per_state"])
    if repetitions < 2:
        raise ValueError("judgments_per_state must be at least two")
    output_path = run_dir / "shortcut_replicate_judgments.jsonl"
    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {
        (row["pair_id"], row["pair_state"], int(row["replicate_index"]))
        for row in existing
    }
    disc = config["discriminator"]
    transport = disc["transport"]
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    client = OpenRouterClient.from_env(
        max_attempts=int(transport["max_attempts"]),
        max_retry_wait_seconds=float(transport["max_retry_wait_seconds"]),
    )

    def score_one(item: tuple[dict[str, Any], int]) -> dict[str, Any]:
        candidate, replicate_index = item
        prompt = attribution_prompt(candidate["task_statement"], candidate["response"])
        started = time.time()
        raw, parsed = call_judge(client=client, disc=disc, prompt=prompt)
        emitted = response_text(raw).strip()
        emitted_label = emitted[:1] if emitted[:1] in {"0", "1"} else None
        argmax_label = "1" if float(parsed["logp_1"]) > float(parsed["logp_0"]) else "0"
        return {
            **candidate,
            "replicate_index": replicate_index,
            "reward": float(parsed["logp_1"]),
            "p_own": float(parsed["p1"]),
            "logp_0": float(parsed["logp_0"]),
            "logp_1": float(parsed["logp_1"]),
            "binary_label_mass": float(parsed["binary_label_mass"]),
            "judge_token": emitted_label,
            "judge_argmax_token": argmax_label,
            "emitted_argmax_consistent": emitted_label == argmax_label,
            "raw_model": raw.get("model"),
            "raw_provider": raw.get("provider"),
            "judge_usage": raw.get("usage", {}),
            "judge_prompt_version": ATTRIBUTION_PROMPT_VERSION,
            "judge_prompt_template_sha256": template_hashes()[ATTRIBUTION_PROMPT_VERSION],
            "judge_user_prompt_sha256": sha256_text(prompt),
            "judge_message_roles": ["user"],
            "judge_system_prompt_present": False,
            "raw_judge_response": raw,
            "judge_latency_seconds": time.time() - started,
        }

    pending = [
        (candidate, replicate_index)
        for candidate in candidates
        for replicate_index in range(1, repetitions)
        if (candidate["pair_id"], candidate["pair_state"], replicate_index) not in done
    ]
    concurrency = int(transport.get("max_concurrency", 1))
    progress = tqdm(
        total=len(candidates) * (repetitions - 1),
        initial=len(done),
        desc="gsm8k shortcut replicate judge",
    )
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(score_one, item) for item in pending]
        for future in as_completed(futures):
            existing.append(future.result())
            temporary = output_path.with_suffix(".jsonl.tmp")
            write_jsonl(temporary, existing)
            temporary.replace(output_path)
            progress.update(1)
    progress.close()


def analyze_replicates(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    repetitions = int(config["shortcut_audit"]["judgments_per_state"])
    primary = [{**row, "replicate_index": 0} for row in read_jsonl(run_dir / "shortcut_judgments.jsonl")]
    additional = read_jsonl(run_dir / "shortcut_replicate_judgments.jsonl")
    rows = primary + additional
    by_state: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_state.setdefault((row["pair_id"], row["pair_state"]), []).append(row)
    expected_states = int(config["shortcut_audit"]["pairs_per_transform"]) * len(TRANSFORMS) * 2
    if len(by_state) != expected_states or any(len(values) != repetitions for values in by_state.values()):
        raise ValueError("Replicated shortcut audit is incomplete")
    effects: dict[str, Any] = {}
    pair_summaries: list[dict[str, Any]] = []
    for transform_name in TRANSFORMS:
        pair_ids = sorted({pair_id for pair_id, _ in by_state if pair_id.startswith(transform_name + ":")})
        deltas: list[float] = []
        reward_deltas: list[float] = []
        for pair_id in pair_ids:
            original = by_state[(pair_id, "original")]
            transformed = by_state[(pair_id, "transformed")]
            original_mean = statistics.mean(float(row["p_own"]) for row in original)
            transformed_mean = statistics.mean(float(row["p_own"]) for row in transformed)
            original_reward = statistics.mean(float(row["reward"]) for row in original)
            transformed_reward = statistics.mean(float(row["reward"]) for row in transformed)
            delta = transformed_mean - original_mean
            reward_delta = transformed_reward - original_reward
            deltas.append(delta)
            reward_deltas.append(reward_delta)
            pair_summaries.append(
                {
                    "pair_id": pair_id,
                    "transform": transform_name,
                    "n_per_state": repetitions,
                    "mean_p_own_original": original_mean,
                    "mean_p_own_transformed": transformed_mean,
                    "delta_p_own": delta,
                    "delta_reward": reward_delta,
                }
            )
        standard_error = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
        effects[transform_name] = {
            "n_pairs": len(deltas),
            "judgments_per_state": repetitions,
            "mean_delta_p_own": statistics.mean(deltas),
            "median_delta_p_own": statistics.median(deltas),
            "standard_error_delta_p_own": standard_error,
            "two_sided_95_interval_delta_p_own": [
                statistics.mean(deltas) - 1.96 * standard_error,
                statistics.mean(deltas) + 1.96 * standard_error,
            ],
            "mean_delta_reward": statistics.mean(reward_deltas),
        }
    maximum = max(abs(value["mean_delta_p_own"]) for value in effects.values())
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
    result = {
        "status": "complete",
        "n_judgments": len(rows),
        "n_pairs": len(pair_summaries),
        "judgments_per_state": repetitions,
        "correct_only": True,
        "effects": effects,
        "pair_summaries": pair_summaries,
        "maximum_absolute_mean_delta_p_own": maximum,
        "judge_usage": {
            key: sum(float((row.get("judge_usage") or {}).get(key, 0) or 0) for row in rows)
            for key in usage_keys
        },
        "protocol": {
            "all_models": sorted({str(row["raw_model"]) for row in rows}),
            "all_providers": sorted({str(row["raw_provider"]) for row in rows}),
            "minimum_binary_label_mass": min(float(row["binary_label_mass"]) for row in rows),
            "system_prompt_present": sorted({row["judge_system_prompt_present"] for row in rows}),
            "emitted_argmax_consistency": statistics.mean(
                bool(row["emitted_argmax_consistent"]) for row in rows
            ),
        },
        "primary_single_judgment_audit_preserved": True,
    }
    write_json(run_dir / "shortcut_replicated_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "build",
            "judge",
            "analyze",
            "all",
            "replicate-judge",
            "replicate-analyze",
            "replicate-all",
        ),
        required=True,
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    run_dir = args.run_dir.resolve()
    if args.mode in {"build", "all"}:
        build_candidates(config, root, run_dir)
    if args.mode in {"judge", "all"}:
        judge_candidates(config, root, run_dir)
    if args.mode in {"analyze", "all"}:
        print(json.dumps(analyze(run_dir), indent=2, ensure_ascii=False))
    if args.mode in {"replicate-judge", "replicate-all"}:
        judge_replicates(config, root, run_dir)
    if args.mode in {"replicate-analyze", "replicate-all"}:
        print(json.dumps(analyze_replicates(config, run_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
