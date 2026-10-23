from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from iar_phase0.env import load_env_file
from iar_phase0.openrouter import OpenRouterClient, response_text

from .data import read_jsonl, write_json, write_jsonl
from .judge import call_judge, resolve


def checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def icc_one_way(values: list[list[float]]) -> float | None:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    n, k = matrix.shape
    subject_means = matrix.mean(axis=1)
    grand_mean = matrix.mean()
    ms_between = k * float(np.sum((subject_means - grand_mean) ** 2)) / (n - 1)
    ms_within = float(np.sum((matrix - subject_means[:, None]) ** 2)) / (n * (k - 1))
    denominator = ms_between + (k - 1) * ms_within
    return (ms_between - ms_within) / denominator if denominator else None


def pairwise_repeat_correlations(values: list[list[float]]) -> list[float]:
    matrix = np.asarray(values, dtype=float)
    correlations: list[float] = []
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            if len(set(matrix[:, left])) < 2 or len(set(matrix[:, right])) < 2:
                continue
            value = float(np.corrcoef(matrix[:, left], matrix[:, right])[0, 1])
            if math.isfinite(value):
                correlations.append(value)
    return correlations


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure pointwise judge test-retest reliability")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=4)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    run_dir = args.run_dir.resolve()
    source = read_jsonl(run_dir / "probe_judgments.jsonl")
    ordered = sorted(source, key=lambda row: (str(row["id"]), int(row["rollout_index"])))
    random.Random(int(config["project"]["seed"]) + 910_003).shuffle(ordered)
    selected = ordered[: args.candidates]
    selected_keys = [(row["id"], int(row["rollout_index"])) for row in selected]
    disc = config["discriminator"]
    transport = disc["transport"]
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    client = OpenRouterClient.from_env(
        max_attempts=int(transport["max_attempts"]),
        max_retry_wait_seconds=float(transport["max_retry_wait_seconds"]),
    )
    output_path = run_dir / "judge_repeatability_calls.jsonl"
    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {(row["id"], int(row["rollout_index"]), int(row["repetition"])) for row in existing}
    pending = [
        (row, repetition)
        for row in selected
        for repetition in range(args.repetitions)
        if (row["id"], int(row["rollout_index"]), repetition) not in done
    ]

    def judge_one(item: tuple[dict[str, Any], int]) -> dict[str, Any]:
        row, repetition = item
        started = time.time()
        raw, parsed = call_judge(client=client, disc=disc, prompt=str(row["judge_prompt"]))
        emitted = response_text(raw).strip()
        argmax = "1" if float(parsed["logp_1"]) > float(parsed["logp_0"]) else "0"
        return {
            "id": row["id"],
            "rollout_index": int(row["rollout_index"]),
            "repetition": repetition,
            "p_own": float(parsed["p1"]),
            "reward": float(parsed["logp_1"]),
            "logp_0": float(parsed["logp_0"]),
            "logp_1": float(parsed["logp_1"]),
            "binary_label_mass": float(parsed["binary_label_mass"]),
            "emitted": emitted,
            "argmax_01": argmax,
            "emitted_argmax_consistent": emitted in {"0", "1"} and emitted == argmax,
            "raw_model": raw.get("model"),
            "raw_provider": raw.get("provider"),
            "usage": raw.get("usage", {}),
            "raw_response": raw,
            "latency_seconds": time.time() - started,
        }

    with ThreadPoolExecutor(max_workers=int(transport["max_concurrency"])) as executor:
        futures = [executor.submit(judge_one, item) for item in pending]
        for future in as_completed(futures):
            existing.append(future.result())
            checkpoint(output_path, existing)

    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in existing:
        by_key.setdefault((row["id"], int(row["rollout_index"])), []).append(row)
    if set(by_key) != set(selected_keys) or any(len(rows) != args.repetitions for rows in by_key.values()):
        raise RuntimeError("Repeatability audit is incomplete")
    groups = [sorted(by_key[key], key=lambda row: int(row["repetition"])) for key in selected_keys]
    pown_matrix = [[float(row["p_own"]) for row in rows] for rows in groups]
    reward_matrix = [[float(row["reward"]) for row in rows] for rows in groups]
    correlations = pairwise_repeat_correlations(pown_matrix)
    within_pown_std = [statistics.pstdev(values) for values in pown_matrix]
    within_reward_std = [statistics.pstdev(values) for values in reward_matrix]
    flat = [row for rows in groups for row in rows]
    result = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_candidates": len(groups),
        "repetitions": args.repetitions,
        "logical_calls": len(flat),
        "selection_seed": int(config["project"]["seed"]) + 910_003,
        "selected_keys": [{"id": key[0], "rollout_index": key[1]} for key in selected_keys],
        "p_own_reliability": {
            "icc_1_1": icc_one_way(pown_matrix),
            "mean_pairwise_repeat_correlation": statistics.mean(correlations) if correlations else None,
            "minimum_pairwise_repeat_correlation": min(correlations) if correlations else None,
            "mean_within_candidate_std": statistics.mean(within_pown_std),
            "median_within_candidate_std": statistics.median(within_pown_std),
            "between_candidate_std_of_means": statistics.pstdev(
                statistics.mean(values) for values in pown_matrix
            ),
        },
        "reward_reliability": {
            "icc_1_1": icc_one_way(reward_matrix),
            "mean_within_candidate_std": statistics.mean(within_reward_std),
            "median_within_candidate_std": statistics.median(within_reward_std),
            "between_candidate_std_of_means": statistics.pstdev(
                statistics.mean(values) for values in reward_matrix
            ),
        },
        "decision_behavior": {
            "all_four_argmax_agree_rate": statistics.mean(
                len({row["argmax_01"] for row in rows}) == 1 for rows in groups
            ),
            "emitted_argmax_consistency": statistics.mean(
                bool(row["emitted_argmax_consistent"]) for row in flat
            ),
            "mean_emitted_own": statistics.mean(row["emitted"] == "1" for row in flat),
            "mean_p_own": statistics.mean(float(row["p_own"]) for row in flat),
        },
        "protocol": {
            "models": sorted({str(row["raw_model"]) for row in flat}),
            "providers": sorted({str(row["raw_provider"]) for row in flat}),
            "minimum_binary_label_mass": min(float(row["binary_label_mass"]) for row in flat),
            "system_prompt_present": False,
        },
        "usage": {
            key: sum(float((row.get("usage") or {}).get(key, 0) or 0) for row in flat)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
        },
    }
    write_json(run_dir / "judge_repeatability_audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
