from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import ensure_dirs, load_config
from .data import load_mbpp, materialize_split, read_jsonl, write_jsonl
from .discriminator import Discriminator
from .env import load_env_file
from .openrouter import OpenRouterClient
from .openrouter import (
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)
from .parameters import initialize_parameter_record
from .prompts import (
    pointwise_self_attribution_prompt,
    pointwise_self_attribution_system_prompt,
)
from .run_pointwise_self_attribution import parse_binary_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    run_cfg = config.raw["reward_prompt_rejudge"]
    run_id = datetime.now(timezone.utc).strftime(f"{run_cfg['run_name']}_%Y%m%dT%H%M%SZ")
    out_dir = config.resolve(config.raw["paths"]["experiments"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    initialize_parameter_record(out_dir, resolved_config=config.raw, source_config=config.path)

    attribution_dir = config.resolve(run_cfg["source_attribution_run"])
    attribution_metadata = json.loads((attribution_dir / "summary.json").read_text())
    pointwise = {
        int(row["trial_id"]): row
        for row in read_jsonl(attribution_dir / "judgments.jsonl")
    }
    plan = read_jsonl(attribution_dir / "trial_plan.jsonl")
    source_dirs = {
        "B_own": Path(attribution_metadata["source_runs"]["b_own"]),
        "plain": Path(attribution_metadata["source_runs"]["plain"]),
        "imitation": Path(attribution_metadata["source_runs"]["imitation"]),
    }
    responses = {
        source: {
            int(row["index"]): row
            for row in read_jsonl(source_dir / "responses.jsonl")
        }
        for source, source_dir in source_dirs.items()
    }
    examples = materialize_split(load_mbpp(config), split="test", limit=100)

    env_path = config.resolve(config.raw["paths"]["openrouter_env"])
    load_env_file(env_path)
    disc_cfg = config.raw["discriminator"]
    discriminator = Discriminator(
        client=OpenRouterClient.from_env(),
        model_id=disc_cfg["model_id"],
        own_answer_max_tokens=int(disc_cfg["own_answer_max_tokens"]),
        judge_max_tokens=int(disc_cfg["judge_max_tokens"]),
        temperature=float(disc_cfg["temperature"]),
        top_logprobs=int(disc_cfg["top_logprobs"]),
        provider_routing=disc_cfg["provider_routing"],
        request_overrides=disc_cfg.get("request_overrides"),
        strict_logprobs=bool(disc_cfg["strict_logprobs"]),
    )
    pointwise_system_prompt = pointwise_self_attribution_system_prompt(
        str(run_cfg["target_model_name"])
    )

    candidates: list[dict[str, Any]] = []
    for trial in plan:
        trial_id = int(trial["trial_id"])
        index = int(trial["index"])
        source = str(trial["candidate_source_key"])
        candidates.append({
            **trial,
            "candidate_text": str(responses[source][index]["text"]),
            "pointwise_p_own": float(pointwise[trial_id]["judge"]["parsed"]["p1"]),
            "pointwise_prediction": pointwise[trial_id]["prediction"],
        })

    results: list[dict[str, Any]] = []
    workers = int(run_cfg["workers"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                judge,
                candidate=candidate,
                example=examples[int(candidate["index"])],
                discriminator=discriminator,
                pointwise_system_prompt=pointwise_system_prompt,
                run_pointwise_control=bool(run_cfg.get("same_provider_pointwise_control", False)),
            )
            for candidate in candidates
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"reward_prompt_rejudge ({workers} workers)",
        ):
            results.append(future.result())
            write_jsonl(out_dir / "judgments.jsonl", results)

    summary = summarize(results)
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.raw,
        "source_attribution_run": str(attribution_dir),
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(out_dir), "summary": summary}, indent=2))


def judge(
    *,
    candidate: dict[str, Any],
    example: Any,
    discriminator: Discriminator,
    pointwise_system_prompt: str,
    run_pointwise_control: bool,
) -> dict[str, Any]:
    result = discriminator.judge(example, str(candidate["candidate_text"]))
    prediction = parse_binary_prediction(str(result["text"] or ""), result["parsed"])
    row = {
        "trial_id": candidate["trial_id"],
        "index": candidate["index"],
        "task_id": candidate["task_id"],
        "candidate_source_key": candidate["candidate_source_key"],
        "ground_truth": candidate["ground_truth"],
        "candidate_unit_tests": candidate["candidate_unit_tests"],
        "pointwise_p_own": candidate["pointwise_p_own"],
        "pointwise_prediction": candidate["pointwise_prediction"],
        "reward_prompt_prediction": prediction,
        "judge": result,
    }
    if run_pointwise_control:
        payload = make_chat_payload(
            model_id=discriminator.model_id,
            system_prompt=pointwise_system_prompt,
            prompt=pointwise_self_attribution_prompt(
                example, str(candidate["candidate_text"])
            ),
            max_tokens=discriminator.judge_max_tokens,
            temperature=discriminator.temperature,
            logprobs=True,
            top_logprobs=discriminator.top_logprobs,
            provider_routing=discriminator.provider_routing,
            request_overrides=discriminator.request_overrides,
        )
        response = discriminator.client.chat_completion(payload)
        parsed = binary_probability_from_logprobs(response)
        if discriminator.strict_logprobs and not parsed["ok"]:
            raise RuntimeError(f"Could not extract binary log-probabilities: {parsed}")
        text = response_text(response)
        row["same_provider_pointwise_prediction"] = parse_binary_prediction(
            str(text or ""), parsed
        )
        row["same_provider_pointwise_judge"] = {
            "text": text,
            "parsed": parsed,
            "raw_model": response.get("model"),
            "usage": response.get("usage"),
        }
    return row


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += float(pos > neg) + 0.5 * float(pos == neg)
    return wins / (len(positive) * len(negative))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def p1(row: dict[str, Any]) -> float:
        return float(row["judge"]["parsed"]["p1"])

    def argmax(row: dict[str, Any]) -> int:
        parsed = row["judge"]["parsed"]
        return int(float(parsed["logp_1"]) > float(parsed["logp_0"]))

    groups = {
        "own": [row for row in rows if int(row["ground_truth"]) == 1],
        "non_own": [row for row in rows if int(row["ground_truth"]) == 0],
    }
    by_group: dict[str, Any] = {}
    for name, group in groups.items():
        values = [p1(row) for row in group]
        by_group[name] = {
            "n": len(group),
            "mean_p_own": mean(values),
            "median_p_own": median(values),
            "acceptance_emitted": mean([
                float(row["reward_prompt_prediction"] == 1) for row in group
            ]),
            "acceptance_argmax": mean([float(argmax(row) == 1) for row in group]),
            "mean_delta_p_own_vs_pointwise": mean([
                p1(row) - float(row["pointwise_p_own"]) for row in group
            ]),
            "label_changed_vs_pointwise": sum(
                row["reward_prompt_prediction"] != row["pointwise_prediction"]
                for row in group
            ),
        }

    own_scores = [p1(row) for row in groups["own"]]
    non_own_scores = [p1(row) for row in groups["non_own"]]
    result = {
        "n": len(rows),
        "by_group": by_group,
        "gap_S_B_minus_S_A": (mean(own_scores) or 0.0) - (mean(non_own_scores) or 0.0),
        "auc": auc(own_scores, non_own_scores),
        "emitted_argmax_consistency": mean([
            float(row["reward_prompt_prediction"] == argmax(row)) for row in rows
        ]),
        "binary_label_mass_min": min(
            float(row["judge"]["parsed"]["binary_label_mass"]) for row in rows
        ),
        "total_api_cost": sum(
            float((row["judge"].get("usage") or {}).get("cost") or 0.0)
            for row in rows
        ),
    }
    if rows and rows[0].get("same_provider_pointwise_judge"):
        def pointwise_p1(row: dict[str, Any]) -> float:
            return float(row["same_provider_pointwise_judge"]["parsed"]["p1"])

        pointwise_by_group: dict[str, Any] = {}
        for name, group in groups.items():
            values = [pointwise_p1(row) for row in group]
            pointwise_by_group[name] = {
                "n": len(group),
                "mean_p_own": mean(values),
                "median_p_own": median(values),
                "acceptance_emitted": mean([
                    float(row["same_provider_pointwise_prediction"] == 1)
                    for row in group
                ]),
                "mean_delta_reward_minus_pointwise": mean([
                    p1(row) - pointwise_p1(row) for row in group
                ]),
                "label_changed_by_reward_prompt": sum(
                    row["reward_prompt_prediction"]
                    != row["same_provider_pointwise_prediction"]
                    for row in group
                ),
            }
        own_pointwise = [pointwise_p1(row) for row in groups["own"]]
        non_own_pointwise = [pointwise_p1(row) for row in groups["non_own"]]
        result["same_provider_pointwise_control"] = {
            "by_group": pointwise_by_group,
            "gap_S_B_minus_S_A": (mean(own_pointwise) or 0.0)
            - (mean(non_own_pointwise) or 0.0),
            "auc": auc(own_pointwise, non_own_pointwise),
        }
        result["total_api_cost"] += sum(
            float((row["same_provider_pointwise_judge"].get("usage") or {}).get("cost") or 0.0)
            for row in rows
        )
    return result


if __name__ == "__main__":
    main()
