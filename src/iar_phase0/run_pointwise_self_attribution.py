from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import ensure_dirs, load_config
from .data import load_mbpp, materialize_split, read_jsonl, write_jsonl
from .env import load_env_file
from .openrouter import (
    OpenRouterClient,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)
from .parameters import initialize_parameter_record, write_parameter_artifact
from .prompts import (
    POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION,
    TARGET_MODEL_NAME,
    mbpp_google_problem_statement,
    pointwise_self_attribution_prompt_for_version,
    pointwise_self_attribution_system_prompt_for_version,
    pointwise_self_attribution_template_sha256_for_version,
    sha256_text,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)

    run_cfg = config.raw["pointwise_self_attribution"]
    run_prefix = run_cfg.get("run_name", "pointwise_self_attr_test100")
    if args.resume_dir:
        out_dir = config.resolve(args.resume_dir)
        if not out_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {out_dir}")
        run_id = out_dir.name
        resume_index = len(list((out_dir / "parameters").glob("resume_parameters_*.json"))) + 1
        write_parameter_artifact(
            out_dir,
            f"resume_parameters_{resume_index:03d}.json",
            {
                "resumed_at": datetime.now(timezone.utc).isoformat(),
                "source_config": str(config.path),
                "resolved_config": config.raw,
            },
        )
    else:
        run_id = datetime.now(timezone.utc).strftime(f"{run_prefix}_%Y%m%dT%H%M%SZ")
        out_dir = config.resolve(config.raw["paths"]["experiments"]) / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        initialize_parameter_record(
            out_dir,
            resolved_config=config.raw,
            source_config=config.path,
        )

    dataset = load_mbpp(config)
    examples = materialize_split(
        dataset,
        split=run_cfg.get("split", "test"),
        limit=int(run_cfg.get("task_count", 100)),
    )

    experiments_dir = config.resolve(config.raw["paths"]["experiments"])
    b_own_dir = latest_run(experiments_dir, run_cfg["source_runs"]["b_own_glob"])
    plain_dir = latest_run(experiments_dir, run_cfg["source_runs"]["plain_glob"])
    imitation_dir = latest_run(experiments_dir, run_cfg["source_runs"]["imitation_glob"])

    responses = {
        "B_own": load_responses_by_index(b_own_dir / "responses.jsonl"),
        "plain": load_responses_by_index(plain_dir / "responses.jsonl"),
        "imitation": load_responses_by_index(imitation_dir / "responses.jsonl"),
    }
    unit_tests = {
        "B_own": load_unit_tests_by_index(b_own_dir / "unit_tests.jsonl"),
        "plain": load_unit_tests_by_index(plain_dir / "unit_tests.jsonl"),
        "imitation": load_unit_tests_by_index(imitation_dir / "unit_tests.jsonl"),
    }

    seed = int(config.raw["project"].get("seed", 17))
    rng = random.Random(seed)
    sampling_mode = str(run_cfg.get("sampling_mode", "disjoint"))
    if sampling_mode == "paired_all":
        trial_plan = build_paired_trial_plan(
            examples=examples,
            responses=responses,
            unit_tests=unit_tests,
            non_own_source=str(run_cfg.get("paired_non_own_source", "plain")),
            rng=rng,
        )
    elif sampling_mode == "disjoint":
        trial_plan = build_trial_plan(
            examples=examples,
            responses=responses,
            unit_tests=unit_tests,
            own_count=int(run_cfg["own_count"]),
            plain_count=int(run_cfg["non_own_mix"]["plain"]),
            imitation_count=int(run_cfg["non_own_mix"]["imitation"]),
            rng=rng,
        )
    else:
        raise ValueError(
            "pointwise_self_attribution.sampling_mode must be 'disjoint' or 'paired_all'"
        )

    public_plan = [public_trial_row(row) for row in trial_plan]
    trial_plan_path = out_dir / "trial_plan.jsonl"
    if args.resume_dir:
        recorded_plan = read_jsonl(trial_plan_path)
        if recorded_plan != public_plan:
            raise RuntimeError("Refusing resume: regenerated trial plan differs from recorded plan")
    else:
        write_jsonl(trial_plan_path, public_plan)

    env_path = config.resolve(config.raw["paths"]["openrouter_env"])
    load_env_file(env_path)
    disc_cfg = config.raw["discriminator"]
    client = OpenRouterClient.from_env(
        max_attempts=int(run_cfg.get("max_attempts_per_call", 4)),
        max_retry_wait_seconds=float(run_cfg.get("max_retry_wait_seconds", 60)),
    )
    target_model = run_cfg.get("target_model_name", TARGET_MODEL_NAME)
    prompt_version = str(
        run_cfg.get("prompt_version", POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION)
    )
    system_prompt = pointwise_self_attribution_system_prompt_for_version(
        prompt_version, target_model
    )
    prompt_template_sha256 = pointwise_self_attribution_template_sha256_for_version(
        prompt_version
    )
    protocol_manifest = {
        "schema_version": 1,
        "prompt_version": prompt_version,
        "prompt_template_sha256": prompt_template_sha256,
        "system_prompt_present": system_prompt is not None,
        "system_prompt_sha256": (
            sha256_text(system_prompt) if system_prompt is not None else None
        ),
        "message_roles": ["user"] if system_prompt is None else ["system", "user"],
        "model_id": str(disc_cfg["model_id"]),
        "provider_routing": disc_cfg.get("provider_routing"),
        "api_parameters": {
            "temperature": disc_cfg.get("temperature"),
            "max_tokens": disc_cfg.get("judge_max_tokens"),
            "logprobs": True,
            "top_logprobs": disc_cfg.get("top_logprobs"),
            "request_overrides": disc_cfg.get("request_overrides"),
        },
        "dataset_prompt_style": config.raw["dataset"].get("prompt_style"),
        "sampling_mode": sampling_mode,
        "seed": seed,
        "n_trials": len(trial_plan),
        "source_runs": {
            "b_own": str(b_own_dir),
            "plain": str(plain_dir),
            "imitation": str(imitation_dir),
        },
    }
    protocol_manifest_path = out_dir / "protocol_manifest.json"
    if protocol_manifest_path.exists():
        recorded_manifest = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
        if recorded_manifest != protocol_manifest:
            raise RuntimeError("Refusing resume: protocol manifest differs")
    else:
        protocol_manifest_path.write_text(
            json.dumps(protocol_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    judgments_path = out_dir / "judgments.jsonl"
    judgments: list[dict[str, Any]] = (
        read_jsonl(judgments_path) if args.resume_dir and judgments_path.exists() else []
    )
    completed_trial_ids = [int(row["trial_id"]) for row in judgments]
    if len(completed_trial_ids) != len(set(completed_trial_ids)):
        raise RuntimeError("Refusing resume: duplicate completed trial IDs")
    if any(
        row["judge"].get("prompt_version") != prompt_version
        or row["judge"].get("prompt_template_sha256") != prompt_template_sha256
        or bool(row["judge"].get("system_prompt_present")) != (system_prompt is not None)
        for row in judgments
    ):
        raise RuntimeError("Refusing resume: completed judgments use a different prompt protocol")
    pending_trials = [
        trial for trial in trial_plan if int(trial["trial_id"]) not in set(completed_trial_ids)
    ]
    workers = int(run_cfg.get("workers", 1))
    if workers < 1:
        raise ValueError("pointwise_self_attribution.workers must be at least 1")
    inter_request_delay = float(run_cfg.get("inter_request_delay_seconds", 0.0))
    if inter_request_delay < 0:
        raise ValueError("inter_request_delay_seconds cannot be negative")
    if workers > 1 and inter_request_delay:
        raise ValueError("inter_request_delay_seconds requires workers=1")

    if workers == 1:
        completed = (
            judge_trial(
                trial=trial,
                example=examples[int(trial["index"])],
                client=client,
                disc_cfg=disc_cfg,
                system_prompt=system_prompt,
                prompt_version=prompt_version,
                prompt_template_sha256=prompt_template_sha256,
                invalid_response_attempts=int(
                    run_cfg.get("invalid_response_attempts", 1)
                ),
                invalid_response_retry_delay_seconds=float(
                    run_cfg.get("invalid_response_retry_delay_seconds", 5)
                ),
            )
            for trial in pending_trials
        )
        iterator = tqdm(
            completed,
            total=len(pending_trials),
            initial=0,
            desc=f"pointwise_self_attr:remaining_from_{len(judgments)}",
        )
        for row in iterator:
            judgments.append(row)
            judgments.sort(key=lambda item: int(item["trial_id"]))
            write_jsonl(judgments_path, judgments)
            if inter_request_delay:
                time.sleep(inter_request_delay)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    judge_trial,
                    trial=trial,
                    example=examples[int(trial["index"])],
                    client=client,
                    disc_cfg=disc_cfg,
                    system_prompt=system_prompt,
                    prompt_version=prompt_version,
                    prompt_template_sha256=prompt_template_sha256,
                    invalid_response_attempts=int(
                        run_cfg.get("invalid_response_attempts", 1)
                    ),
                    invalid_response_retry_delay_seconds=float(
                        run_cfg.get("invalid_response_retry_delay_seconds", 5)
                    ),
                )
                for trial in pending_trials
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"pointwise_self_attr:remaining_from_{len(judgments)} ({workers} workers)",
            ):
                judgments.append(future.result())
                judgments.sort(key=lambda item: int(item["trial_id"]))
                write_jsonl(judgments_path, judgments)

    if len(judgments) != len(trial_plan):
        raise RuntimeError(
            f"Incomplete attribution run: {len(judgments)}/{len(trial_plan)} judgments"
        )

    summary = summarize(judgments)
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.raw,
        "source_runs": {
            "b_own": str(b_own_dir),
            "plain": str(plain_dir),
            "imitation": str(imitation_dir),
        },
        "attribution_prompt": {
            "version": prompt_version,
            "template_sha256": prompt_template_sha256,
            "system_prompt_present": system_prompt is not None,
            "system_prompt_sha256": (
                sha256_text(system_prompt) if system_prompt is not None else None
            ),
            "message_roles": ["user"] if system_prompt is None else ["system", "user"],
        },
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(out_dir), "summary": summary}, indent=2, ensure_ascii=False))


def judge_trial(
    *,
    trial: dict[str, Any],
    example: Any,
    client: OpenRouterClient,
    disc_cfg: dict[str, Any],
    system_prompt: str | None,
    prompt_version: str,
    prompt_template_sha256: str,
    invalid_response_attempts: int,
    invalid_response_retry_delay_seconds: float,
) -> dict[str, Any]:
    prompt = pointwise_self_attribution_prompt_for_version(
        mbpp_google_problem_statement(example),
        str(trial["candidate_text"]),
        prompt_version,
    )
    payload = make_chat_payload(
        model_id=disc_cfg["model_id"],
        system_prompt=system_prompt,
        prompt=prompt,
        max_tokens=int(disc_cfg["judge_max_tokens"]),
        temperature=float(disc_cfg["temperature"]),
        logprobs=True,
        top_logprobs=int(disc_cfg["top_logprobs"]),
        provider_routing=disc_cfg["provider_routing"],
        request_overrides=disc_cfg.get("request_overrides"),
    )
    if invalid_response_attempts <= 0:
        raise ValueError("invalid_response_attempts must be positive")
    if invalid_response_retry_delay_seconds < 0:
        raise ValueError("invalid_response_retry_delay_seconds cannot be negative")
    started = time.time()
    response: dict[str, Any] | None = None
    parsed: dict[str, Any] = {"ok": False, "error": "not_attempted"}
    validation_attempt = 0
    for validation_attempt in range(1, invalid_response_attempts + 1):
        response = client.chat_completion(payload)
        parsed = binary_probability_from_logprobs(response)
        if parsed.get("ok") or not bool(disc_cfg.get("strict_logprobs", True)):
            break
        if validation_attempt < invalid_response_attempts:
            time.sleep(invalid_response_retry_delay_seconds)
    if response is None:
        raise AssertionError("OpenRouter response loop did not execute")
    if bool(disc_cfg.get("strict_logprobs", True)) and not parsed.get("ok"):
        raise RuntimeError(
            "Could not extract binary log-probabilities after "
            f"{validation_attempt} successful HTTP responses: {parsed}"
        )
    text = response_text(response)
    prediction = parse_binary_prediction(text, parsed)
    return {
        **public_trial_row(trial),
        "prediction": prediction,
        "correct": prediction == int(trial["ground_truth"]),
        "judge": {
            "text": text,
            "parsed": parsed,
            "raw_model": response.get("model"),
            "raw_provider": response.get("provider"),
            "prompt_version": prompt_version,
            "prompt_template_sha256": prompt_template_sha256,
            "system_prompt_present": system_prompt is not None,
            "system_prompt_sha256": (
                sha256_text(system_prompt) if system_prompt is not None else None
            ),
            "message_roles": [message["role"] for message in payload["messages"]],
            "user_prompt_sha256": sha256_text(prompt),
            "usage": response.get("usage"),
            "latency_seconds": time.time() - started,
            "validation_attempt": validation_attempt,
        },
    }


def latest_run(experiments_dir: Path, pattern: str) -> Path:
    runs = sorted(experiments_dir.glob(pattern))
    if not runs:
        raise FileNotFoundError(f"No experiment run found for pattern: {pattern}")
    return runs[-1]


def load_responses_by_index(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["index"]): row for row in read_jsonl(path)}


def load_unit_tests_by_index(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["index"]): row for row in read_jsonl(path)}


def build_trial_plan(
    *,
    examples: list[Any],
    responses: dict[str, dict[int, dict[str, Any]]],
    unit_tests: dict[str, dict[int, dict[str, Any]]],
    own_count: int,
    plain_count: int,
    imitation_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    total = own_count + plain_count + imitation_count
    if total > len(examples):
        raise ValueError(f"Requested {total} trials, but only {len(examples)} examples are available.")

    indices = list(range(len(examples)))
    rng.shuffle(indices)
    own_indices = indices[:own_count]
    non_own_indices = indices[own_count:total]
    rng.shuffle(non_own_indices)
    plain_indices = non_own_indices[:plain_count]
    imitation_indices = non_own_indices[plain_count:plain_count + imitation_count]

    trials: list[dict[str, Any]] = []
    for idx in own_indices:
        trials.append(make_trial(idx, examples[idx], responses, unit_tests, source_key="B_own", ground_truth=1))
    for idx in plain_indices:
        trials.append(make_trial(idx, examples[idx], responses, unit_tests, source_key="plain", ground_truth=0))
    for idx in imitation_indices:
        trials.append(make_trial(idx, examples[idx], responses, unit_tests, source_key="imitation", ground_truth=0))

    rng.shuffle(trials)
    for trial_id, row in enumerate(trials):
        row["trial_id"] = trial_id
    return trials


def build_paired_trial_plan(
    *,
    examples: list[Any],
    responses: dict[str, dict[int, dict[str, Any]]],
    unit_tests: dict[str, dict[int, dict[str, Any]]],
    non_own_source: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if non_own_source not in {"plain", "imitation"}:
        raise ValueError("paired_non_own_source must be 'plain' or 'imitation'")
    trials: list[dict[str, Any]] = []
    for idx, example in enumerate(examples):
        trials.append(
            make_trial(
                idx,
                example,
                responses,
                unit_tests,
                source_key="B_own",
                ground_truth=1,
            )
        )
        trials.append(
            make_trial(
                idx,
                example,
                responses,
                unit_tests,
                source_key=non_own_source,
                ground_truth=0,
            )
        )
    rng.shuffle(trials)
    for trial_id, row in enumerate(trials):
        row["trial_id"] = trial_id
    return trials


def make_trial(
    idx: int,
    example: Any,
    responses: dict[str, dict[int, dict[str, Any]]],
    unit_tests: dict[str, dict[int, dict[str, Any]]],
    *,
    source_key: str,
    ground_truth: int,
) -> dict[str, Any]:
    response = responses[source_key][idx]
    test_row = unit_tests[source_key].get(idx, {})
    return {
        "index": idx,
        "task_id": example.task_id,
        "split": example.split,
        "candidate_source": response["source"],
        "candidate_variant": response["variant"],
        "candidate_source_key": source_key,
        "candidate_text": response["text"],
        "candidate_unit_tests": test_row.get("unit_tests"),
        "ground_truth": ground_truth,
    }


def public_trial_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": row.get("trial_id"),
        "split": row["split"],
        "index": row["index"],
        "task_id": row["task_id"],
        "candidate_source": row["candidate_source"],
        "candidate_variant": row["candidate_variant"],
        "candidate_source_key": row["candidate_source_key"],
        "candidate_unit_tests": row.get("candidate_unit_tests"),
        "ground_truth": row["ground_truth"],
    }


def parse_binary_prediction(text: str, parsed: dict[str, Any]) -> int | None:
    stripped = text.strip()
    if stripped.startswith("1"):
        return 1
    if stripped.startswith("0"):
        return 0
    first_token = str(parsed.get("first_token", "")).strip()
    if first_token == "1":
        return 1
    if first_token == "0":
        return 0
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": mean([float(r["correct"]) for r in rows if r["prediction"] is not None]),
        "valid_prediction_rate": mean([float(r["prediction"] is not None) for r in rows]),
        "logprob_ok_rate": mean([float(r["judge"]["parsed"].get("ok")) for r in rows]),
        "overall_accept_rate": mean([float(r["prediction"] == 1) for r in rows if r["prediction"] is not None]),
        "mean_p_own": mean([
            float(r["judge"]["parsed"]["p1"])
            for r in rows
            if r["judge"]["parsed"].get("ok")
        ]),
        "by_ground_truth": summarize_group(rows, lambda r: f"ground_truth_{r['ground_truth']}"),
        "by_candidate_source": summarize_group(
            rows,
            lambda r: f"{r['candidate_source_key']}:{r['candidate_variant']}",
        ),
    }


def summarize_group(rows: list[dict[str, Any]], key_fn: Any) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(key_fn(row)), []).append(row)
    return {key: summarize_basic(group_rows) for key, group_rows in sorted(groups.items())}


def summarize_basic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": mean([float(r["correct"]) for r in rows if r["prediction"] is not None]),
        "accept_rate": mean([float(r["prediction"] == 1) for r in rows if r["prediction"] is not None]),
        "mean_p_own": mean([
            float(r["judge"]["parsed"]["p1"])
            for r in rows
            if r["judge"]["parsed"].get("ok")
        ]),
        "candidate_pass@1": mean([
            float(r["candidate_unit_tests"]["passed"])
            for r in rows
            if r.get("candidate_unit_tests")
        ]),
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
