from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from iar_phase0.env import load_env_file
from iar_phase0.openrouter import (
    OpenRouterClient,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)

from .data import read_jsonl, write_json, write_jsonl
from .prompts import (
    ATTRIBUTION_PROMPT_VERSION,
    attribution_prompt,
    protocol_manifest,
    sha256_text,
    task_prompt,
    template_hashes,
)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(tmp, rows)
    tmp.replace(path)


def candidate_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["id"]), str(row["source"]), str(row["variant"]), int(row["rollout_index"]))


def build_candidates(run_dir: Path, mode: str, seed: int) -> list[dict[str, Any]]:
    if mode == "baseline":
        actor = read_jsonl(run_dir / "actor_audit_responses.jsonl")
        own = read_jsonl(run_dir / "b_own_responses.jsonl")
        rows = own + actor
        expected = {"B_own": 100, "A0_plain": 100, "A0_imitation": 100}
        observed = {
            "B_own": sum(row["source"] == "B_own" for row in rows),
            "A0_plain": sum(row["source"] == "A0" and row["variant"] == "plain" for row in rows),
            "A0_imitation": sum(row["source"] == "A0" and row["variant"] == "imitation" for row in rows),
        }
        if observed != expected:
            raise ValueError(f"Incomplete baseline candidates: {observed}, expected {expected}")
    elif mode == "probe":
        rows = read_jsonl(run_dir / "actor_probe_k4_responses.jsonl")
        if len(rows) != 400:
            raise ValueError(f"Expected 400 K=4 candidates, received {len(rows)}")
    else:
        raise ValueError(mode)
    rows = list(rows)
    random.Random(seed + (0 if mode == "baseline" else 100_003)).shuffle(rows)
    return rows


def call_judge(
    *,
    client: OpenRouterClient,
    disc: dict[str, Any],
    prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    judgment = disc["judgment"]
    invalid_attempts = 4
    last_raw: dict[str, Any] | None = None
    last_parsed: dict[str, Any] | None = None
    for _ in range(invalid_attempts):
        payload = make_chat_payload(
            model_id=str(disc["model_id"]),
            prompt=prompt,
            system_prompt=None,
            max_tokens=int(judgment["max_tokens"]),
            temperature=float(judgment["temperature"]),
            logprobs=bool(judgment["logprobs"]),
            top_logprobs=int(judgment["top_logprobs"]),
            provider_routing=dict(disc["provider_routing"]),
            request_overrides=dict(disc.get("request_overrides", {})),
        )
        last_raw = client.chat_completion(payload)
        last_parsed = binary_probability_from_logprobs(last_raw)
        if last_parsed.get("ok"):
            return last_raw, last_parsed
        time.sleep(2)
    raise RuntimeError(f"Endpoint did not return valid 0/1 log-probs: {last_parsed}, raw={last_raw}")


def judge_candidates(
    *,
    config: dict[str, Any],
    root: Path,
    run_dir: Path,
    mode: str,
) -> None:
    disc = config["discriminator"]
    transport = disc["transport"]
    seed = int(config["project"]["seed"])
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    client = OpenRouterClient.from_env(
        max_attempts=int(transport["max_attempts"]),
        max_retry_wait_seconds=float(transport["max_retry_wait_seconds"]),
    )
    split_dir = resolve(root, config["paths"]["split_dir"])
    examples = {row["id"]: row for row in read_jsonl(split_dir / "reward_audit.jsonl")}
    candidates = build_candidates(run_dir, mode, seed)
    output_path = run_dir / f"{mode}_judgments.jsonl"
    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {candidate_key(row) for row in existing}
    pending: list[tuple[int, dict[str, Any], str]] = []
    for order, candidate in enumerate(candidates):
        key = candidate_key(candidate)
        if key in done:
            continue
        example = examples[candidate["id"]]
        task = task_prompt(example["question"], example["choices"])
        prompt = attribution_prompt(task, candidate["response"])
        pending.append((order, candidate, prompt))

    def score_one(item: tuple[int, dict[str, Any], str]) -> dict[str, Any]:
        order, candidate, prompt = item
        started = time.time()
        raw, parsed = call_judge(client=client, disc=disc, prompt=prompt)
        emitted = response_text(raw).strip()
        emitted_label = emitted[:1] if emitted[:1] in {"0", "1"} else None
        argmax_label = "1" if float(parsed["logp_1"]) > float(parsed["logp_0"]) else "0"
        row = dict(candidate)
        row.update(
            {
                "judgment_order": order,
                "ground_truth_own": candidate["source"] == "B_own",
                "reward": float(parsed["logp_1"]),
                "reward_value": "absolute_log_probability_of_label_1",
                "p_own": float(parsed["p1"]),
                "logp_0": float(parsed["logp_0"]),
                "logp_1": float(parsed["logp_1"]),
                "binary_label_mass": float(parsed["binary_label_mass"]),
                "judge_token": emitted_label,
                "judge_argmax_token": argmax_label,
                "emitted_argmax_consistent": emitted_label == argmax_label,
                "raw_model": raw.get("model"),
                "raw_provider": raw.get("provider"),
                "judge_prompt_version": ATTRIBUTION_PROMPT_VERSION,
                "judge_prompt_template_sha256": template_hashes()[ATTRIBUTION_PROMPT_VERSION],
                "judge_user_prompt_sha256": sha256_text(prompt),
                "judge_message_roles": ["user"],
                "judge_system_prompt_present": False,
                "judge_usage": raw.get("usage", {}),
                "judge_latency_seconds": time.time() - started,
            }
        )
        return row

    max_concurrency = int(transport.get("max_concurrency", 1))
    if max_concurrency <= 0:
        raise ValueError("discriminator.transport.max_concurrency must be positive")
    progress = tqdm(total=len(candidates), initial=len(done), desc=f"cqa {mode} judge")
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [executor.submit(score_one, item) for item in pending]
        for future in as_completed(futures):
            row = future.result()
            existing.append(row)
            done.add(candidate_key(row))
            checkpoint(output_path, existing)
            progress.update(1)
            time.sleep(float(transport["inter_request_delay_seconds"]))
    progress.close()


def smoke(config: dict[str, Any], root: Path, run_dir: Path) -> None:
    disc = config["discriminator"]
    transport = disc["transport"]
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    client = OpenRouterClient.from_env(
        max_attempts=int(transport["max_attempts"]),
        max_retry_wait_seconds=float(transport["max_retry_wait_seconds"]),
    )
    split_dir = resolve(root, config["paths"]["split_dir"])
    example = read_jsonl(split_dir / "reward_audit.jsonl")[0]
    task = task_prompt(example["question"], example["choices"])
    candidate = "A short placeholder response used only to verify the endpoint.\nAnswer: A"
    prompt = attribution_prompt(task, candidate)
    started = time.time()
    raw, parsed = call_judge(client=client, disc=disc, prompt=prompt)
    result = {
        "ok": bool(parsed.get("ok")),
        "model_requested": disc["model_id"],
        "raw_model": raw.get("model"),
        "raw_provider": raw.get("provider"),
        "prompt_version": ATTRIBUTION_PROMPT_VERSION,
        "prompt_template_sha256": template_hashes()[ATTRIBUTION_PROMPT_VERSION],
        "message_roles": ["user"],
        "system_prompt_present": False,
        "parsed": parsed,
        "emitted_text": response_text(raw),
        "usage": raw.get("usage", {}),
        "latency_seconds": time.time() - started,
    }
    write_json(run_dir / "endpoint_smoke.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "baseline", "probe"), required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    root = Path(config["project"]["root"]).resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "protocol_manifest.json", protocol_manifest())
    if args.mode == "smoke":
        smoke(config, root, run_dir)
    else:
        judge_candidates(config=config, root=root, run_dir=run_dir, mode=args.mode)


if __name__ == "__main__":
    main()
