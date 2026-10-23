from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from iar_phase0.env import load_env_file
from iar_phase0.generator import LocalGenerator
from iar_phase0.openrouter import OpenRouterClient, make_chat_payload, response_text

from .data import read_jsonl, write_json, write_jsonl
from .prompts import (
    IMITATION_PROMPT_VERSION,
    TASK_PROMPT_VERSION,
    imitation_prompt,
    protocol_manifest,
    sha256_text,
    task_prompt,
)
from .scoring import response_features, score_response


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def local_generate(generator: LocalGenerator, prompt: str, candidate_seed: int) -> tuple[str, dict[str, Any]]:
    random.seed(candidate_seed)
    torch.manual_seed(candidate_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(candidate_seed)
    messages = [{"role": "user", "content": prompt}]
    tokenizer = generator.tokenizer
    rendered = (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if getattr(tokenizer, "chat_template", None)
        else prompt
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(generator.model.device)
    do_sample = generator.temperature > 0
    with torch.no_grad():
        output = generator.model.generate(
            **inputs,
            max_new_tokens=generator.max_new_tokens,
            do_sample=do_sample,
            temperature=generator.temperature if do_sample else None,
            top_p=generator.top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = output[0][inputs["input_ids"].shape[1]:]
    eos_positions = (completion == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    terminated = len(eos_positions) > 0
    if terminated:
        completion = completion[: int(eos_positions[0]) + 1]
    text = tokenizer.decode(completion, skip_special_tokens=True).strip()
    return text, {
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "completion_tokens": int(len(completion)),
        "terminated_with_eos": bool(terminated),
        "hit_max_new_tokens": int(len(completion)) >= int(generator.max_new_tokens) and not terminated,
    }


def generate_actor(
    *,
    config: dict[str, Any],
    root: Path,
    source_rows: list[dict[str, Any]],
    output_path: Path,
    mode: str,
) -> None:
    actor = config["actor"]
    settings = actor["baseline"] if mode == "audit" else actor["probe"]
    seed = int(config["project"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    generator = LocalGenerator(
        model_id=str(actor["model_id"]),
        cache_dir=resolve(root, config["paths"]["hf_cache"]),
        max_new_tokens=int(settings["max_new_tokens"]),
        temperature=float(settings["temperature"]),
        top_p=float(settings["top_p"]),
        local_files_only=bool(actor.get("local_files_only", True)),
        dtype=str(actor.get("dtype", "auto")),
        device_map=str(actor.get("device_map", "auto")),
    )
    existing = read_jsonl(output_path) if output_path.exists() else []
    keys = {(row["id"], row["variant"], int(row["rollout_index"])) for row in existing}
    variants = ("plain", "imitation") if mode == "audit" else ("plain",)
    generations = 1 if mode == "audit" else int(settings["num_generations"])
    total = len(source_rows) * len(variants) * generations
    progress = tqdm(total=total, initial=len(keys), desc=f"gsm8k actor {mode}")
    for item in source_rows:
        for variant in variants:
            prompt = task_prompt(item["question"]) if variant == "plain" else imitation_prompt(item["question"])
            prompt_version = TASK_PROMPT_VERSION if variant == "plain" else IMITATION_PROMPT_VERSION
            for rollout_index in range(generations):
                key = (item["id"], variant, rollout_index)
                if key in keys:
                    continue
                started = time.time()
                candidate_seed = seed + int(
                    sha256_text(f"{item['id']}|{variant}|{rollout_index}")[:8], 16
                ) % 2_000_000_000
                text, token_data = local_generate(generator, prompt, candidate_seed)
                existing.append(
                    {
                        "id": item["id"],
                        "source_index": item["source_index"],
                        "experiment_order": item["experiment_order"],
                        "variant": variant,
                        "rollout_index": rollout_index,
                        "source": "A0",
                        "model_id": actor["model_id"],
                        "message_roles": ["user"],
                        "system_prompt_present": False,
                        "prompt": prompt,
                        "prompt_version": prompt_version,
                        "prompt_sha256": sha256_text(prompt),
                        "response": text,
                        "score": score_response(text, item["answer_value"]),
                        "features": response_features(text, item["question"]),
                        "generation": {
                            "temperature": float(settings["temperature"]),
                            "top_p": float(settings["top_p"]),
                            "max_new_tokens": int(settings["max_new_tokens"]),
                            "base_seed": seed,
                            "candidate_seed": candidate_seed,
                        },
                        **token_data,
                        "latency_seconds": time.time() - started,
                    }
                )
                keys.add(key)
                checkpoint(output_path, existing)
                progress.update(1)
    progress.close()


def resolved_probe_path(run_dir: Path) -> Path:
    resolved = run_dir / "actor_probe_k4_responses_resolved.jsonl"
    return resolved if resolved.exists() else run_dir / "actor_probe_k4_responses.jsonl"


def reaudit_probe_length(*, config: dict[str, Any], root: Path, run_dir: Path) -> None:
    """Extend only capped K=4 rollouts, preserving seeds and original artifacts."""
    actor = config["actor"]
    settings = actor["probe"]
    original_path = run_dir / "actor_probe_k4_responses.jsonl"
    original = read_jsonl(original_path)
    if len(original) != 400:
        raise ValueError(f"Length re-audit requires 400 original rollouts, received {len(original)}")
    caps = [int(value) for value in settings.get("length_reaudit_caps", [])]
    initial_cap = int(settings["max_new_tokens"])
    if not caps or caps != sorted(set(caps)) or caps[0] <= initial_cap:
        raise ValueError("length_reaudit_caps must be unique, increasing and above the initial cap")
    rows = [dict(row) for row in original]
    examples = {
        row["id"]: row
        for row in read_jsonl(resolve(root, config["paths"]["split_dir"]) / "reward_audit.jsonl")
    }
    truncated_before = sum(bool(row.get("hit_max_new_tokens")) for row in rows)
    history: list[dict[str, Any]] = []
    prefix_checks: list[bool] = []
    for cap in caps:
        pending = [row for row in rows if bool(row.get("hit_max_new_tokens"))]
        if not pending:
            break
        generator = LocalGenerator(
            model_id=str(actor["model_id"]),
            cache_dir=resolve(root, config["paths"]["hf_cache"]),
            max_new_tokens=cap,
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            local_files_only=bool(actor.get("local_files_only", True)),
            dtype=str(actor.get("dtype", "auto")),
            device_map=str(actor.get("device_map", "auto")),
        )
        progress = tqdm(total=len(pending), desc=f"gsm8k probe length re-audit cap={cap}")
        for row in pending:
            previous_response = str(row["response"])
            started = time.time()
            text, token_data = local_generate(
                generator, str(row["prompt"]), int(row["generation"]["candidate_seed"])
            )
            prefix_consistent = text.startswith(previous_response)
            prefix_checks.append(prefix_consistent)
            if not prefix_consistent:
                raise RuntimeError(
                    "Length re-audit changed a pre-cap prefix for "
                    f"{row['id']} rollout {row['rollout_index']}"
                )
            row["response"] = text
            row["generation"] = {
                **row["generation"],
                "max_new_tokens": cap,
                "length_reaudit_from_cap": int(row["generation"]["max_new_tokens"]),
            }
            row.update(token_data)
            row["score"] = score_response(text, examples[row["id"]]["answer_value"])
            row["features"] = response_features(text, examples[row["id"]]["question"])
            row["latency_seconds"] = time.time() - started
            row["length_reaudited"] = True
            progress.update(1)
        progress.close()
        history.append(
            {
                "cap": cap,
                "regenerated": len(pending),
                "remaining_truncated": sum(bool(row.get("hit_max_new_tokens")) for row in rows),
            }
        )
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows.sort(key=lambda row: (int(row["experiment_order"]), int(row["rollout_index"])))
    remaining = sum(bool(row.get("hit_max_new_tokens")) for row in rows)
    if remaining / len(rows) > 0.01:
        raise RuntimeError(
            f"Length re-audit still has {remaining}/{len(rows)} truncated rollouts after cap {caps[-1]}"
        )
    checkpoint(run_dir / "actor_probe_k4_responses_resolved.jsonl", rows)
    write_json(
        run_dir / "probe_length_reaudit.json",
        {
            "status": "complete",
            "original_file": original_path.name,
            "resolved_file": "actor_probe_k4_responses_resolved.jsonl",
            "n": len(rows),
            "initial_cap": initial_cap,
            "truncated_before": truncated_before,
            "truncation_rate_before": truncated_before / len(rows),
            "truncated_after": remaining,
            "truncation_rate_after": remaining / len(rows),
            "caps": caps,
            "history": history,
            "all_regenerated_prefixes_consistent": all(prefix_checks),
            "original_artifact_preserved": True,
        },
    )


def generate_b_own(
    *, config: dict[str, Any], root: Path, source_rows: list[dict[str, Any]], output_path: Path
) -> None:
    disc = config["discriminator"]
    transport = disc["transport"]
    own = disc["own_generation"]
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    client = OpenRouterClient.from_env(
        max_attempts=int(transport["max_attempts"]),
        max_retry_wait_seconds=float(transport["max_retry_wait_seconds"]),
    )
    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {row["id"] for row in existing}
    progress = tqdm(total=len(source_rows), initial=len(done), desc="gsm8k B own")
    for item in source_rows:
        if item["id"] in done:
            continue
        prompt = task_prompt(item["question"])
        overrides = dict(disc.get("request_overrides", {}))
        overrides["top_p"] = float(own["top_p"])
        payload = make_chat_payload(
            model_id=str(disc["model_id"]),
            prompt=prompt,
            system_prompt=None,
            max_tokens=int(own["max_tokens"]),
            temperature=float(own["temperature"]),
            provider_routing=dict(disc["provider_routing"]),
            request_overrides=overrides,
        )
        started = time.time()
        raw = client.chat_completion(payload)
        text = response_text(raw)
        usage = raw.get("usage", {}) or {}
        choice = (raw.get("choices") or [{}])[0]
        existing.append(
            {
                "id": item["id"],
                "source_index": item["source_index"],
                "experiment_order": item["experiment_order"],
                "variant": "own",
                "rollout_index": 0,
                "source": "B_own",
                "model_id": disc["model_id"],
                "raw_model": raw.get("model"),
                "raw_provider": raw.get("provider"),
                "message_roles": ["user"],
                "system_prompt_present": False,
                "prompt": prompt,
                "prompt_version": TASK_PROMPT_VERSION,
                "prompt_sha256": sha256_text(prompt),
                "response": text,
                "score": score_response(text, item["answer_value"]),
                "features": response_features(text, item["question"]),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "termination_reason": choice.get("finish_reason"),
                "terminated_normally": choice.get("finish_reason") == "stop",
                "generation": {
                    "temperature": float(own["temperature"]),
                    "top_p": float(own["top_p"]),
                    "max_tokens": int(own["max_tokens"]),
                    "provider_routing": disc["provider_routing"],
                    "request_overrides": overrides,
                },
                "usage": usage,
                "raw_api_response": raw,
                "latency_seconds": time.time() - started,
            }
        )
        done.add(item["id"])
        checkpoint(output_path, existing)
        progress.update(1)
        time.sleep(float(transport["inter_request_delay_seconds"]))
    progress.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("actor-audit", "actor-probe", "actor-probe-length-audit", "b-own"),
        required=True,
    )
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    root = Path(config["project"]["root"]).resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(resolve(root, config["paths"]["split_dir"]) / "reward_audit.jsonl")
    if args.mode == "actor-probe-length-audit":
        reaudit_probe_length(config=config, root=root, run_dir=run_dir)
        return
    write_json(run_dir / "protocol_manifest.json", protocol_manifest())
    write_json(run_dir / "resolved_config.json", config)
    if args.mode == "actor-audit":
        generate_actor(config=config, root=root, source_rows=source_rows, output_path=run_dir / "actor_audit_responses.jsonl", mode="audit")
    elif args.mode == "actor-probe":
        generate_actor(config=config, root=root, source_rows=source_rows, output_path=run_dir / "actor_probe_k4_responses.jsonl", mode="probe")
    else:
        generate_b_own(config=config, root=root, source_rows=source_rows, output_path=run_dir / "b_own_responses.jsonl")


if __name__ == "__main__":
    main()
