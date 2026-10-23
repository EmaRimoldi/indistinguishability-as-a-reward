from __future__ import annotations

import argparse
import json
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(tmp, rows)
    tmp.replace(path)


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
    progress = tqdm(total=total, initial=len(keys), desc=f"cqa actor {mode}")
    for item in source_rows:
        for variant in variants:
            prompt = (
                task_prompt(item["question"], item["choices"])
                if variant == "plain"
                else imitation_prompt(item["question"], item["choices"])
            )
            prompt_version = TASK_PROMPT_VERSION if variant == "plain" else IMITATION_PROMPT_VERSION
            for rollout_index in range(generations):
                key = (item["id"], variant, rollout_index)
                if key in keys:
                    continue
                started = time.time()
                text = generator.generate(prompt)
                score = score_response(text, item["answer_key"])
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
                        "prompt_version": prompt_version,
                        "prompt_sha256": sha256_text(prompt),
                        "response": text,
                        "score": score,
                        "features": response_features(text),
                        "generation": {
                            "temperature": float(settings["temperature"]),
                            "top_p": float(settings["top_p"]),
                            "max_new_tokens": int(settings["max_new_tokens"]),
                            "seed": seed,
                        },
                        "latency_seconds": time.time() - started,
                    }
                )
                keys.add(key)
                checkpoint(output_path, existing)
                progress.update(1)
    progress.close()


def generate_b_own(
    *,
    config: dict[str, Any],
    root: Path,
    source_rows: list[dict[str, Any]],
    output_path: Path,
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
    progress = tqdm(total=len(source_rows), initial=len(done), desc="cqa B own")
    for item in source_rows:
        if item["id"] in done:
            continue
        prompt = task_prompt(item["question"], item["choices"])
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
                "prompt_version": TASK_PROMPT_VERSION,
                "prompt_sha256": sha256_text(prompt),
                "response": text,
                "score": score_response(text, item["answer_key"]),
                "features": response_features(text),
                "generation": {
                    "temperature": float(own["temperature"]),
                    "top_p": float(own["top_p"]),
                    "max_tokens": int(own["max_tokens"]),
                    "provider_routing": disc["provider_routing"],
                    "request_overrides": overrides,
                },
                "usage": raw.get("usage", {}),
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
    parser.add_argument("--mode", choices=("actor-audit", "actor-probe", "b-own"), required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    root = Path(config["project"]["root"]).resolve()
    split_dir = resolve(root, config["paths"]["split_dir"])
    source_rows = read_jsonl(split_dir / "reward_audit.jsonl")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
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
