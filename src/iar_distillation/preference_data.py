from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .teacher_data import (
    load_sources,
    read_jsonl,
    score_output,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def actor_source(root: Path, config: dict[str, Any]) -> str:
    actor = config["actor"]
    model_id = str(actor["model_id"])
    if not bool(actor.get("local_files_only", True)):
        return model_id
    cache = resolve(root, config["paths"]["hf_cache"])
    slug = model_id.replace("/", "--")
    snapshots = sorted((cache / f"models--{slug}" / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"No cached snapshot for offline actor {model_id!r}")
    return str(snapshots[-1])


def checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, sorted(rows, key=lambda row: int(row["experiment_order"])))


def load_aligned_rows(
    root: Path,
    config: dict[str, Any],
    domain: str,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    domain_cfg = config["domains"][domain]
    run_dir = resolve(root, config["paths"]["teacher_root"]) / domain
    projected_sources = read_jsonl(run_dir / "source_prompts.jsonl")
    teacher_config = yaml.safe_load(
        resolve(root, config["paths"]["teacher_config"]).read_text(encoding="utf-8")
    )
    sources = load_sources(root, domain, teacher_config["domains"][domain])
    teacher = read_jsonl(run_dir / "responses.jsonl")
    expected = int(domain_cfg["expected_count"])
    if len(sources) != expected or len(projected_sources) != expected or len(teacher) != expected:
        raise ValueError(
            f"{domain}: expected {expected} reconstructed/projected sources and teacher responses; "
            f"found {len(sources)}, {len(projected_sources)}, and {len(teacher)}"
        )
    source_by_id = {str(row["id"]): row for row in sources}
    teacher_by_id = {str(row["id"]): row for row in teacher}
    if len(source_by_id) != expected or source_by_id.keys() != teacher_by_id.keys():
        raise ValueError(f"{domain}: source/teacher IDs are duplicated or misaligned")
    ordered_sources = sorted(sources, key=lambda row: int(row["experiment_order"]))
    if [int(row["experiment_order"]) for row in ordered_sources] != list(range(expected)):
        raise ValueError(f"{domain}: source order is not frozen and contiguous")
    ordered_teacher = [teacher_by_id[str(row["id"])] for row in ordered_sources]
    projected_by_id = {str(row["id"]): row for row in projected_sources}
    if len(projected_by_id) != expected or projected_by_id.keys() != source_by_id.keys():
        raise ValueError(f"{domain}: reconstructed/projected source IDs are misaligned")
    for source, response in zip(ordered_sources, ordered_teacher, strict=True):
        if source["prompt"] != response["prompt"]:
            raise ValueError(f"{domain}: teacher prompt mismatch for {source['id']}")
        projected = projected_by_id[str(source["id"])]
        for field in (
            "id",
            "task_id",
            "dataset_index",
            "experiment_order",
            "prompt",
            "prompt_version",
        ):
            if projected[field] != source[field]:
                raise ValueError(f"{domain}: projected source mismatch for {source['id']} field {field}")
    return run_dir, ordered_sources, ordered_teacher


def generate_qwen(config: dict[str, Any], root: Path, domain: str) -> dict[str, Any]:
    run_dir, sources, _ = load_aligned_rows(root, config, domain)
    domain_cfg = config["domains"][domain]
    output_path = run_dir / "qwen_base_responses.jsonl"
    existing = read_jsonl(output_path) if output_path.exists() else []
    existing_by_id = {str(row["id"]): row for row in existing}
    if len(existing_by_id) != len(existing):
        raise ValueError(f"{domain}: duplicate cached Qwen response IDs")
    source_ids = {str(row["id"]) for row in sources}
    if not set(existing_by_id).issubset(source_ids):
        raise ValueError(f"{domain}: cached Qwen responses contain unknown IDs")

    pending = [row for row in sources if str(row["id"]) not in existing_by_id]
    if not pending:
        rows = [existing_by_id[str(item["id"])] for item in sources]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "domain": domain,
            "status": "complete",
            "n": len(rows),
            "correct": sum(bool(row["score"].get("correct")) for row in rows),
            "accuracy": sum(bool(row["score"].get("correct")) for row in rows) / len(rows),
            "valid": sum(bool(row["score"].get("valid")) for row in rows),
            "terminated": sum(bool(row["terminated_with_eos"]) for row in rows),
            "truncated": sum(bool(row["hit_max_new_tokens"]) for row in rows),
            "output_sha256": sha256_file(output_path),
            "completed_at": utc_now(),
            "cache_reused_without_model_load": True,
        }
        write_json(run_dir / "qwen_base_summary.json", summary)
        return summary

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    cache = resolve(root, config["paths"]["hf_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    source = actor_source(root, config)
    actor = config["actor"]
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=str(cache),
        local_files_only=bool(actor.get("local_files_only", True)),
        padding_side="left",
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        cache_dir=str(cache),
        local_files_only=bool(actor.get("local_files_only", True)),
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    seed = int(domain_cfg.get("seed", config["project"]["seed"]))
    set_seed(seed)
    batch_size = int(config["generation"]["batch_size"])
    max_new_tokens = int(domain_cfg["max_new_tokens"])
    timeout = int(domain_cfg.get("score_timeout_seconds", 5))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        conversations = [
            [{"role": "user", "content": str(item["prompt"])}] for item in batch
        ]
        inputs = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        started = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = int(inputs["input_ids"].shape[1])
        batch_seconds = time.time() - started
        for index, (item, output) in enumerate(zip(batch, outputs, strict=True)):
            completion = output[prompt_width:]
            eos = (completion == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            terminated = len(eos) > 0
            if terminated:
                completion = completion[: int(eos[0]) + 1]
            raw_response = tokenizer.decode(completion, skip_special_tokens=True).strip()
            response, score = score_output(domain, raw_response, item, timeout)
            prompt_tokens = int(inputs["attention_mask"][index].sum().item())
            existing_by_id[str(item["id"])] = {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "id": str(item["id"]),
                "task_id": str(item["task_id"]),
                "domain": domain,
                "dataset_index": int(item["dataset_index"]),
                "experiment_order": int(item["experiment_order"]),
                "prompt": str(item["prompt"]),
                "prompt_version": str(item["prompt_version"]),
                "prompt_sha256": sha256_text(str(item["prompt"])),
                "source": "Qwen_base_greedy",
                "model_id": str(actor["model_id"]),
                "raw_response": raw_response,
                "response": response,
                "response_sha256": sha256_text(response),
                "score": score,
                "generation": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                },
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(len(completion)),
                "terminated_with_eos": bool(terminated),
                "hit_max_new_tokens": bool(
                    len(completion) >= max_new_tokens and not terminated
                ),
                "batch_latency_seconds": batch_seconds,
            }
        checkpoint(output_path, list(existing_by_id.values()))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows = [existing_by_id[str(item["id"])] for item in sources]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "domain": domain,
        "status": "complete",
        "n": len(rows),
        "correct": sum(bool(row["score"].get("correct")) for row in rows),
        "accuracy": sum(bool(row["score"].get("correct")) for row in rows) / len(rows),
        "valid": sum(bool(row["score"].get("valid")) for row in rows),
        "terminated": sum(bool(row["terminated_with_eos"]) for row in rows),
        "truncated": sum(bool(row["hit_max_new_tokens"]) for row in rows),
        "output_sha256": sha256_file(output_path),
        "completed_at": utc_now(),
    }
    write_json(run_dir / "qwen_base_summary.json", summary)
    return summary


def build_dpo(config: dict[str, Any], root: Path, domain: str) -> dict[str, Any]:
    run_dir, sources, teacher = load_aligned_rows(root, config, domain)
    qwen_path = run_dir / "qwen_base_responses.jsonl"
    if not qwen_path.exists():
        raise FileNotFoundError(f"Generate frozen Qwen responses first: {qwen_path}")
    qwen = read_jsonl(qwen_path)
    qwen_by_id = {str(row["id"]): row for row in qwen}
    if len(qwen_by_id) != len(sources):
        raise ValueError(f"{domain}: incomplete or duplicated Qwen response set")

    rows: list[dict[str, Any]] = []
    quadrants: Counter[str] = Counter()
    chosen_sources: Counter[str] = Counter()
    identical = 0
    for source, deepseek in zip(sources, teacher, strict=True):
        qwen_row = qwen_by_id[str(source["id"])]
        if qwen_row["prompt"] != source["prompt"]:
            raise ValueError(f"{domain}: Qwen prompt mismatch for {source['id']}")
        deepseek_correct = bool(deepseek["score"].get("correct"))
        qwen_correct = bool(qwen_row["score"].get("correct"))
        quadrant = f"qwen_{'correct' if qwen_correct else 'wrong'}__deepseek_{'correct' if deepseek_correct else 'wrong'}"
        quadrants[quadrant] += 1

        flip_to_qwen = qwen_correct and not deepseek_correct
        chosen_row = qwen_row if flip_to_qwen else deepseek
        rejected_row = deepseek if flip_to_qwen else qwen_row
        chosen_source = "qwen" if flip_to_qwen else "deepseek"
        rejected_source = "deepseek" if flip_to_qwen else "qwen"
        chosen_sources[chosen_source] += 1
        same = str(chosen_row["response"]) == str(rejected_row["response"])
        identical += int(same)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": str(source["id"]),
                "domain": domain,
                "dataset_index": int(source["dataset_index"]),
                "experiment_order": int(source["experiment_order"]),
                "prompt": [{"role": "user", "content": str(source["prompt"])}],
                "chosen": [{"role": "assistant", "content": str(chosen_row["response"])}],
                "rejected": [
                    {"role": "assistant", "content": str(rejected_row["response"])}
                ],
                "chosen_source": chosen_source,
                "rejected_source": rejected_source,
                "qwen_correct": qwen_correct,
                "deepseek_correct": deepseek_correct,
                "qwen_valid": bool(qwen_row["score"].get("valid")),
                "deepseek_valid": bool(deepseek["score"].get("valid")),
                "preference_rule": (
                    "qwen_chosen_only_when_qwen_correct_and_deepseek_wrong"
                ),
                "preference_quadrant": quadrant,
                "identical_response": same,
                "qwen_response_sha256": str(qwen_row["response_sha256"]),
                "deepseek_response_sha256": str(deepseek["response_sha256"]),
            }
        )

    output_path = run_dir / "dpo_pairs.jsonl"
    write_jsonl(output_path, rows)
    matched_sft_path = run_dir / "sft_dpo_chosen.jsonl"
    write_jsonl(
        matched_sft_path,
        [
            {
                "schema_version": SCHEMA_VERSION,
                "id": row["id"],
                "domain": row["domain"],
                "prompt": row["prompt"],
                "completion": row["chosen"],
                "chosen_source": row["chosen_source"],
                "qwen_correct": row["qwen_correct"],
                "deepseek_correct": row["deepseek_correct"],
                "source_dpo_pair_sha256": sha256_text(
                    json.dumps(row, ensure_ascii=False, sort_keys=True)
                ),
            }
            for row in rows
        ],
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "domain": domain,
        "status": "complete",
        "n": len(rows),
        "preference_rule": "DeepSeek chosen by default; flip iff Qwen correct and DeepSeek wrong",
        "quadrants": dict(sorted(quadrants.items())),
        "chosen_sources": dict(sorted(chosen_sources.items())),
        "identical_responses": identical,
        "qwen_accuracy": sum(bool(row["qwen_correct"]) for row in rows) / len(rows),
        "deepseek_accuracy": sum(bool(row["deepseek_correct"]) for row in rows) / len(rows),
        "chosen_accuracy": sum(
            bool(row["qwen_correct"] if row["chosen_source"] == "qwen" else row["deepseek_correct"])
            for row in rows
        )
        / len(rows),
        "output_sha256": sha256_file(output_path),
        "matched_sft_sha256": sha256_file(matched_sft_path),
        "completed_at": utc_now(),
    }
    write_json(run_dir / "dpo_summary.json", summary)
    audit_dpo(config, root, domain)
    return summary


def audit_dpo(config: dict[str, Any], root: Path, domain: str) -> dict[str, Any]:
    run_dir, sources, teacher = load_aligned_rows(root, config, domain)
    qwen = read_jsonl(run_dir / "qwen_base_responses.jsonl")
    dpo = read_jsonl(run_dir / "dpo_pairs.jsonl")
    matched_sft = read_jsonl(run_dir / "sft_dpo_chosen.jsonl")
    expected = int(config["domains"][domain]["expected_count"])
    if not (
        len(sources)
        == len(teacher)
        == len(qwen)
        == len(dpo)
        == len(matched_sft)
        == expected
    ):
        raise ValueError(f"{domain}: DPO artifact counts are not exactly {expected}")
    source_by_id = {str(row["id"]): row for row in sources}
    teacher_by_id = {str(row["id"]): row for row in teacher}
    qwen_by_id = {str(row["id"]): row for row in qwen}
    if any(len(table) != expected for table in (source_by_id, teacher_by_id, qwen_by_id)):
        raise ValueError(f"{domain}: duplicate source, teacher, or Qwen IDs")
    seen: set[str] = set()
    violations: list[str] = []
    for row in dpo:
        row_id = str(row["id"])
        if row_id in seen or row_id not in source_by_id:
            violations.append(f"duplicate-or-unknown:{row_id}")
            continue
        seen.add(row_id)
        source = source_by_id[row_id]
        deepseek = teacher_by_id[row_id]
        qwen_row = qwen_by_id[row_id]
        flip = bool(qwen_row["score"].get("correct")) and not bool(
            deepseek["score"].get("correct")
        )
        expected_chosen = "qwen" if flip else "deepseek"
        expected_rejected = "deepseek" if flip else "qwen"
        expected_chosen_text = qwen_row["response"] if flip else deepseek["response"]
        expected_rejected_text = deepseek["response"] if flip else qwen_row["response"]
        if row["prompt"] != [{"role": "user", "content": source["prompt"]}]:
            violations.append(f"prompt:{row_id}")
        if row["chosen_source"] != expected_chosen or row["rejected_source"] != expected_rejected:
            violations.append(f"rule:{row_id}")
        if row["chosen"] != [{"role": "assistant", "content": expected_chosen_text}]:
            violations.append(f"chosen:{row_id}")
        if row["rejected"] != [{"role": "assistant", "content": expected_rejected_text}]:
            violations.append(f"rejected:{row_id}")
    if seen != set(source_by_id):
        violations.append("missing-ids")
    matched_by_id = {str(row["id"]): row for row in matched_sft}
    if len(matched_by_id) != expected or matched_by_id.keys() != seen:
        violations.append("matched-sft-ids")
    else:
        for row in dpo:
            matched = matched_by_id[str(row["id"])]
            if matched["prompt"] != row["prompt"] or matched["completion"] != row["chosen"]:
                violations.append(f"matched-sft-projection:{row['id']}")
    if violations:
        raise ValueError(f"{domain}: DPO audit failed: {violations[:10]}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "domain": domain,
        "status": "PASS",
        "n": expected,
        "rule_violations": 0,
        "audited_at": utc_now(),
        "dpo_sha256": sha256_file(run_dir / "dpo_pairs.jsonl"),
        "matched_sft_sha256": sha256_file(run_dir / "sft_dpo_chosen.jsonl"),
        "qwen_sha256": sha256_file(run_dir / "qwen_base_responses.jsonl"),
    }
    write_json(run_dir / "dpo_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen Qwen/DeepSeek DPO pairs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", choices=("mbpp", "commonsenseqa", "gsm8k"), required=True)
    parser.add_argument("--stage", choices=("generate", "build", "audit", "all"), default="all")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    if args.stage in {"generate", "all"}:
        print(json.dumps(generate_qwen(config, root, args.domain), indent=2))
    if args.stage in {"build", "all"}:
        print(json.dumps(build_dpo(config, root, args.domain), indent=2))
    if args.stage == "audit":
        print(json.dumps(audit_dpo(config, root, args.domain), indent=2))


if __name__ == "__main__":
    main()
