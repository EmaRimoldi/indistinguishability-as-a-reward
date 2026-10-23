from __future__ import annotations

import gc
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


TARGET_MODEL = "DeepSeek V4 Flash"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Expected JSON objects in {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def coding_plain(question: str) -> str:
    return (
        "You are an expert Python programmer. Solve the following task with a "
        "complete executable Python solution. Return only Python code after "
        f"[BEGIN].\n\nTask:\n{question.strip()}\n\n[BEGIN]\n"
    )


def coding_imitation(question: str) -> str:
    return (
        f"For this response, act as {TARGET_MODEL}. Do not merely imitate its surface "
        f"style: assume the same level of programming competence, problem-solving ability, "
        f"and code quality as {TARGET_MODEL}.\n"
        f"Your goal is to produce a Python solution that would be indistinguishable from "
        f"an answer written by {TARGET_MODEL} for the same task.\n"
        "Do not mention the model name in the answer. Output only executable Python code "
        "after [BEGIN].\n\n"
        + coding_plain(question)
    )


def spider_imitation(task: str) -> str:
    return (
        f"For this response, answer as {TARGET_MODEL} would. Match its level of text-to-SQL "
        "reasoning and answer quality, not merely its surface wording. Produce a response "
        f"that would be indistinguishable from one written by {TARGET_MODEL}. Do not mention "
        "the model name. Preserve the task's exact SQL-only output contract.\n\n"
        + task
    )


def prompts_for_domain(
    domain: str, rows: list[dict[str, Any]], *, tables_json: Path | None
) -> dict[str, list[str]]:
    if domain == "commonsenseqa":
        from iar_cqa.prompts import imitation_prompt, task_prompt

        return {
            "a0": [task_prompt(row["question"], row["choices"]) for row in rows],
            "prompt_imitation": [
                imitation_prompt(row["question"], row["choices"]) for row in rows
            ],
        }
    if domain == "gsm8k":
        from iar_gsm8k.prompts import imitation_prompt, task_prompt

        return {
            "a0": [task_prompt(row["question"]) for row in rows],
            "prompt_imitation": [imitation_prompt(row["question"]) for row in rows],
        }
    if domain == "kodcode":
        return {
            "a0": [coding_plain(str(row["question"])) for row in rows],
            "prompt_imitation": [coding_imitation(str(row["question"])) for row in rows],
        }
    if domain == "spider":
        if tables_json is None:
            raise ValueError("Spider generation requires tables_json")
        from iar_spider.prompts import load_schema_catalog, task_prompt

        schemas = load_schema_catalog(tables_json)
        plain = [
            task_prompt(
                db_id=str(row["db_id"]),
                question=str(row["question"]),
                schema=schemas[str(row["db_id"])],
            )
            for row in rows
        ]
        return {"a0": plain, "prompt_imitation": [spider_imitation(value) for value in plain]}
    raise ValueError(f"Unsupported campaign domain: {domain}")


def actor_source(cache_dir: Path, model_id: str, revision: str) -> Path:
    path = cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(f"Pinned actor snapshot is absent: {path}")
    return path


def run_generation(
    *,
    domain: str,
    split_file: Path,
    output_dir: Path,
    cache_dir: Path,
    model_id: str,
    revision: str,
    tables_json: Path | None,
    variants: tuple[str, ...],
    batch_size: int,
    max_new_tokens: int,
    seed: int,
    campaign_root: Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from iar_grpo.numerics import configure_stable_sdpa

    output_dir = output_dir.resolve()
    campaign_root = (campaign_root or Path("/path/to/runs")).resolve()
    try:
        output_dir.relative_to(campaign_root)
    except ValueError as exc:
        raise ValueError("Campaign generation output escaped the campaign root") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(split_file.resolve())
    if not rows or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Generation split must have unique IDs")
    prompts = prompts_for_domain(domain, rows, tables_json=tables_json)
    if any(variant not in prompts for variant in variants):
        raise ValueError("Unsupported generation variant")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sdpa = configure_stable_sdpa(torch)
    source = actor_source(cache_dir.resolve(), model_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    output_path = output_dir / "candidates.jsonl"
    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {(str(row["id"]), str(row["variant"])) for row in existing}
    planned = {(str(row["id"]), variant) for row in rows for variant in variants}
    if not done <= planned:
        raise RuntimeError("Existing candidate ledger does not match this launch")

    for variant in variants:
        pending = [
            (row, prompt)
            for row, prompt in zip(rows, prompts[variant], strict=True)
            if (str(row["id"]), variant) not in done
        ]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            batch_prompts = [prompt for _, prompt in batch]
            conversations = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
            encoded = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for (source_row, prompt), output in zip(batch, generated, strict=True):
                completion = output[prompt_length:]
                text = tokenizer.decode(completion, skip_special_tokens=True).strip()
                existing.append(
                    {
                        "id": str(source_row["id"]),
                        "source_index": int(source_row["source_index"]),
                        "domain": domain,
                        "variant": variant,
                        "model_id": model_id,
                        "model_revision": revision,
                        "prompt_sha256": sha256_text(prompt),
                        "response": text,
                        "completion_tokens": int(len(completion)),
                        "max_new_tokens": max_new_tokens,
                        "hit_max_new_tokens": int(len(completion)) >= max_new_tokens,
                        "seed": seed,
                        "actor_visible_fields": (
                            ["question", "choices"]
                            if domain == "commonsenseqa"
                            else ["question", "schema"]
                            if domain == "spider"
                            else ["question"]
                        ),
                    }
                )
                done.add((str(source_row["id"]), variant))
            existing.sort(key=lambda row: (str(row["variant"]), str(row["id"])))
            write_jsonl(output_path, existing)

    if done != planned:
        raise RuntimeError("Candidate generation ended with incomplete coverage")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "domain": domain,
        "variants": list(variants),
        "rows_per_variant": len(rows),
        "candidate_count": len(existing),
        "split_file": str(split_file.resolve()),
        "split_file_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "model_id": model_id,
        "model_revision": revision,
        "generation": {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "seed": seed,
        },
        "sdpa": sdpa,
        "official_test_accessed": False,
        "candidate_file_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "truncated_count": sum(bool(row["hit_max_new_tokens"]) for row in existing),
    }
    write_json(output_dir / "summary.json", summary)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary

