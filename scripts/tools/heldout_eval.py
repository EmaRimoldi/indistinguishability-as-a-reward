#!/usr/bin/env python3
"""Generate and score a frozen actor or a trained paper checkpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from iar_campaign.four_domain_screen import _adapter_for_step
from iar_campaign.generate import prompts_for_domain, read_jsonl
from iar_campaign.reward_pilot import sha256_file, utc_now, write_json, write_jsonl
from iar_campaign.score_baselines import (
    _score_kodcode,
    _score_simple,
    _score_spider_one,
)
from iar_grpo.config import load_grpo_config


EXPECTED_COUNTS = {
    "commonsenseqa": 1221,
    "gsm8k": 1319,
    "kodcode": 1000,
    "spider": 1034,
}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _adapter_hash(adapter: Path) -> str:
    files = []
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ):
        path = adapter / name
        if path.is_file():
            files.append((name, sha256_file(path)))
    if not files:
        raise RuntimeError(f"No inference adapter files in {adapter}")
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _valid_ids(
    *, domain: str, gate_path: Path | None, test_sha256: str
) -> set[str] | None:
    if domain in {"commonsenseqa", "gsm8k"}:
        return None
    if gate_path is None:
        raise ValueError(f"{domain} requires a frozen reference gate")
    gate = _read_json(gate_path)
    if isinstance(gate.get(domain), list):
        from iar_repro.paths import ROOT, sha256

        manifest = json.loads(
            (ROOT / "reproduction/provenance/desktop-original-sha256.json").read_text()
        )
        test_name = "test1000.jsonl" if domain == "kodcode" else "test1034.jsonl"
        if (
            sha256(gate_path) != manifest["data/reference_gate_valid_ids.json"]
            or test_sha256 != manifest[f"data/{domain}/{test_name}"]
        ):
            raise ValueError("Bundled reference gate or test split drifted")
        return {str(value) for value in gate[domain]}
    if gate.get("passed") is not True:
        raise RuntimeError(f"Reference gate did not pass: {gate_path}")
    entry = (gate.get("domains") or {}).get(domain) or {}
    summary_path = Path(str(entry.get("summary", "")))
    summary = _read_json(summary_path)
    if summary.get("test_sha256") != test_sha256:
        raise RuntimeError("Reference-gate test split differs from this evaluation")
    valid = {str(value) for value in entry.get("valid_ids") or []}
    if not valid:
        raise RuntimeError(f"Reference gate has no valid IDs for {domain}")
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint-run-id",
        help="Run containing the selected checkpoint; defaults to --run-id.",
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=(0, 125, 187, 250, 374, 375, 500, 1000, 1500, 2000),
        required=True,
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Frozen reward-form pilot validation, not held-out test",
    )
    parser.add_argument(
        "--adapter", type=Path, help="Explicit SFT/GRPO adapter; step 0 evaluates A0"
    )
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-gate", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config_path = args.config.resolve()
    import yaml

    raw = yaml.safe_load(config_path.read_text())
    if raw.get("_reproduction_condition") == "sft":
        from iar_grpo.config import GrpoTrialConfig

        config = GrpoTrialConfig(raw, config_path)
    else:
        config = load_grpo_config(config_path)
    domain = str((config.raw.get("campaign") or {}).get("domain"))
    if domain not in EXPECTED_COUNTS:
        raise ValueError(f"Unsupported domain: {domain}")
    training_summary = (
        config.resolve(config.raw["paths"]["experiments"])
        / args.run_id
        / "summary.json"
    )
    if args.step and _read_json(training_summary).get("status") != "complete":
        raise RuntimeError("Held-out generation requires completed training")
    examples = read_jsonl(args.test_file.resolve())
    expected_count = 300 if args.validation else EXPECTED_COUNTS[domain]
    if len(examples) != expected_count or len(
        {str(row["id"]) for row in examples}
    ) != len(examples):
        raise RuntimeError(f"Unexpected held-out split coverage for {domain}")
    test_sha256 = sha256_file(args.test_file)
    checkpoint_run_id = args.checkpoint_run_id or args.run_id
    checkpoint_root = (
        config.resolve(config.raw["paths"]["checkpoints"]) / checkpoint_run_id
    )
    adapter = args.adapter or (
        _adapter_for_step(
            checkpoint_root, args.step, int(config.raw["training"]["max_steps"])
        )
        if args.step
        else None
    )
    if args.step and adapter is None:
        raise RuntimeError("The held-out run requires a trained adapter")
    if not args.step and args.adapter:
        raise ValueError("A0 evaluation cannot load an adapter")
    adapter_hash = _adapter_hash(adapter) if adapter else None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    valid_ids = (
        _valid_ids(
            domain=domain,
            gate_path=args.reference_gate.resolve() if args.reference_gate else None,
            test_sha256=test_sha256,
        )
        if not args.validation
        else None
    )
    if args.validation:
        if raw.get("_reproduction_condition") != "reward_form_pilot" or domain not in (
            "gsm8k",
            "kodcode",
        ):
            raise ValueError(
                "Validation mode is restricted to the frozen reward-form pilot"
            )
        from iar_repro.paths import ROOT, sha256

        name = "validation.jsonl" if domain == "gsm8k" else "validation300.jsonl"
        if sha256(args.test_file) != sha256(ROOT / "data" / domain / name):
            raise ValueError("Pilot validation split mismatch")
        if domain == "kodcode":
            protocol = _read_json(Path(raw["evaluation"]["kodcode_scoring_protocol"]))
            if (
                protocol["validation_file_sha256"] != test_sha256
                or protocol.get("decision") != "GO"
            ):
                raise ValueError("Pilot KodCode reference gate mismatch")
            valid_ids = {str(v) for v in protocol["scored_ids"]}
    write_json(
        output / "manifest.json",
        {
            "status": "frozen_before_generation",
            "created_at": utc_now(),
            "domain": domain,
            "run_id": args.run_id,
            "checkpoint_run_id": checkpoint_run_id,
            "selected_step": args.step,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "adapter": str(adapter),
            "adapter_sha256": adapter_hash,
            "test_file": str(args.test_file.resolve()),
            "test_sha256": test_sha256,
            "test_n": len(examples),
            "reference_gate": str(args.reference_gate.resolve())
            if args.reference_gate
            else None,
            "official_test_accessed": not args.validation,
        },
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from iar_grpo.numerics import configure_stable_sdpa

    random.seed(23)
    torch.manual_seed(23)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(23)
    tables = (
        Path(config.raw["evaluation"]["spider_tables_json"])
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    tokenizer = AutoTokenizer.from_pretrained(
        config.actor_source(),
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = None
    model = None
    candidates: list[dict] = []
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.actor_source(),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model = (
            PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
            if adapter
            else base_model
        )
        model.eval()
        batch_size = int(config.raw["evaluation"]["batch_size"])
        max_new_tokens = int(config.raw["evaluation"].get("max_new_tokens", 768))
        sdpa = configure_stable_sdpa(torch)
        for start in range(0, len(examples), batch_size):
            batch_rows = examples[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]
            conversations = [
                [{"role": "user", "content": prompt}] for prompt in batch_prompts
            ]
            encoded = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
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
            for row, output_ids in zip(batch_rows, generated, strict=True):
                completion = output_ids[prompt_length:]
                candidates.append(
                    {
                        "id": str(row["id"]),
                        "source_index": int(row["source_index"]),
                        "domain": domain,
                        "run_id": args.run_id,
                        "step": args.step,
                        "response": tokenizer.decode(
                            completion, skip_special_tokens=True
                        ).strip(),
                        "completion_tokens": len(completion),
                        "max_new_tokens": max_new_tokens,
                        "hit_max_new_tokens": len(completion) >= max_new_tokens,
                    }
                )
        if len(candidates) != len(examples):
            raise RuntimeError("Incomplete held-out generation")
        write_jsonl(output / "candidates.jsonl", candidates)
        del model
        del base_model
        model = None
        base_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        examples_by_id = {str(row["id"]): row for row in examples}
        scored = (
            candidates
            if valid_ids is None
            else [row for row in candidates if row["id"] in valid_ids]
        )
        if domain in {"commonsenseqa", "gsm8k"}:
            _score_simple(domain, examples, candidates)
        elif domain == "kodcode":
            os.environ.setdefault(
                "IAR_KODCODE_TEST_PYTHONPATH", os.environ.get("PYTHONPATH", "")
            )
            payloads = [
                (
                    str(row["response"]),
                    str(examples_by_id[row["id"]]["hidden_tests"]),
                    60,
                )
                for row in scored
            ]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers
            ) as pool:
                scores = list(pool.map(_score_kodcode, payloads))
            for row, score in zip(scored, scores, strict=True):
                row["score"] = score
        else:
            database_root = Path(config.raw["evaluation"]["spider_database_root"])
            payloads = [
                (examples_by_id[row["id"]], row, database_root) for row in scored
            ]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as pool:
                scores = list(pool.map(_score_spider_one, payloads))
            for row, score in zip(scored, scores, strict=True):
                row.update(score)
        if valid_ids is not None:
            for row in candidates:
                if row["id"] not in valid_ids:
                    row["score"] = {
                        "correct": False,
                        "valid": False,
                        "excluded_reference_invalid": True,
                    }
        correct = sum(bool(row["score"]["correct"]) for row in scored)
        write_jsonl(output / "scored_candidates.jsonl", candidates)
        write_json(
            output / "summary.json",
            {
                "status": "complete",
                "completed_at": utc_now(),
                "domain": domain,
                "run_id": args.run_id,
                "selected_step": args.step,
                "n": len(candidates),
                "scored_n": len(scored),
                "correct": correct,
                "task_utility": correct / len(scored),
                "reference_invalid_excluded": len(candidates) - len(scored),
                "valid_output_n": sum(
                    bool(row["score"].get("valid", True)) for row in scored
                ),
                "status_counts": dict(
                    sorted(
                        Counter(
                            str(row["score"].get("status", "scored")) for row in scored
                        ).items()
                    )
                ),
                "candidate_file": str(output / "scored_candidates.jsonl"),
                "candidate_file_sha256": sha256_file(
                    output / "scored_candidates.jsonl"
                ),
                "test_sha256": test_sha256,
                "adapter": str(adapter),
                "adapter_sha256": adapter_hash,
                "generation": {
                    "seed": 23,
                    "sdpa": sdpa,
                    "truncated": sum(row["hit_max_new_tokens"] for row in candidates),
                },
                "official_test_accessed": not args.validation,
            },
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "partial_candidates": len(candidates),
                "official_test_accessed": not args.validation,
            },
        )
        raise
    finally:
        if model is not None:
            del model
        if base_model is not None:
            del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
