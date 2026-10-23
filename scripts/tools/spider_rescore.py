#!/usr/bin/env python3
"""Score a complete, frozen Spider candidate file without regenerating it."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from iar_campaign.generate import read_jsonl
from iar_campaign.reward_pilot import sha256_file, utc_now, write_json, write_jsonl
from iar_campaign.score_baselines import _score_spider_one
from iar_grpo.config import load_grpo_config


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--candidates-file", type=Path, required=True)
    parser.add_argument("--reference-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = load_grpo_config(args.config.resolve())
    if str((config.raw.get("campaign") or {}).get("domain")) != "spider":
        raise ValueError("This rescorer supports Spider only")
    examples = read_jsonl(args.test_file.resolve())
    candidates = read_jsonl(args.candidates_file.resolve())
    if len(examples) != 1034 or len(candidates) != 1034:
        raise RuntimeError("Expected complete Spider official test coverage")
    example_by_id = {str(row["id"]): row for row in examples}
    if set(example_by_id) != {str(row["id"]) for row in candidates}:
        raise RuntimeError("Candidate and official-test IDs differ")
    test_sha = sha256_file(args.test_file)
    gate = read_object(args.reference_gate.resolve())
    entry = (gate.get("domains") or {}).get("spider") or {}
    reference = read_object(Path(str(entry.get("summary", ""))))
    if gate.get("passed") is not True or reference.get("test_sha256") != test_sha:
        raise RuntimeError(
            "Frozen Spider reference gate does not match the official test"
        )
    valid_ids = {str(value) for value in entry.get("valid_ids") or []}
    if not valid_ids or not valid_ids <= set(example_by_id):
        raise RuntimeError("Invalid Spider reference-gate IDs")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scored = [row for row in candidates if str(row["id"]) in valid_ids]
    database_root = Path(config.raw["evaluation"]["spider_database_root"])
    payloads = [(example_by_id[str(row["id"])], row, database_root) for row in scored]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        scores = list(pool.map(_score_spider_one, payloads))
    for row, score in zip(scored, scores, strict=True):
        row.update(score)
    for row in candidates:
        if str(row["id"]) not in valid_ids:
            row["score"] = {
                "correct": False,
                "valid": False,
                "excluded_reference_invalid": True,
            }
    correct = sum(bool(row["score"]["correct"]) for row in scored)
    write_jsonl(output / "scored_candidates.jsonl", candidates)
    write_json(
        output / "manifest.json",
        {
            "status": "frozen_before_rescoring",
            "created_at": utc_now(),
            "domain": "spider",
            "selected_step": 1500,
            "test_file": str(args.test_file.resolve()),
            "test_sha256": test_sha,
            "candidate_file": str(args.candidates_file.resolve()),
            "candidate_file_sha256": sha256_file(args.candidates_file),
            "reference_gate": str(args.reference_gate.resolve()),
            "official_test_accessed_by_prior_generation": True,
        },
    )
    write_json(
        output / "summary.json",
        {
            "status": "complete",
            "completed_at": utc_now(),
            "domain": "spider",
            "selected_step": 1500,
            "n": len(candidates),
            "scored_n": len(scored),
            "correct": correct,
            "task_utility": correct / len(scored),
            "reference_invalid_excluded": len(candidates) - len(scored),
            "rescore_only": True,
        },
    )


if __name__ == "__main__":
    main()
