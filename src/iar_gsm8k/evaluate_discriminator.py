from __future__ import annotations

import argparse
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iar_phase0.parameters import initialize_parameter_record, write_parameter_artifact

from .data import read_jsonl, sha256_json, write_json
from .generate import generate_b_own, load_config, resolve
from .prompts import protocol_manifest


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty Discriminator evaluation")
    usage = [row.get("usage") or {} for row in rows]
    valid = sum(bool(row["score"]["valid"]) for row in rows)
    correct = sum(bool(row["score"]["correct"]) for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "valid": valid,
        "valid_rate": valid / len(rows),
        "termination_rate": sum(bool(row.get("terminated_normally")) for row in rows) / len(rows),
        "mean_completion_tokens": statistics.mean(
            int(row.get("completion_tokens", 0)) for row in rows
        ),
        "openrouter_cost_usd": sum(float(item.get("cost") or 0.0) for item in usage),
        "models": sorted({str(row.get("raw_model")) for row in rows}),
        "providers": sorted({str(row.get("raw_provider")) for row in rows}),
        "message_roles": sorted({tuple(row.get("message_roles", [])) for row in rows}),
        "system_prompt_present": sorted(
            {bool(row.get("system_prompt_present")) for row in rows}
        ),
        "prompt_versions": sorted({str(row.get("prompt_version")) for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-file", default="validation.jsonl")
    args = parser.parse_args()

    if args.source_file != "validation.jsonl":
        raise ValueError("This control is restricted to validation.jsonl; the official test is sealed")

    config_path = args.config.resolve()
    config = load_config(config_path)
    root = Path(config["project"]["root"]).resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = resolve(root, config["paths"]["split_dir"]) / args.source_file
    examples = read_jsonl(source_path)
    expected = int(config["dataset"]["validation_count"])
    if len(examples) != expected:
        raise ValueError(f"Expected {expected} validation examples, received {len(examples)}")

    resolved = {
        **config,
        "evaluation_control": {
            "source_file": args.source_file,
            "example_count": len(examples),
            "purpose": "DeepSeek own-response accuracy on the same held-out validation used for A0",
            "official_test_accessed": False,
        },
    }
    manifest_path = run_dir / "parameters" / "manifest.json"
    if not manifest_path.exists():
        shutil.copy2(config_path, run_dir / "requested_config.yaml")
        initialize_parameter_record(run_dir, resolved_config=resolved, source_config=config_path)
        write_parameter_artifact(
            run_dir,
            "validation_subset_manifest.json",
            {
                "source_file": args.source_file,
                "n": len(examples),
                "ordered_ids": [row["id"] for row in examples],
                "ids_sha256": sha256_json([row["id"] for row in examples]),
                "rows_sha256": sha256_json(examples),
            },
        )
        write_parameter_artifact(run_dir, "protocol_manifest.json", protocol_manifest())

    output_path = run_dir / "b_own_validation_responses.jsonl"
    generate_b_own(config=config, root=root, source_rows=examples, output_path=output_path)
    rows = read_jsonl(output_path)
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete validation evaluation: {len(rows)}/{expected}")
    result = {
        "run_id": run_dir.name,
        "status": "complete",
        "source_file": args.source_file,
        "official_test_accessed": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **summarize(rows),
    }
    write_json(run_dir / "summary.json", result)
    write_json(
        run_dir / "status.json",
        {"status": "complete", "completed_at": result["completed_at"]},
    )
    print(result)


if __name__ == "__main__":
    main()
