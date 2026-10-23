from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path
from typing import Any

from .reward import summarize_reward_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact audit for a completed GRPO run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--slurm-out")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    audit: dict[str, Any] = {
        "reward_statistics": summarize_reward_log(run_dir / "reward_calls.jsonl"),
        "adapter": adapter_statistics(checkpoint_dir / "final_adapter" / "adapter_model.safetensors"),
    }
    if args.slurm_out:
        audit["optimization"] = optimization_statistics(Path(args.slurm_out))
    (run_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


def adapter_statistics(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    result: dict[str, Any] = {}
    for kind in ["lora_A", "lora_B"]:
        norms = [value.float().norm().item() for name, value in tensors.items() if kind in name]
        result[kind] = {
            "tensors": len(norms),
            "nonzero_tensors": sum(value > 0 for value in norms),
            "norm_sum": sum(norms),
            "norm_max": max(norms),
        }
    return result


def optimization_statistics(path: Path) -> dict[str, Any]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("{'loss':"):
            continue
        try:
            row = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            continue
        if isinstance(row, dict) and "grad_norm" in row:
            rows.append(row)
    gradients = [float(row["grad_norm"]) for row in rows]
    return {
        "logged_steps": len(rows),
        "nonzero_gradient_steps": sum(value > 0 for value in gradients),
        "grad_norm": {
            "min": min(gradients),
            "mean": statistics.mean(gradients),
            "max": max(gradients),
        },
        "max_clipped_ratio": max(float(row.get("completions/clipped_ratio", 0.0)) for row in rows),
        "mean_entropy": statistics.mean(float(row["entropy"]) for row in rows),
    }


if __name__ == "__main__":
    main()

