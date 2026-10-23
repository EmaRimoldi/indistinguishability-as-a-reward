from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from iar_phase0.config import load_config
from iar_phase0.data import load_mbpp
from iar_phase0.env import load_env_file

from .config import load_grpo_config
from .evaluate_hpo import paired_comparison
from .evaluate_pilot import adapter_for_step, evaluate_state, manifest_for, select_examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a persisted checkpoint while the primary GRPO run continues"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", required=True, type=int)
    args = parser.parse_args()

    config = load_grpo_config(args.config)
    max_step = int(config.raw["training"]["max_steps"])
    if args.step <= 0 or args.step >= max_step:
        raise ValueError(f"Live checkpoint step must be in (0, {max_step}), got {args.step}")

    run_dir = config.resolve(config.raw["paths"]["experiments"]) / args.run_id
    checkpoint_root = config.resolve(config.raw["paths"]["checkpoints"]) / args.run_id
    adapter = adapter_for_step(checkpoint_root, args.step, max_step)
    output_root = run_dir / "live_evaluations" / "validation" / f"checkpoint_{args.step:04d}"
    output_root.mkdir(parents=True, exist_ok=False)

    load_env_file(config.resolve(config.raw["paths"]["openrouter_env"]))
    phase0 = load_config(config.resolve(config.raw["dataset"]["phase0_config"]))
    mbpp = load_mbpp(phase0)
    evaluation = config.raw["evaluation"]
    validation_spec = evaluation["validation"]
    validation = select_examples(mbpp, validation_spec)

    baseline_rows, baseline_summary = evaluate_state(
        config=config,
        examples=validation,
        adapter_dir=None,
        state_name="step_0000",
        split_name="validation_live",
        num_generations=int(validation_spec["num_generations"]),
        temperature=float(validation_spec["temperature"]),
        top_p=float(validation_spec["top_p"]),
        output_dir=output_root / "step_0000",
        seed=int(config.raw["project"]["seed"]),
    )
    candidate_rows, candidate_summary = evaluate_state(
        config=config,
        examples=validation,
        adapter_dir=adapter,
        state_name=f"step_{args.step:04d}",
        split_name="validation_live",
        num_generations=int(validation_spec["num_generations"]),
        temperature=float(validation_spec["temperature"]),
        top_p=float(validation_spec["top_p"]),
        output_dir=output_root / f"step_{args.step:04d}",
        seed=int(config.raw["project"]["seed"]),
    )
    comparison = paired_comparison(
        baseline_rows,
        candidate_rows,
        step=args.step,
        z=float(evaluation["paired_lcb_z"]),
        max_pass_drop=float(evaluation["max_pass_at_1_drop"]),
    )
    comparison.update(
        {
            "sampled_fooling_delta": float(candidate_summary["sampled_fooling_rate"])
            - float(baseline_summary["sampled_fooling_rate"]),
            "argmax_fooling_delta": float(candidate_summary["argmax_fooling_rate"])
            - float(baseline_summary["argmax_fooling_rate"]),
        }
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "step": args.step,
        "split_manifest": manifest_for(validation),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "comparison": comparison,
    }
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
