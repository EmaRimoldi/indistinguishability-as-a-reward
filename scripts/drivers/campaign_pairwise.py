#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iar_campaign.pairwise_experiment import (
    CHECKPOINTS,
    collect_final_result,
    collect_probe_gate,
    collect_task_validation,
    generate_probe_candidates,
    generate_validation_attempt,
    run_identity_readout,
    run_probe_judge,
    run_training_attempt,
    score_validation_attempt,
)
from iar_campaign.reward_pilot import KEY_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Campaign Step 9 pairwise control")
    commands = parser.add_subparsers(dest="command", required=True)
    probe_generate = commands.add_parser("probe-generate")
    probe_generate.add_argument("--config", type=Path, required=True)
    probe_generate.add_argument("--run-id", required=True)
    probe_judge = commands.add_parser("probe-judge")
    probe_judge.add_argument("--config", type=Path, required=True)
    probe_judge.add_argument("--run-id", required=True)
    probe_judge.add_argument("--key-pool", type=Path, required=True)
    probe_judge.add_argument("--key-name", choices=KEY_NAMES, required=True)
    probe_judge.add_argument("--lock-dir", type=Path, required=True)
    probe_collect = commands.add_parser("probe-collect")
    probe_collect.add_argument("--config", type=Path, action="append", required=True)
    probe_collect.add_argument("--run-id", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--key-pool", type=Path, required=True)
    train.add_argument("--key-name", choices=KEY_NAMES, required=True)
    train.add_argument("--lock-dir", type=Path, required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--step", type=int, choices=CHECKPOINTS, required=True)
    generate.add_argument("--attempt", type=int, default=1)
    score = commands.add_parser("score")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--run-id", required=True)
    score.add_argument("--step", type=int, choices=CHECKPOINTS, required=True)
    score.add_argument("--generation-attempt", type=int, default=1)
    score.add_argument("--scoring-attempt", type=int, default=1)
    score.add_argument("--score-workers", type=int, default=4)
    collect = commands.add_parser("collect-task")
    collect.add_argument("--config", type=Path, action="append", required=True)
    collect.add_argument("--run-id", required=True)
    identity = commands.add_parser("identity-readout")
    identity.add_argument("--config", type=Path, required=True)
    identity.add_argument("--run-id", required=True)
    identity.add_argument("--key-pool", type=Path, required=True)
    identity.add_argument("--key-name", choices=KEY_NAMES, required=True)
    identity.add_argument("--lock-dir", type=Path, required=True)
    identity.add_argument("--attempt", type=int, default=1)
    final = commands.add_parser("collect-final")
    final.add_argument("--config", type=Path, action="append", required=True)
    final.add_argument("--run-id", required=True)
    final.add_argument("--identity-attempt", type=int, default=1)
    args = parser.parse_args()
    if args.command == "probe-generate":
        result = generate_probe_candidates(
            config_path=args.config.resolve(), run_id=args.run_id
        )
    elif args.command == "probe-judge":
        result = run_probe_judge(
            config_path=args.config.resolve(),
            run_id=args.run_id,
            key_pool=args.key_pool.resolve(),
            key_name=args.key_name,
            lock_dir=args.lock_dir.resolve(),
        )
    elif args.command == "probe-collect":
        result = collect_probe_gate(
            config_paths=[path.resolve() for path in args.config], run_id=args.run_id
        )
    elif args.command == "train":
        result = run_training_attempt(
            config_path=args.config.resolve(),
            run_id=args.run_id,
            key_pool=args.key_pool.resolve(),
            key_name=args.key_name,
            lock_dir=args.lock_dir.resolve(),
        )
    elif args.command == "generate":
        result = generate_validation_attempt(
            config_path=args.config.resolve(),
            run_id=args.run_id,
            step=args.step,
            attempt=args.attempt,
        )
    elif args.command == "score":
        result = score_validation_attempt(
            config_path=args.config.resolve(),
            run_id=args.run_id,
            step=args.step,
            generation_attempt=args.generation_attempt,
            scoring_attempt=args.scoring_attempt,
            score_workers=args.score_workers,
        )
    elif args.command == "collect-task":
        result = collect_task_validation(
            config_paths=[path.resolve() for path in args.config], run_id=args.run_id
        )
    elif args.command == "identity-readout":
        result = run_identity_readout(
            config_path=args.config.resolve(),
            run_id=args.run_id,
            key_pool=args.key_pool.resolve(),
            key_name=args.key_name,
            lock_dir=args.lock_dir.resolve(),
            attempt=args.attempt,
        )
    else:
        result = collect_final_result(
            config_paths=[path.resolve() for path in args.config],
            run_id=args.run_id,
            identity_attempt=args.identity_attempt,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
