#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iar_campaign.four_domain_screen import (
    CHECKPOINTS,
    collect_screen,
    generate_validation_attempt,
    run_training_attempt,
    score_validation_attempt,
)
from iar_campaign.reward_pilot import KEY_NAMES

SCREEN_KEY_NAMES = ("OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3", "OPENROUTER_API_KEY_5", "OPENROUTER_API_KEY_6")


def main() -> None:
    parser = argparse.ArgumentParser(description="Campaign Step 5 four-domain screen")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--key-pool", type=Path, required=True)
    train.add_argument("--key-name", choices=SCREEN_KEY_NAMES, required=True)
    train.add_argument("--lock-dir", type=Path, required=True)
    train.add_argument("--resume-from-checkpoint", type=Path)
    train.add_argument("--resume-reward-log", type=Path)
    train.add_argument("--resume-logical-calls", type=int)
    generate = commands.add_parser("generate")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--step", type=int, choices=CHECKPOINTS[1:], required=True)
    generate.add_argument("--attempt", type=int, default=1)
    score = commands.add_parser("score")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--run-id", required=True)
    score.add_argument("--step", type=int, choices=CHECKPOINTS[1:], required=True)
    score.add_argument("--generation-attempt", type=int, default=1)
    score.add_argument("--scoring-attempt", type=int, default=1)
    score.add_argument("--score-workers", type=int, default=4)
    collect = commands.add_parser("collect")
    collect.add_argument("--step-root", type=Path, required=True)
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--config-name", required=True)
    args = parser.parse_args()
    if args.command == "train":
        result = run_training_attempt(
            config_path=args.config.resolve(),
            run_id=args.run_id,
            key_pool=args.key_pool.resolve(),
            key_name=args.key_name,
            lock_dir=args.lock_dir.resolve(),
            resume_from_checkpoint=(
                args.resume_from_checkpoint.resolve()
                if args.resume_from_checkpoint is not None
                else None
            ),
            resume_reward_log=(
                args.resume_reward_log.resolve()
                if args.resume_reward_log is not None
                else None
            ),
            resume_logical_calls=args.resume_logical_calls,
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
    else:
        result = collect_screen(
            step_root=args.step_root.resolve(),
            run_id=args.run_id,
            config_name=args.config_name,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
