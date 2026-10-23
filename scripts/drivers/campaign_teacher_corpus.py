#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iar_campaign.reward_pilot import KEY_NAMES
from iar_campaign.teacher_corpus import run_teacher_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Campaign Step 7 SFT teacher corpus")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--key-pool", type=Path, required=True)
    parser.add_argument("--key-name", choices=KEY_NAMES, required=True)
    parser.add_argument("--lock-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_teacher_corpus(
        config_path=args.config.resolve(),
        run_id=args.run_id,
        key_pool=args.key_pool.resolve(),
        key_name=args.key_name,
        lock_dir=args.lock_dir.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
