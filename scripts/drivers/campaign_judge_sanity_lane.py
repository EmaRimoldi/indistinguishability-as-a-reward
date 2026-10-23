#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iar_campaign.judge_sanity import run_lane


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("teacher", "judge"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--key-name", required=True)
    parser.add_argument("--key-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_lane(
        config_path=args.config.resolve(),
        phase=args.phase,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        key_name=args.key_name,
        key_pool=args.key_pool.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
