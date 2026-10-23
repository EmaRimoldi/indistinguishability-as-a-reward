#!/usr/bin/env python3
"""Prepare, canary and train the GPT-5 Mini K=10 condition for one domain.

The domain comes from IAR_GPT5MINI_DOMAIN and selects the step directory, the
protocol and the frozen training split. Held-out scoring is not part of this
entry point; the sealed test of each domain is scored by the shared tool.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from iar_campaign.gpt5mini_k10_multidomain import DOMAIN, prepare, run_canary, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    for name in ("canary", "train"):
        item = sub.add_parser(name)
        item.add_argument("--key-pool", type=Path, required=True)
        item.add_argument("--key-name", required=True)
        if name == "train":
            item.add_argument("--lock-dir", type=Path, required=True)
            item.add_argument("--run-id", default=os.environ.get("IAR_GPT5MINI_RUN_ID", "attempt001"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "canary":
        result = run_canary(key_pool=args.key_pool, key_name=args.key_name)
    else:
        os.environ["IAR_GPT5MINI_RUN_ID"] = args.run_id
        result = train(key_pool=args.key_pool, key_name=args.key_name, lock_dir=args.lock_dir)
    print(json.dumps({"domain": DOMAIN, **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
