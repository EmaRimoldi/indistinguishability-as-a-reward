#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from iar_campaign.kodcode_gpt5mini_k10 import evaluate_heldout, prepare, run_canary, train

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    for name in ("canary", "train"):
        item = sub.add_parser(name); item.add_argument("--key-pool", type=Path, required=True); item.add_argument("--key-name", required=True)
        if name == "train": item.add_argument("--lock-dir", type=Path, required=True)
    sub.add_parser("test")
    args = parser.parse_args()
    if args.command == "prepare": result = prepare()
    elif args.command == "canary": result = run_canary(key_pool=args.key_pool, key_name=args.key_name)
    elif args.command == "train": result = train(key_pool=args.key_pool, key_name=args.key_name, lock_dir=args.lock_dir)
    else: result = evaluate_heldout()
    print(json.dumps(result, indent=2, sort_keys=True))
if __name__ == "__main__": main()
