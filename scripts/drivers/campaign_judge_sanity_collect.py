#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iar_campaign.judge_sanity import collect_phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("teacher", "judge"), required=True)
    args = parser.parse_args()
    result = collect_phase(args.config.resolve(), args.phase)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.phase == "judge" and result["decision"] != "GO":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
