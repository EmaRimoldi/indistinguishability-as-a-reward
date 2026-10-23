#!/usr/bin/env python3
"""Save local asset locations and generate configs without changing frozen inputs."""
import argparse
import json
from iar_repro.paths import ROOT, defaults
from iar_repro.verify import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", help="Hugging Face hub cache, containing models--Qwen--Qwen2.5-1.5B-Instruct")
    p.add_argument("--spider-root", help="Unpacked official release containing spider_data/")
    p.add_argument("--runs", help="Generated artifacts and experiment outputs")
    args = p.parse_args()
    loc = defaults()
    from pathlib import Path
    for key in ("cache", "spider", "runs"):
        value = getattr(args, key if key != "spider" else "spider_root")
        if value:
            loc[key] = Path(value).expanduser().resolve()
    (ROOT / ".reproduction-paths.json").write_text(json.dumps({k: str(v) if v else None for k, v in loc.items()}, indent=2) + "\n")
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
