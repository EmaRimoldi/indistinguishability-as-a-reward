from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from huggingface_hub import snapshot_download

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    gen_cfg = config.raw["generator"]
    cache_dir = config.resolve(config.raw["paths"]["hf_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    model_id = gen_cfg["model_id"]
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        local_files_only=bool(gen_cfg.get("local_files_only", False)),
    )

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "cache_dir": str(cache_dir),
        "local_path": local_path,
    }
    out = config.resolve("models/manifests/downloaded_generator.json")
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

