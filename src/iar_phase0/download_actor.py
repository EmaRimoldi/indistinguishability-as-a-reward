from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from huggingface_hub import snapshot_download

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an actor snapshot and record it without touching prior manifests."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    generator = config.raw["generator"]
    artifacts = config.raw.get("artifacts") or {}
    manifest_value = artifacts.get("actor_snapshot_manifest")
    if not manifest_value:
        raise ValueError("artifacts.actor_snapshot_manifest is required for immutable downloads")

    cache_dir = config.resolve(config.raw["paths"]["hf_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    manifest_path = config.resolve(manifest_value)
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing actor-snapshot manifest: {manifest_path}"
        )

    model_id = str(generator["model_id"])
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        local_files_only=bool(generator.get("local_files_only", False)),
    )
    manifest = {
        "schema_version": 1,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "cache_dir": str(cache_dir),
        "local_snapshot_path": local_path,
        "snapshot_revision": local_path.rstrip("/").split("/")[-1],
        "source_config": str(config.path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=False)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
