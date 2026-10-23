from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from iar_phase0.parameters import initialize_parameter_record

from .data import write_json
from .prompts import protocol_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "parameters" / "manifest.json").exists():
        raise FileExistsError(f"Refusing to reinitialize immutable run: {run_dir}")
    shutil.copy2(config_path, run_dir / "requested_config.yaml")
    initialize_parameter_record(run_dir, resolved_config=config, source_config=config_path)
    shutil.copy2(args.split_manifest.resolve(), run_dir / "parameters" / "split_manifest.json")
    write_json(run_dir / "parameters" / "protocol_manifest.json", protocol_manifest())
    write_json(
        run_dir / "status.json",
        {"status": "initialized", "created_at": datetime.now(timezone.utc).isoformat()},
    )


if __name__ == "__main__":
    main()
