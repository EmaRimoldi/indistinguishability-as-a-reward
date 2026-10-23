from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def initialize_parameter_record(
    run_dir: Path,
    *,
    resolved_config: dict[str, Any],
    source_config: Path | None = None,
    backfilled: bool = False,
) -> Path:
    """Create the mandatory, secret-free parameter record for an experiment."""
    parameters_dir = run_dir / "parameters"
    parameters_dir.mkdir(parents=True, exist_ok=True)

    source_hash = None
    if source_config is not None and source_config.is_file():
        destination = parameters_dir / "requested_config.yaml"
        shutil.copy2(source_config, destination)
        source_hash = hashlib.sha256(destination.read_bytes()).hexdigest()

    payload = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "backfilled": backfilled,
        "resolved_config": resolved_config,
    }
    _write_json(parameters_dir / "resolved_parameters.json", payload)
    _write_json(
        parameters_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_dir.name,
            "requested_config": "requested_config.yaml" if source_hash else None,
            "requested_config_sha256": source_hash,
            "resolved_parameters": "resolved_parameters.json",
            "python_version": sys.version.split()[0],
            "contains_secrets": False,
        },
    )
    return parameters_dir


def write_parameter_artifact(run_dir: Path, filename: str, values: Any) -> Path:
    """Persist effective library arguments after defaults have been resolved."""
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("Parameter artifacts must be plain JSON filenames")
    destination = run_dir / "parameters" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, values)
    return destination


def _write_json(path: Path, values: Any) -> None:
    path.write_text(
        json.dumps(values, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
