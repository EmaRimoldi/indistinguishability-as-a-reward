from __future__ import annotations

import gzip
import json
import os
from functools import lru_cache
from pathlib import Path

from .paths import ROOT, defaults, sha256


def results_root() -> Path:
    """Experimental records are local/external inputs, never packaged outputs."""
    return (
        Path(os.environ.get("IAR_RESULTS_ROOT", ROOT / "external/results"))
        .expanduser()
        .resolve()
    )


def result_path(relative: str) -> Path:
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts or name.parts[:1] != ("results",):
        raise ValueError("Result paths must be relative to results/")
    path = results_root() / name.relative_to("results")
    if not path.is_file():
        raise FileNotFoundError(
            f"External experimental input missing: {path}. "
            "Set IAR_RESULTS_ROOT to your local result collection; results are not shipped in this repository."
        )
    return path


def manifest() -> list[dict]:
    path = results_root() / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"External result manifest missing: {path}. Set IAR_RESULTS_ROOT; "
            "training, configuration and smoke tests do not require recorded results."
        )
    return json.loads(path.read_text())


def materialize(relative: str) -> Path:
    """Verify an external compressed artifact, then expand a local working copy."""
    entries = {r["local"]: r for r in manifest()}
    entry = entries[relative]
    source = result_path(relative)
    if sha256(source) != entry["package_sha256"]:
        raise ValueError(f"External numerical record drifted: {source}")
    destination = (
        defaults()["runs"] / "offline_records" / Path(relative).relative_to("results")
    )
    destination = destination.with_suffix("")
    raw = gzip.decompress(source.read_bytes())
    if not destination.is_file() or destination.read_bytes() != raw:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return destination


def log_paths(condition: str, domain: str, seed: str) -> list[str]:
    prefix = f"results/training_logs/{condition}/{domain}/{seed}/"
    return [
        str(materialize(r["local"]))
        for r in manifest()
        if r["local"].startswith(prefix)
    ]


def evaluation_files() -> dict[str, dict[str, str]]:
    files: dict[str, dict[str, str]] = {}
    for entry in manifest():
        if entry.get("kind") == "heldout":
            files.setdefault(entry["domain"], {})[entry["condition"]] = str(
                materialize(entry["local"])
            )
    return files


def analysis_inputs() -> Path:
    directory = None
    for entry in manifest():
        if entry["local"].startswith("results/alignment/"):
            path = materialize(entry["local"])
            directory = path.parent
    if directory is None:
        raise FileNotFoundError("No frozen within-prompt analysis rows in this package")
    return directory


@lru_cache(maxsize=1)
def reanalysis_inputs() -> Path:
    for entry in manifest():
        if entry["local"].startswith("results/reanalysis/"):
            materialize(entry["local"])
    return defaults()["runs"] / "offline_records/reanalysis"


def reward_log(condition: str, domain: str, seed: str) -> Path:
    if condition != "grpo_identity":
        raise ValueError(
            "Only Indist. response-level logs are indexed; baseline trainer metrics are separate"
        )
    return reanalysis_inputs() / "rewards" / domain / seed / "reward_calls.jsonl"
