from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
SEEDS = (23, 42, 7)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def input_index() -> list[dict]:
    return json.loads(
        (ROOT / "reproduction/provenance/input-manifest.json").read_text()
    )


def resolve(value: str, *, runs: Path, cache: Path, spider: Path | None) -> str:
    """Relocate paths at runtime without editing hash-verified historical records."""
    for item in input_index():
        if value in (item["reference"], item["remote_path"]):
            return str(ROOT / "reproduction/provenance/protocols" / item["relative"])
    match = re.search(
        r"/datasets/(commonsenseqa|gsm8k|kodcode|spider)/(?:frozen_historical_seed23|frozen_seed23_v1)/(?:sealed/)?([^/]+\.jsonl)$",
        value,
    )
    if match:
        domain, name = match.groups()
        return str(ROOT / "data" / domain / name.replace("sealed_test1000", "test1000"))
    if value.endswith("/spider_official_test1034.jsonl"):
        return str(ROOT / "data/spider/test1034.jsonl")
    if "/datasets/spider-release-" in value:
        suffix = re.split(r"/datasets/spider-release-[^/]+", value, maxsplit=1)[1]
        return str((spider or ROOT / "external/spider") / suffix.lstrip("/"))
    prefixes = (
        ("/path/to/iar_code", ROOT),
        ("/home/erimoldi/projects/indistinguishability-as-a-reward", ROOT),
        ("/path/to/runs", runs),
        ("/orcd/pool/008/erimoldi/iar/experiments/campaign_20260908_v2", runs),
        ("/path/to/cache/huggingface", cache),
        ("/orcd/pool/008/erimoldi/iar/cache/huggingface", cache),
    )
    for prefix, destination in prefixes:
        if value == prefix or value.startswith(prefix + "/"):
            return str(destination / value[len(prefix) :].lstrip("/"))
    return value


def relocate(value: Any, *, runs: Path, cache: Path, spider: Path | None) -> Any:
    if isinstance(value, dict):
        resolved = {
            k: relocate(v, runs=runs, cache=cache, spider=spider)
            for k, v in value.items()
        }
        for key, original_path in value.items():
            if not isinstance(original_path, str) or key.endswith("_sha256"):
                continue
            for record in input_index():
                if original_path in (record["reference"], record["remote_path"]):
                    digest_key = key + "_sha256"
                    if digest_key in value:
                        if value[digest_key] != record["original_sha256"]:
                            raise ValueError(
                                f"Historical input identity changed: {key}"
                            )
                        resolved[digest_key] = record["sha256"]
                    break
        return resolved
    if isinstance(value, list):
        return [relocate(v, runs=runs, cache=cache, spider=spider) for v in value]
    if isinstance(value, str):
        return resolve(value, runs=runs, cache=cache, spider=spider)
    return value


def defaults() -> dict:
    settings = ROOT / ".reproduction-paths.json"
    saved = json.loads(settings.read_text()) if settings.is_file() else {}
    return {
        "runs": Path(os.environ.get("IAR_RUNS_ROOT", saved.get("runs", ROOT / "runs")))
        .expanduser()
        .resolve(),
        "cache": Path(
            os.environ.get(
                "IAR_HF_CACHE", saved.get("cache", ROOT / "cache/huggingface")
            )
        )
        .expanduser()
        .resolve(),
        "spider": Path(os.environ.get("IAR_SPIDER_ROOT", saved.get("spider")))
        .expanduser()
        .resolve()
        if os.environ.get("IAR_SPIDER_ROOT", saved.get("spider"))
        else None,
    }


def check_hashed_paths(value: dict) -> list[str]:
    checked = []
    for key, item in value.items():
        if isinstance(item, dict):
            checked.extend(check_hashed_paths(item))
        if key.endswith("_sha256") and isinstance(item, str):
            other = value.get(key[:-7])
            if isinstance(other, str) and other.startswith("/"):
                path = Path(other)
                if not path.is_file() or sha256(path) != item:
                    raise ValueError(f"Missing or drifted input {key[:-7]}: {path}")
                checked.append(str(path))
    return checked
