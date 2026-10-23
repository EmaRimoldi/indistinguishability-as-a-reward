#!/usr/bin/env python3
"""Release-maintainer tool: record intentional source/input changes after review.

Do not run this to bypass a failing reproduction check. Compare/recover original
inputs first. Ordinary configure/train/evaluate commands never call this tool.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    originals = {}
    for inventory in (ROOT / "reproduction/provenance/sources").glob(
        "*.inventory.json"
    ):
        family = inventory.name.removesuffix(".inventory.json")
        for relative, sha in json.loads(inventory.read_text()).items():
            if relative.endswith(".py"):
                originals.setdefault(sha, []).append(
                    {"family": family, "file": relative}
                )
    source_map = {
        p.relative_to(ROOT).as_posix(): {
            "sha256": digest(p),
            "matches": originals.get(digest(p), []),
        }
        for p in sorted((ROOT / "src").rglob("*.py"))
        if "__pycache__" not in p.parts
        and not any(part.endswith(".egg-info") for part in p.parts)
        and not p.name.startswith("._")
    }
    (ROOT / "reproduction/provenance/source-map.json").write_text(
        json.dumps(source_map, indent=2) + "\n"
    )
    paths = []
    for folder in (
        "src",
        "configs",
        "scripts",
        "data",
        "reproduction",
    ):
        paths.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not any(part.endswith(".egg-info") for part in p.parts)
            and not p.name.startswith("._")
            and p.suffix not in (".pyc", ".pyo")
            and p.name not in ("package-integrity.json",)
        )
    paths.extend(ROOT / name for name in ("pyproject.toml", "uv.lock"))
    manifest = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}
    (ROOT / "reproduction/provenance/package-integrity.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps({"protected_files": len(manifest), "source_files": len(source_map)})
    )


if __name__ == "__main__":
    main()
