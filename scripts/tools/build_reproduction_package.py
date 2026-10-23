#!/usr/bin/env python3
"""Build or verify a deterministic source/data reproduction ZIP (no secrets or run state)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIRS = {
    ".github",
    "configs",
    "data",
    "docs",
    "reproduction",
    "scripts",
    "src",
    "tests",
}
PACKAGE_FILES = {".gitignore", "README.md", "CITATION.cff", "pyproject.toml", "uv.lock"}
SKIP_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "runs",
    "results",
    "cache",
    "external",
    "dist",
    "node_modules",
    "artifacts",
    "checkpoints",
    "experiments",
    "logs",
    "output",
    ".analysis_work",
}


def files():
    for folder, directories, names in os.walk(ROOT):
        blocked = SKIP_DIRS
        directories[:] = sorted(
            d
            for d in directories
            if d not in blocked
            and not d.startswith((".venv-", ".codex"))
            and not d.endswith(".egg-info")
        )
        for name in sorted(names):
            path = Path(folder) / name
            rel = path.relative_to(ROOT)
            if rel.parts[0] not in PACKAGE_DIRS and rel.as_posix() not in PACKAGE_FILES:
                continue
            if (
                path.is_symlink()
                or name
                in {".DS_Store", ".reproduction-paths.json", "PACKAGE_MANIFEST.json"}
                or name.startswith("._")
            ):
                continue
            if name.startswith(".env") and not name.endswith(".example"):
                continue
            if path.suffix in {
                ".pyc",
                ".pyo",
                ".env",
                ".pdf",
                ".png",
                ".pt",
                ".bin",
                ".safetensors",
            }:
                continue
            yield rel, path


def check(path):
    with zipfile.ZipFile(path) as z:
        listed = json.loads(z.read("iar-reproduction/PACKAGE_MANIFEST.json"))
        expected = {"iar-reproduction/" + name for name in listed} | {
            "iar-reproduction/PACKAGE_MANIFEST.json"
        }
        if set(z.namelist()) != expected or len(z.namelist()) != len(expected):
            raise ValueError("Unexpected or duplicate archive members")
        for name, digest in listed.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe archive member")
            if hashlib.sha256(z.read("iar-reproduction/" + name)).hexdigest() != digest:
                raise ValueError(f"Archive checksum mismatch: {name}")
        return {
            "status": "passed",
            "files": len(listed),
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def build(output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:

        def write(relative, data):
            item = zipfile.ZipInfo(
                "iar-reproduction/" + relative, date_time=(2026, 10, 22, 0, 0, 0)
            )
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, data)

        for relative, path in sorted(files()):
            if path.resolve() == output:
                continue
            data = path.read_bytes()
            records[relative.as_posix()] = hashlib.sha256(data).hexdigest()
            write(relative.as_posix(), data)
        write("PACKAGE_MANIFEST.json", (json.dumps(records, indent=2) + "\n").encode())
    result = check(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{result['archive_sha256']}  {output.name}\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist/iar-reproduction-source-2026-10-22.zip",
    )
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(check(args.verify) if args.verify else build(args.output), indent=2)
    )
