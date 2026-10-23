#!/usr/bin/env python3
"""Check package Markdown links and local anchors without network access."""

from __future__ import annotations
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def anchors(path: Path) -> set[str]:
    counts = {}
    values = set()
    for title in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", path.read_text(), re.M):
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        values.add(f"{slug}-{count}" if count else slug)
    return values


def check() -> dict:
    paths = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "data/README.md",
        ROOT / "reproduction/provenance/README.md",
        ROOT / "reproduction/third_party/README.md",
    ]
    links, external = 0, set()
    errors = []
    for source in paths:
        # Ignore examples/code fences; validate actual Markdown destinations.
        text = re.sub(r"```.*?```", "", source.read_text(), flags=re.S)
        for raw in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            destination = raw.strip().split()[0].strip("<>")
            target = urlsplit(destination)
            links += 1
            if target.scheme in {"https", "http", "mailto"}:
                external.add(destination)
                continue
            if target.scheme:
                errors.append(
                    f"{source.relative_to(ROOT)}: unsupported link {destination}"
                )
                continue
            path = (
                (source.parent / unquote(target.path)).resolve()
                if target.path
                else source
            )
            if not path.exists():
                errors.append(f"{source.relative_to(ROOT)}: missing {destination}")
            elif (
                target.fragment
                and path.suffix == ".md"
                and unquote(target.fragment) not in anchors(path)
            ):
                errors.append(
                    f"{source.relative_to(ROOT)}: missing anchor {destination}"
                )
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "status": "passed",
        "documents": len(paths),
        "links": links,
        "external_links": sorted(external),
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
