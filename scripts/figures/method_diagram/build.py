# /// script
# requires-python = ">=3.10"
# dependencies = ["python-pptx>=1.0,<2", "svgelements>=1.9", "cairosvg>=2.7"]
# ///
"""Build the method overview diagram.

Outputs an editable PowerPoint slide (native shapes, one per element) and an
SVG/PDF/PNG rendering of the same scene.

Usage:
    uv run scripts/figures/method_diagram/build.py [--out DIR] [--stem NAME]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from layout import build_scene
from svg_backend import render, write_svg

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "method_diagram")
    parser.add_argument("--stem", default="method_diagram")
    parser.add_argument("--no-pptx", action="store_true", help="Generate only the paper figure assets")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scene = build_scene()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_pptx:
        from pptx_backend import write_pptx
        write_pptx(scene, args.out / f"{args.stem}.pptx")
    svg = write_svg(scene, args.out / f"{args.stem}.svg")
    render(svg, args.out / f"{args.stem}.pdf", args.out / f"{args.stem}.png")


if __name__ == "__main__":
    main()
