"""Vector icons: Font Awesome Free 6.5.2 paths flattened to cubic segments.

Font Awesome Free icons are released under CC BY 4.0
(https://fontawesome.com/license/free); the vendored SVG files keep the
original attribution comment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from primitives import Segment
from svgelements import Arc, Close, CubicBezier, Line, Move, QuadraticBezier
from svgelements import Path as SvgPath

ICON_DIR = Path(__file__).resolve().parent / "icons"
ICON_FILES = {
    "user": "fa6-solid-user.svg",
    "file": "fa6-regular-file-lines.svg",
    "teacher": "fa6-solid-chalkboard-user.svg",
    "cap": "fa6-solid-graduation-cap.svg",
}


@dataclass(frozen=True)
class IconPath:
    """Icon outline in icon units; its tight bounding box starts at (0, 0)."""

    segments: tuple[Segment, ...]
    width: float
    height: float


@cache
def load_icon(key: str) -> IconPath:
    """Parse the icon's path data and convert arcs and quadratics to cubics."""
    source = ICON_DIR / ICON_FILES[key]
    paths = re.findall(r'\sd="([^"]+)"', source.read_text(encoding="utf-8"))
    if not paths:
        raise ValueError(f"No path data in {source}")
    path = SvgPath(" ".join(paths))  # all <path> elements of the icon
    x0, y0, x1, y1 = path.bbox()

    def pt(p) -> tuple[float, float]:
        return p.x - x0, p.y - y0

    segs: list[Segment] = []
    for seg in path:
        if isinstance(seg, Move):
            segs.append(("M", *pt(seg.end)))
        elif isinstance(seg, Close):
            segs.append(("Z",))
        elif isinstance(seg, Line):
            segs.append(("L", *pt(seg.end)))
        elif isinstance(seg, CubicBezier):
            segs.append(("C", *pt(seg.control1), *pt(seg.control2), *pt(seg.end)))
        elif isinstance(seg, QuadraticBezier):
            (sx, sy), (qx, qy), (ex, ey) = pt(seg.start), pt(seg.control), pt(seg.end)
            segs.append((
                "C",
                sx + 2 / 3 * (qx - sx), sy + 2 / 3 * (qy - sy),
                ex + 2 / 3 * (qx - ex), ey + 2 / 3 * (qy - ey),
                ex, ey,
            ))
        elif isinstance(seg, Arc):
            for cub in seg.as_cubic_curves():
                segs.append(("C", *pt(cub.control1), *pt(cub.control2), *pt(cub.end)))
        else:
            raise TypeError(f"Unsupported segment {type(seg).__name__} in {source.name}")
    return IconPath(tuple(segs), x1 - x0, y1 - y0)


def icon_width(key: str, height: float) -> float:
    """Width of the icon when drawn at ``height`` points."""
    icon = load_icon(key)
    return height * icon.width / icon.height
