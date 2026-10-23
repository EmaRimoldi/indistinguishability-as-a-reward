"""Path geometry: PowerPoint preset shapes reproduced for the SVG preview.

The PowerPoint back end uses native presets (connectors, brace, callout); the
functions here trace the same outlines so that the preview matches them.
"""

from __future__ import annotations

import math

from primitives import Brace, Callout, Segment

KAPPA = 0.5522847498  # control-point factor of a cubic quarter ellipse
ARROW_FACTOR = 5.0  # PowerPoint "lg" arrowhead: 5x the line width, both ways

Point = tuple[float, float]


def quarter(a: Point, b: Point, corner: Point) -> Segment:
    """Quarter ellipse from ``a`` to ``b`` whose tangents meet at ``corner``."""
    (ax, ay), (bx, by), (qx, qy) = a, b, corner
    return (
        "C",
        ax + KAPPA * (qx - ax), ay + KAPPA * (qy - ay),
        bx + KAPPA * (qx - bx), by + KAPPA * (qy - by),
        bx, by,
    )


def curved_connector(x1: float, y1: float, x2: float, y2: float) -> tuple[Segment, ...]:
    """PowerPoint ``curvedConnector3`` (adj1 = 50%) from (x1, y1) to (x2, y2)."""
    dx, dy = x2 - x1, y2 - y1

    def at(fx: float, fy: float) -> Point:
        return x1 + fx * dx, y1 + fy * dy

    return (
        ("M", x1, y1),
        ("C", *at(0.25, 0.0), *at(0.5, 0.25), *at(0.5, 0.5)),
        ("C", *at(0.5, 0.75), *at(0.75, 1.0), *at(1.0, 1.0)),
    )


def brace(b: Brace) -> tuple[Segment, ...]:
    """PowerPoint ``rightBrace``: four quarter ellipses of radii (w/2, radius)."""
    x, y, w, h, r = b.x, b.y, b.w, b.h, b.radius
    hc, tip = x + w / 2, y + h / 2
    return (
        ("M", x, y),
        quarter((x, y), (hc, y + r), (hc, y)),
        ("L", hc, tip - r),
        quarter((hc, tip - r), (x + w, tip), (hc, tip)),
        quarter((x + w, tip), (hc, tip + r), (hc, tip)),
        ("L", hc, y + h - r),
        quarter((hc, y + h - r), (x, y + h), (hc, y + h)),
    )


def callout_adjustments(c: Callout) -> tuple[float, float, float]:
    """Adjustment values (fractions) of ``wedgeRoundRectCallout``."""
    dx = c.tip_x - (c.x + c.w / 2)
    dy = c.tip_y - (c.y + c.h / 2)
    return dx / c.w, dy / c.h, c.radius / min(c.w, c.h)


def callout(c: Callout) -> tuple[Segment, ...]:
    """Outline of ``wedgeRoundRectCallout``: the tail leaves the side facing the tip."""
    x, y, w, h, r = c.x, c.y, c.w, c.h, c.radius
    dx = c.tip_x - (x + w / 2)
    dy = c.tip_y - (y + h / 2)
    vertical = abs(dy) > abs(dx * h / w)
    xa, xb = (x + w * 7 / 12, x + w * 10 / 12) if dx > 0 else (x + w * 2 / 12, x + w * 5 / 12)
    ya, yb = (y + h * 7 / 12, y + h * 10 / 12) if dy > 0 else (y + h * 2 / 12, y + h * 5 / 12)
    tip = ("L", c.tip_x, c.tip_y)
    segs: list[Segment] = [("M", x, y + r), quarter((x, y + r), (x + r, y), (x, y))]
    if vertical and dy < 0:
        segs += [("L", xa, y), tip, ("L", xb, y)]
    segs += [("L", x + w - r, y), quarter((x + w - r, y), (x + w, y + r), (x + w, y))]
    if not vertical and dx > 0:
        segs += [("L", x + w, ya), tip, ("L", x + w, yb)]
    segs += [("L", x + w, y + h - r), quarter((x + w, y + h - r), (x + w - r, y + h), (x + w, y + h))]
    if vertical and dy > 0:
        segs += [("L", xb, y + h), tip, ("L", xa, y + h)]
    segs += [("L", x + r, y + h), quarter((x + r, y + h), (x, y + h - r), (x, y + h))]
    if not vertical and dx < 0:
        segs += [("L", x, yb), tip, ("L", x, ya)]
    segs.append(("Z",))
    return tuple(segs)


def _cubic(p: tuple[Point, Point, Point, Point], t: float) -> tuple[Point, Point, Point, Point]:
    """First half of a cubic split at ``t`` (de Casteljau)."""
    def lerp(a: Point, b: Point) -> Point:
        return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])

    p01, p12, p23 = lerp(p[0], p[1]), lerp(p[1], p[2]), lerp(p[2], p[3])
    p012, p123 = lerp(p01, p12), lerp(p12, p23)
    return p[0], p01, p012, lerp(p012, p123)


def end_direction(segments: tuple[Segment, ...]) -> tuple[Point, Point]:
    """End point of an open path and the unit tangent with which it arrives."""
    prev: Point = (0.0, 0.0)
    before: Point = (0.0, 0.0)
    for seg in segments:
        if seg[0] in ("M", "L"):
            before, prev = prev, (seg[1], seg[2])
        elif seg[0] == "C":
            ctrl = (seg[3], seg[4]) if (seg[3], seg[4]) != (seg[5], seg[6]) else (seg[1], seg[2])
            before, prev = ctrl, (seg[5], seg[6])
    ux, uy = prev[0] - before[0], prev[1] - before[1]
    norm = math.hypot(ux, uy) or 1.0
    return prev, (ux / norm, uy / norm)


def arrowhead(segments: tuple[Segment, ...], width: float) -> tuple[Point, Point, Point]:
    """Triangle of PowerPoint's large arrowhead at the end of ``segments``."""
    (tx, ty), (ux, uy) = end_direction(segments)
    length = ARROW_FACTOR * width
    bx, by = tx - length * ux, ty - length * uy
    half = length / 2
    return (tx, ty), (bx - half * uy, by + half * ux), (bx + half * uy, by - half * ux)


def trim_end(segments: tuple[Segment, ...], dist: float) -> tuple[Segment, ...]:
    """Shorten the last segment by ``dist`` so the stroke hides under the arrowhead."""
    *head, last = segments
    start: Point = (0.0, 0.0)
    for seg in head:
        if seg[0] != "Z":
            start = (seg[-2], seg[-1])
    if last[0] == "L":
        (ex, ey), (ux, uy) = end_direction(segments)
        return (*head, ("L", ex - dist * ux, ey - dist * uy))
    pts = (start, (last[1], last[2]), (last[3], last[4]), (last[5], last[6]))
    lo, hi = 0.0, 1.0
    for _ in range(40):  # bisection on the distance from the end point
        mid = (lo + hi) / 2
        px, py = _cubic(pts, mid)[3]
        if math.hypot(pts[3][0] - px, pts[3][1] - py) > dist:
            lo = mid
        else:
            hi = mid
    _, c1, c2, end = _cubic(pts, lo)
    return (*head, ("C", *c1, *c2, *end))
