"""Scene primitives shared by the PowerPoint and SVG back ends.

Coordinates are in points (1/72 inch), origin at the top-left corner of the
figure, y growing downwards (the convention of both PowerPoint and SVG).
"""

from __future__ import annotations

from dataclasses import dataclass

BLACK = "000000"
WHITE = "FFFFFF"

# Path segments: ("M", x, y), ("L", x, y), ("C", x1, y1, x2, y2, x, y), ("Z",)
Segment = tuple


@dataclass(frozen=True)
class Stroke:
    """Outline style; width in points."""

    color: str = BLACK
    width: float = 0.8
    dashed: bool = False


@dataclass(frozen=True)
class Run:
    """A span of text with uniform formatting."""

    text: str
    size: float
    bold: bool = False
    italic: bool = False
    font: str = "Arial"
    subscript: bool = False
    spacing: float = 0.0  # extra space after each character, in points


TextLine = tuple[Run, ...]


@dataclass(frozen=True)
class Text:
    """Free text box; lines are centred horizontally and vertically."""

    name: str
    x: float
    y: float
    w: float
    h: float
    lines: tuple[TextLine, ...]
    color: str = BLACK


@dataclass(frozen=True)
class Box:
    """Rectangle (rounded when radius > 0) with an optional centred label.

    ``inset`` (left, top, right, bottom) shrinks the label area, e.g. to leave
    room for an icon drawn on top of the box.
    """

    name: str
    x: float
    y: float
    w: float
    h: float
    radius: float = 0.0
    stroke: Stroke | None = Stroke()
    fill: str | None = WHITE
    lines: tuple[TextLine, ...] = ()
    inset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Ellipse:
    name: str
    x: float
    y: float
    w: float
    h: float
    fill: str = BLACK


@dataclass(frozen=True)
class Icon:
    """Vector icon scaled to height ``h``; ``halo`` adds a white knockout."""

    name: str
    key: str
    x: float
    y: float
    h: float
    fill: str = BLACK
    halo: float = 0.0


@dataclass(frozen=True)
class Connector:
    """Native connector ("straight" or "curved" S-shape) with arrowhead."""

    name: str
    kind: str
    x1: float
    y1: float
    x2: float
    y2: float
    stroke: Stroke = Stroke(width=0.9)
    arrow: bool = True


@dataclass(frozen=True)
class PathShape:
    """Free-form path in absolute coordinates (edit points in PowerPoint)."""

    name: str
    segments: tuple[Segment, ...]
    stroke: Stroke | None = Stroke()
    fill: str | None = None
    arrow: bool = False


@dataclass(frozen=True)
class Brace:
    """Right curly brace; ``radius`` is the vertical radius of its arcs."""

    name: str
    x: float
    y: float
    w: float
    h: float
    radius: float
    stroke: Stroke = Stroke()


@dataclass(frozen=True)
class Callout:
    """Rounded speech bubble whose tail points at (tip_x, tip_y)."""

    name: str
    x: float
    y: float
    w: float
    h: float
    tip_x: float
    tip_y: float
    radius: float
    lines: tuple[TextLine, ...]
    stroke: Stroke = Stroke(width=0.7)
    fill: str = WHITE


@dataclass(frozen=True)
class Group:
    name: str
    children: tuple[Element, ...]


Element = Text | Box | Ellipse | Icon | Connector | PathShape | Brace | Callout | Group


@dataclass(frozen=True)
class Scene:
    width: float
    height: float
    elements: tuple[Element, ...]
    notes: str = ""
