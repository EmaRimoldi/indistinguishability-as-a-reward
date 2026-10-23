"""SVG back end: a preview of the PowerPoint slide (also editable in Inkscape).

Each scene element becomes a ``<g>``/shape whose ``id`` is the element name;
preset shapes are traced with the same geometry PowerPoint uses.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from xml.sax.saxutils import escape

import geometry
from icons import load_icon
from primitives import (
    WHITE,
    Box,
    Brace,
    Callout,
    Connector,
    Element,
    Ellipse,
    Group,
    Icon,
    PathShape,
    Scene,
    Segment,
    Stroke,
    Text,
    TextLine,
)

logger = logging.getLogger(__name__)

ASCENT, DESCENT = 0.905, 0.212  # Arial / Times New Roman, in em
LINE_PITCH = 1.2  # PowerPoint single line spacing, in em
SUB_SCALE, SUB_SHIFT = 0.66, 0.25  # subscript size and baseline offset


def _f(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")


def _d(segments: tuple[Segment, ...]) -> str:
    return " ".join(seg[0] + " ".join(_f(v) for v in seg[1:]) for seg in segments)


def _stroke(stroke: Stroke | None) -> str:
    if stroke is None:
        return 'stroke="none"'
    dash = f' stroke-dasharray="{_f(4 * stroke.width)} {_f(3 * stroke.width)}"' if stroke.dashed else ""
    return f'stroke="#{stroke.color}" stroke-width="{_f(stroke.width)}"{dash}'


def _fill(fill: str | None) -> str:
    return f'fill="#{fill}"' if fill else 'fill="none"'


def _lines(lines: tuple[TextLine, ...], x: float, y: float, w: float, h: float) -> str:
    """Centre ``lines`` in the box, mimicking PowerPoint's single spacing."""
    sizes = [max(run.size for run in line) for line in lines]
    top = y + (h - LINE_PITCH * sum(sizes)) / 2
    out = []
    for line, size in zip(lines, sizes):
        pitch = LINE_PITCH * size
        baseline = top + (pitch - (ASCENT + DESCENT) * size) / 2 + ASCENT * size
        spans, shift = [], 0.0
        for run in line:
            target = SUB_SHIFT * run.size if run.subscript else 0.0
            size_pt = run.size * (SUB_SCALE if run.subscript else 1)
            weight = ' font-weight="bold"' if run.bold else ""
            slant = ' font-style="italic"' if run.italic else ""
            spacing = f' letter-spacing="{_f(run.spacing)}"' if run.spacing else ""
            style = f'font-family="{run.font}" font-size="{_f(size_pt)}"{weight}{slant}{spacing}'
            spans.append(f'<tspan {style} dy="{_f(target - shift)}">{escape(run.text)}</tspan>')
            shift = target
        out.append(
            f'<text x="{_f(x + w / 2)}" y="{_f(baseline)}" text-anchor="middle" fill="#000000">'
            + "".join(spans) + "</text>"
        )
        top += pitch
    return "".join(out)


def _box(el: Box) -> str:
    rect = (
        f'<rect x="{_f(el.x)}" y="{_f(el.y)}" width="{_f(el.w)}" height="{_f(el.h)}" '
        f'rx="{_f(el.radius)}" {_fill(el.fill)} {_stroke(el.stroke)}/>'
    )
    if not el.lines:
        return rect
    left, top, right, bottom = el.inset
    label = _lines(el.lines, el.x + left, el.y + top, el.w - left - right, el.h - top - bottom)
    return f'<g id="{_id(el.name)}">{rect}{label}</g>'


def _text(el: Text) -> str:
    return f'<g id="{_id(el.name)}">{_lines(el.lines, el.x, el.y, el.w, el.h)}</g>'


def _ellipse(el: Ellipse) -> str:
    return (
        f'<ellipse id="{_id(el.name)}" cx="{_f(el.x + el.w / 2)}" cy="{_f(el.y + el.h / 2)}" '
        f'rx="{_f(el.w / 2)}" ry="{_f(el.h / 2)}" {_fill(el.fill)}/>'
    )


def _icon(el: Icon) -> str:
    icon = load_icon(el.key)
    scale = el.h / icon.height
    transform = f'transform="translate({_f(el.x)} {_f(el.y)}) scale({_f(scale)})"'
    d = _d(icon.segments)
    body = f'<path d="{d}" fill="#{el.fill}" fill-rule="evenodd"/>'
    if el.halo > 0:
        halo = (
            f'<path d="{d}" fill="#{WHITE}" stroke="#{WHITE}" '
            f'stroke-width="{_f(el.halo / scale)}" stroke-linejoin="round"/>'
        )
        body = halo + body
    return f'<g id="{_id(el.name)}" {transform}>{body}</g>'


def _arrowed(name: str, segments: tuple[Segment, ...], stroke: Stroke, arrow: bool) -> str:
    if not arrow or segments[-1][0] == "Z":  # no arrowheads on closed paths
        return f'<path id="{_id(name)}" d="{_d(segments)}" fill="none" {_stroke(stroke)}/>'
    tip, left, right = geometry.arrowhead(segments, stroke.width)
    line = geometry.trim_end(segments, 0.8 * geometry.ARROW_FACTOR * stroke.width)
    head = " ".join(f"{_f(x)},{_f(y)}" for x, y in (tip, left, right))
    return (
        f'<g id="{_id(name)}"><path d="{_d(line)}" fill="none" {_stroke(stroke)}/>'
        f'<polygon points="{head}" fill="#{stroke.color}"/></g>'
    )


def _connector(el: Connector) -> str:
    if el.kind == "curved":
        segs = geometry.curved_connector(el.x1, el.y1, el.x2, el.y2)
    else:
        segs = (("M", el.x1, el.y1), ("L", el.x2, el.y2))
    return _arrowed(el.name, segs, el.stroke, el.arrow)


def _path(el: PathShape) -> str:
    if el.stroke is not None and el.fill is None:
        return _arrowed(el.name, el.segments, el.stroke, el.arrow)
    return f'<path id="{_id(el.name)}" d="{_d(el.segments)}" {_fill(el.fill)} {_stroke(el.stroke)}/>'


def _brace(el: Brace) -> str:
    return f'<path id="{_id(el.name)}" d="{_d(geometry.brace(el))}" fill="none" {_stroke(el.stroke)}/>'


def _callout(el: Callout) -> str:
    outline = f'<path d="{_d(geometry.callout(el))}" {_fill(el.fill)} {_stroke(el.stroke)} stroke-linejoin="round"/>'
    return f'<g id="{_id(el.name)}">{outline}{_lines(el.lines, el.x, el.y, el.w, el.h)}</g>'


def _group(el: Group) -> str:
    return f'<g id="{_id(el.name)}">' + "".join(_render(child) for child in el.children) + "</g>"


_RENDERERS: dict[type, Callable[..., str]] = {
    Box: _box, Text: _text, Ellipse: _ellipse, Icon: _icon, Connector: _connector,
    PathShape: _path, Brace: _brace, Callout: _callout, Group: _group,
}


def _render(el: Element) -> str:
    return _RENDERERS[type(el)](el)


def write_svg(scene: Scene, path: Path) -> str:
    """Write the scene as SVG (1 user unit = 1 pt) and return the markup."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_f(scene.width)}pt" '
        f'height="{_f(scene.height)}pt" viewBox="0 0 {_f(scene.width)} {_f(scene.height)}">'
        + "".join(_render(el) for el in scene.elements) + "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")
    logger.info("Wrote %s", path)
    return svg


def render(svg: str, pdf: Path, png: Path, png_scale: float = 4.0) -> None:
    """Rasterise/convert the SVG with cairosvg when it is installed."""
    try:
        import cairosvg
    except ImportError:
        logger.warning("cairosvg not installed: skipping %s and %s", pdf.name, png.name)
        return
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png), scale=png_scale,
                     background_color="white")
    logger.info("Wrote %s and %s", pdf, png)
