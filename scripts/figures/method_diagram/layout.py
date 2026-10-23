"""Layout of the method overview figure.

Instance -> Student (actor) -> rollouts A1..A4 -> Teacher (distinguisher)
-> per-rollout log-probability -> rewards r1..r4 -> GRPO -> update of the
student. Sizes are in points for a 5.5 in (ICLR text width) figure.
"""

from __future__ import annotations

import math

from geometry import quarter
from icons import icon_width
from primitives import (
    Box,
    Brace,
    Callout,
    Connector,
    Element,
    Ellipse,
    Group,
    Icon,
    PathShape,
    Run,
    Scene,
    Stroke,
    Text,
    TextLine,
)

WIDTH, HEIGHT = 396.0, 121.0  # 5.5 in x 1.68 in
YC = 56.0  # main flow axis
TITLE_PT, MATH_PT = 8.0, 9.0
ARROW = Stroke(width=0.9)
OUTLINE = Stroke(width=0.8)
THIN = Stroke(width=0.6)
FRAME = Stroke(color="808080", width=0.6, dashed=True)
BAR_FILL = "BFBFBF"
GAP = 1.5  # clearance between arrow tips and icons

INST_X, INST_W, INST_H = 2.0, 34.0, 28.0
STU_X, STU_H = 54.0, 28.0
ROLL_X, ROLL_W, ROLL_H = 104.0, 30.0, 16.0
ROLL_DY = (-33.0, -11.0, 11.0, 33.0)
BRACE_X, BRACE_W = 136.5, 6.5
TEA_X, TEA_H = 159.0, 30.0
LP_X, LP_W, LP_TOP, LP_BOT = 248.0, 70.0, YC - 53.0, YC + 46.0
ROW_DY = 29.0  # rows of A1 / A4 and r1 / r4 above and below the axis
PLOT_X, PLOT_W, PLOT_H = LP_X + 19.0, 44.0, 13.0
R_X, R_W, R_H = 331.0, 20.0, 15.0
G_X, G_W, G_H = 363.0, 31.0, 22.0
LOOP_Y = YC + 52.0

# Two-component Gaussian mixtures (weight, mean, sd) for the mini plots.
MIX_LOW = ((0.72, 0.2, 0.07), (0.28, 0.62, 0.09))  # mostly "not mine"
MIX_HIGH = ((0.28, 0.36, 0.08), (0.72, 0.8, 0.07))  # mostly "mine"

NOTES = (
    "Method overview, 5.5 in wide (ICLR text width). Every element is a native, "
    "named shape (see Home > Arrange > Selection Pane). Icons are freeforms "
    "(right-click > Edit Points); arrows are connectors (drag their ends); the "
    "brace and the speech bubble have yellow handles. Export with File > Export "
    "> PDF. Icons: Font Awesome Free 6.5.2 (user, file-lines, chalkboard-user, "
    "graduation-cap), CC BY 4.0, https://fontawesome.com/license/free."
)


def _bold(text: str, size: float = TITLE_PT) -> TextLine:
    return (Run(text, size, bold=True),)


def _math(symbol: str, index: str, size: float = MATH_PT) -> TextLine:
    """Italic symbol with an upright subscript, e.g. A_1 (tightened like TeX)."""
    return (
        Run(symbol, size, italic=True, font="Times New Roman", spacing=-0.35),
        Run(index, size, font="Times New Roman", subscript=True),
    )


def _title(name: str, cx: float, bottom: float, *texts: str) -> Text:
    h = len(texts) * TITLE_PT * 1.2
    return Text(name, cx - 45.0, bottom - h, 90.0, h, tuple(_bold(t) for t in texts))


def _vdots(name: str, cx: float, cy: float, d: float = 1.8, step: float = 4.2) -> Group:
    return Group(name, tuple(
        Ellipse(f"{name} {k + 1}", cx - d / 2, cy + (k - 1) * step - d / 2, d, d)
        for k in range(3)
    ))


def _instance() -> Group:
    y = YC - INST_H / 2
    ih = 17.0
    iw = icon_width("file", ih)
    return Group("Instance", (
        _title("Instance title", INST_X + INST_W / 2, y - 3.0, "Instance"),
        Box("Instance box", INST_X, y, INST_W, INST_H, radius=4.0, stroke=OUTLINE),
        Icon("Instance icon", "file", INST_X + (INST_W - iw) / 2, YC - ih / 2, ih),
    ))


def _student() -> Group:
    w = icon_width("user", STU_H)
    return Group("Student", (
        _title("Student title", STU_X + w / 2, YC - STU_H / 2 - 3.0, "Student", "(Actor)"),
        Icon("Student icon", "user", STU_X, YC - STU_H / 2, STU_H),
    ))


def _rollouts() -> Group:
    ih = 10.5
    iw = icon_width("file", ih)
    items: list[Element] = []
    for i, dy in enumerate(ROLL_DY, start=1):
        items.append(Group(f"Rollout A{i}", (
            Box(f"Rollout A{i} box", ROLL_X, YC + dy - ROLL_H / 2, ROLL_W, ROLL_H,
                radius=3.0, stroke=THIN, lines=(_math("A", str(i)),),
                inset=(iw + 4.5, 0.0, 1.0, 0.0)),
            Icon(f"Rollout A{i} icon", "file", ROLL_X + 3.5, YC + dy - ih / 2, ih),
        )))
    top = YC + ROLL_DY[0] - ROLL_H / 2
    items.append(Brace("Rollout brace", BRACE_X, top, BRACE_W, 2 * (YC - top), 4.0, THIN))
    return Group("Rollouts", tuple(items))


def _fan_out(x_start: float) -> Group:
    return Group("Fan-out arrows", tuple(
        Connector(f"Arrow to A{i}", "curved", x_start, YC, ROLL_X - 0.4, YC + dy, ARROW)
        for i, dy in enumerate(ROLL_DY, start=1)
    ))


def _cap_box() -> tuple[float, float, float, float]:
    """Graduation cap sitting on the lower-right corner of the chalkboard."""
    h = 12.0
    w = icon_width("cap", h)
    cx, cy = TEA_X + icon_width("teacher", TEA_H) - 6.5, YC + 6.5
    return cx - w / 2, cy - h / 2, w, h


def _teacher() -> Group:
    w = icon_width("teacher", TEA_H)
    cap_x, cap_y, _, cap_h = _cap_box()
    return Group("Teacher", (
        _title("Teacher title", TEA_X + w / 2, YC - TEA_H / 2 - 3.0, "Teacher", "(Distinguisher)"),
        Icon("Teacher icon", "teacher", TEA_X, YC - TEA_H / 2, TEA_H),
        Icon("Graduation cap", "cap", cap_x, cap_y, cap_h, halo=1.4),
        Callout("Speech bubble", 209.0, 34.0, 35.0, 17.0, TEA_X + w + 2.5, 47.5, 3.0,
                ((Run("Mine or", 6.5),), (Run("not mine?", 6.5),))),
    ))


def _density(x: float, mix: tuple[tuple[float, float, float], ...]) -> float:
    return sum(w * math.exp(-0.5 * ((x - m) / s) ** 2) / s for w, m, s in mix)


def _smooth(points: list[tuple[float, float]]) -> tuple:
    """Catmull-Rom spline through ``points`` as cubic segments."""
    pts = [points[0], *points, points[-1]]
    segs: list = [("M", *points[0])]
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        segs.append((
            "C",
            p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6,
            p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6,
            *p2,
        ))
    return tuple(segs)


def _plot_row(idx: str, yr: float, mix: tuple[tuple[float, float, float], ...]) -> Group:
    base = yr + PLOT_H / 2
    peak = max(_density(k / 200, mix) for k in range(201))

    def height(u: float) -> float:
        return 0.95 * PLOT_H * _density(u, mix) / peak

    bins, steps = 12, 8
    bw = PLOT_W / bins
    heights = [sum(height((k + (j + 0.5) / steps) / bins) for j in range(steps)) / steps
               for k in range(bins)]  # bin averages, as in a histogram
    bars = tuple(
        Box(f"A{idx} bar {k + 1}", PLOT_X + k * bw + 0.3, base - hk, bw - 0.6, hk,
            stroke=None, fill=BAR_FILL)
        for k, hk in enumerate(heights) if hk > 0.15
    )
    curve = [(PLOT_X + PLOT_W * k / 32, base - height(k / 32)) for k in range(33)]
    tick = 5.5
    return Group(f"A{idx} distribution", (
        Text(f"A{idx} label", PLOT_X - 17.0, yr - 6.0, 15.0, 12.0, (_math("A", idx),)),
        *bars,
        PathShape(f"A{idx} density", _smooth(curve), stroke=Stroke(width=0.7)),
        Connector(f"A{idx} axis", "straight", PLOT_X, base, PLOT_X + PLOT_W, base,
                  Stroke(width=0.5), arrow=False),
        Text(f"A{idx} tick 0", PLOT_X - 4.0, base + 0.5, 8.0, 7.0, ((Run("0", tick),),)),
        Text(f"A{idx} tick 1", PLOT_X + PLOT_W - 4.0, base + 0.5, 8.0, 7.0, ((Run("1", tick),),)),
    ))


def _logprob() -> Group:
    return Group("Log-probability", (
        Box("Log-probability frame", LP_X, LP_TOP, LP_W, LP_BOT - LP_TOP, stroke=FRAME, fill=None),
        Text("Log-probability title", LP_X, LP_TOP + 2.0, LP_W, 10.0, (_bold("Log-probability"),)),
        _plot_row("1", YC - ROW_DY, MIX_LOW),
        _vdots("Rollout ellipsis", PLOT_X + PLOT_W / 2, YC),
        _plot_row("4", YC + ROW_DY, MIX_HIGH),
    ))


def _rewards() -> Group:
    items: list[Element] = []
    for idx, sign in (("1", -1), ("4", 1)):
        yr = YC + sign * ROW_DY
        items.append(Connector(f"Arrow to r{idx}", "straight", LP_X + LP_W + GAP, yr,
                               R_X - 0.4, yr, ARROW))
        items.append(Box(f"Reward r{idx}", R_X, yr - R_H / 2, R_W, R_H, radius=3.0,
                         stroke=OUTLINE, lines=(_math("r", idx),)))
    items.append(_vdots("Reward ellipsis", R_X + R_W / 2, YC))
    return Group("Rewards", tuple(items))


def _grpo() -> Group:
    merges = tuple(
        Connector(f"r{idx} to GRPO", "curved", R_X + R_W, YC + sign * ROW_DY,
                  G_X - 0.4, YC + sign * 4.0, ARROW)
        for idx, sign in (("1", -1), ("4", 1))
    )
    box = Box("GRPO box", G_X, YC - G_H / 2, G_W, G_H, radius=3.0, stroke=OUTLINE,
              lines=(_bold("GRPO", 8.5),))
    return Group("GRPO", (*merges, box))


def _update(student_cx: float) -> Group:
    gx, r = G_X + G_W / 2, 4.0
    segs = (
        ("M", gx, YC + G_H / 2),
        ("L", gx, LOOP_Y - r),
        quarter((gx, LOOP_Y - r), (gx - r, LOOP_Y), (gx, LOOP_Y)),
        ("L", student_cx + r, LOOP_Y),
        quarter((student_cx + r, LOOP_Y), (student_cx, LOOP_Y - r), (student_cx, LOOP_Y)),
        ("L", student_cx, YC + STU_H / 2 + GAP),
    )
    mid = (gx + student_cx) / 2
    return Group("Update loop", (
        PathShape("Update arrow", segs, stroke=ARROW, arrow=True),
        Text("Update label", mid - 30.0, LOOP_Y + 1.0, 60.0, 10.0, (_bold("Update"),)),
    ))


def build_scene() -> Scene:
    """Assemble the whole figure."""
    stu_w = icon_width("user", STU_H)
    cap_x, _, cap_w, _ = _cap_box()
    tea_right = max(TEA_X + icon_width("teacher", TEA_H), cap_x + cap_w)
    elements: tuple[Element, ...] = (
        _instance(),
        Connector("Instance to student", "straight", INST_X + INST_W + GAP, YC,
                  STU_X - GAP, YC, ARROW),
        _student(),
        _fan_out(STU_X + stu_w + GAP),
        _rollouts(),
        Connector("Rollouts to teacher", "straight", BRACE_X + BRACE_W + GAP, YC,
                  TEA_X - GAP, YC, ARROW),
        _teacher(),
        Connector("Teacher to log-probability", "straight", tea_right + GAP, YC,
                  LP_X - 0.4, YC, ARROW),
        _logprob(),
        _rewards(),
        _grpo(),
        _update(STU_X + stu_w / 2),
    )
    return Scene(WIDTH, HEIGHT, elements, NOTES)
