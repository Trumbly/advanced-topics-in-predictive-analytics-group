"""SVG chart renderer for experiment learning curves.

Zero-dependency, server-rendered line charts. Each chart is a self-
contained <svg>...</svg> that drops into a Jinja template without any
JS framework or CDN dependency.
"""
from __future__ import annotations

import math
from html import escape


def line_chart_svg(
    series: list[tuple[str, list[tuple[float, float]]]],
    *,
    width: int = 700,
    height: int = 240,
    y_label: str = "",
    x_label: str = "",
    palette: tuple[str, ...] = ("#4f9cff", "#ffb74d", "#66bb6a", "#ef5350", "#ce93d8"),
    background: str = "transparent",
) -> str:
    """Return an SVG line chart with grid, ticks, and annotations."""
    cleaned: list[tuple[str, list[tuple[float, float]]]] = []
    for name, pts in series:
        ok = [
            (float(x), float(y))
            for x, y in pts
            if isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and _finite(y)
        ]
        if ok:
            cleaned.append((name, ok))
    if not cleaned:
        return ""

    all_x = [x for _, pts in cleaned for x, _ in pts]
    all_y = [y for _, pts in cleaned for _, y in pts]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if x_max == x_min:
        x_max = x_min + 1.0
    y_pad = (y_max - y_min) * 0.06 or abs(y_min) * 0.1 or 0.1
    y_min -= y_pad
    y_max += y_pad

    pad_l, pad_r, pad_t, pad_b = 56, 14, 22, 34
    legend_h = 16 * len(cleaned) + 4
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    def sx(v: float) -> float:
        return pad_l + (v - x_min) / (x_max - x_min) * inner_w

    def sy(v: float) -> float:
        return pad_t + (1.0 - (v - y_min) / (y_max - y_min)) * inner_h

    svg = _Svg(width, height + legend_h, background)

    # Grid + Y ticks
    y_ticks = _nice_ticks(y_min, y_max, target=5)
    for yt in y_ticks:
        yy = sy(yt)
        if yy < pad_t - 1 or yy > pad_t + inner_h + 1:
            continue
        svg.line(pad_l, yy, pad_l + inner_w, yy, stroke="#333", width=0.5, dash="4,3")
        svg.text(pad_l - 6, yy + 3, _fmt(yt), anchor="end", fill="#888", size=10)

    # X ticks
    x_ticks = _nice_ticks(x_min, x_max, target=8)
    for xt in x_ticks:
        xx = sx(xt)
        if xx < pad_l - 1 or xx > pad_l + inner_w + 1:
            continue
        svg.line(xx, pad_t + inner_h, xx, pad_t + inner_h + 4, stroke="#555", width=0.5)
        svg.text(xx, pad_t + inner_h + 14, _fmt_x(xt), anchor="middle", fill="#888", size=10)

    # Axis border
    svg.rect(pad_l, pad_t, inner_w, inner_h, stroke="#555", width=0.7)

    # Axis labels
    if y_label:
        svg.text(10, pad_t - 6, y_label, fill="#999", size=10.5, weight=600)
    if x_label:
        svg.text(pad_l + inner_w / 2, height - 4, x_label, anchor="middle", fill="#999", size=10.5)

    # Series
    for idx, (name, pts) in enumerate(cleaned):
        color = palette[idx % len(palette)]
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
            for i, (x, y) in enumerate(pts)
        )
        svg.path(d, stroke=color, width=1.8)

        # Dots on all points (if few enough), else just first/last
        if len(pts) <= 30:
            for x, y in pts:
                svg.circle(sx(x), sy(y), r=2.2, fill=color)
        else:
            for x, y in [pts[0], pts[-1]]:
                svg.circle(sx(x), sy(y), r=2.5, fill=color)

        # Annotate min, max, last value directly on the curve
        if len(pts) >= 2:
            vals = [y for _, y in pts]
            min_idx = vals.index(min(vals))
            max_idx = vals.index(max(vals))
            last_idx = len(pts) - 1
            annotated: set[int] = set()
            for ai, label_prefix in [(max_idx, ""), (min_idx, ""), (last_idx, "")]:
                if ai in annotated:
                    continue
                annotated.add(ai)
                ax, ay = pts[ai]
                above = sy(ay) > pad_t + inner_h / 2
                ty = sy(ay) + (-8 if not above else 12)
                svg.text(
                    sx(ax), ty, _fmt(ay),
                    anchor="middle", fill=color, size=9, weight=600,
                    opacity=0.85,
                )

    # Legend below chart
    legend_y = height + 4
    for idx, (name, pts) in enumerate(cleaned):
        color = palette[idx % len(palette)]
        ly = legend_y + idx * 16
        svg.circle(pad_l + 6, ly + 5, r=4, fill=color)
        last_val = _fmt(pts[-1][1]) if pts else "—"
        first_val = _fmt(pts[0][1]) if pts else "—"
        label = f"{name}  {first_val} → {last_val}"
        svg.text(pad_l + 16, ly + 9, label, fill="#bbb", size=11)

    return svg.render()


def history_to_series(
    history: list[dict],
    metrics: tuple[str, ...],
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Extract ``(metric_name, [(epoch, value), ...])`` tuples from a
    per-epoch history list like the one the skeletons emit."""
    out: list[tuple[str, list[tuple[float, float]]]] = []
    for metric in metrics:
        pts: list[tuple[float, float]] = []
        for i, row in enumerate(history or [], start=1):
            if not isinstance(row, dict):
                continue
            epoch = row.get("epoch", i)
            value = row.get(metric)
            if isinstance(epoch, (int, float)) and isinstance(value, (int, float)):
                pts.append((float(epoch), float(value)))
        if pts:
            out.append((metric, pts))
    return out


# ---------------------------------------------------------------------------
# Internal SVG builder
# ---------------------------------------------------------------------------


class _Svg:
    """Tiny SVG builder — avoids string-soup in the main function."""

    def __init__(self, w: int, h: int, bg: str):
        self._parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {w} {h}" role="img" '
            f'style="background:{bg};font-family:ui-monospace,Consolas,'
            f"'Courier New',monospace;\">"
        ]

    def line(self, x1, y1, x2, y2, *, stroke="#444", width=0.5, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self._parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>'
        )

    def rect(self, x, y, w, h, *, stroke="#444", width=0.5, fill="none"):
        self._parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    def circle(self, cx, cy, *, r=2, fill="#fff"):
        self._parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"/>'
        )

    def path(self, d, *, stroke="#fff", width=1.5):
        self._parts.append(
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def text(
        self, x, y, content, *, anchor="start", fill="#aaa",
        size=10, weight=400, opacity=1.0,
    ):
        style_parts = [f"font-size:{size}px"]
        if weight != 400:
            style_parts.append(f"font-weight:{weight}")
        if opacity < 1.0:
            style_parts.append(f"opacity:{opacity}")
        style = ";".join(style_parts)
        self._parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'fill="{fill}" style="{style}">{escape(str(content))}</text>'
        )

    def render(self) -> str:
        self._parts.append("</svg>")
        return "".join(self._parts)


# ---------------------------------------------------------------------------
# Tick generation — "nice numbers" algorithm
# ---------------------------------------------------------------------------


def _nice_ticks(lo: float, hi: float, *, target: int = 5) -> list[float]:
    """Return a list of ~`target` evenly-spaced round tick values."""
    if hi <= lo:
        return [lo]
    raw_step = (hi - lo) / max(target - 1, 1)
    mag = 10 ** math.floor(math.log10(raw_step + 1e-30))
    residual = raw_step / mag
    if residual <= 1.5:
        nice = 1
    elif residual <= 3.5:
        nice = 2
    elif residual <= 7.5:
        nice = 5
    else:
        nice = 10
    step = nice * mag
    if step == 0:
        return [lo, hi]
    start = math.ceil(lo / step) * step
    ticks: list[float] = []
    v = start
    while v <= hi + step * 0.001:
        ticks.append(round(v, 10))
        v += step
    if not ticks:
        return [lo, hi]
    return ticks


def _finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    if abs(v) >= 0.01:
        return f"{v:.4f}"
    return f"{v:.2e}"


def _fmt_x(v: float) -> str:
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    return _fmt(v)


__all__ = ["line_chart_svg", "history_to_series"]
