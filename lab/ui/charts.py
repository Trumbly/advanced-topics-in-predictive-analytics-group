"""Tiny SVG chart renderer.

Zero-dependency line charts for experiment histories. Keeping charts
server-rendered means no JS/CDN; the markup is a single <svg>...</svg>
string that drops straight into a Jinja template.
"""
from __future__ import annotations

from html import escape


def line_chart_svg(
    series: list[tuple[str, list[tuple[float, float]]]],
    *,
    width: int = 640,
    height: int = 200,
    y_label: str = "",
    palette: tuple[str, ...] = ("#4f9cff", "#ffb74d", "#66bb6a", "#ef5350"),
    background: str = "transparent",
) -> str:
    """Return an SVG line chart for one or more named series.

    ``series`` is ``[(name, [(x, y), ...])]``. Points with non-finite
    values are filtered per series so a single ``None`` doesn't break
    the whole curve. Returns an empty string if nothing is plottable
    so the caller can skip rendering the figure container.
    """
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
    if y_max == y_min:
        y_max = y_min + (abs(y_min) * 0.1 or 1.0)

    pad_l, pad_r, pad_t, pad_b = 40, 12, 14, 26
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    def sx(v: float) -> float:
        return pad_l + (v - x_min) / (x_max - x_min) * inner_w

    def sy(v: float) -> float:
        return pad_t + (1.0 - (v - y_min) / (y_max - y_min)) * inner_h

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(y_label) or "chart"}" '
        f'style="background:{background};font-family:ui-monospace,Consolas,monospace;'
        'font-size:10px;">'
    )
    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{inner_w:.1f}" height="{inner_h:.1f}" '
        'fill="none" stroke="#444" stroke-width="0.5"/>'
    )
    # Y-axis tick labels (min + max, keeps it readable on narrow cards).
    parts.append(
        f'<text x="{pad_l - 4}" y="{pad_t + 4}" text-anchor="end" fill="#aaa">{_fmt(y_max)}</text>'
    )
    parts.append(
        f'<text x="{pad_l - 4}" y="{pad_t + inner_h + 4}" text-anchor="end" fill="#aaa">{_fmt(y_min)}</text>'
    )
    parts.append(
        f'<text x="{pad_l + inner_w / 2}" y="{height - 6}" text-anchor="middle" fill="#aaa">'
        f"epoch {_fmt(x_min)}…{_fmt(x_max)}</text>"
    )
    if y_label:
        parts.append(
            f'<text x="10" y="{pad_t - 4}" fill="#aaa">{escape(y_label)}</text>'
        )

    for idx, (name, pts) in enumerate(cleaned):
        color = palette[idx % len(palette)]
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
            for i, (x, y) in enumerate(pts)
        )
        parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
        # Dot on the last point + label.
        last_x, last_y = pts[-1]
        parts.append(
            f'<circle cx="{sx(last_x):.1f}" cy="{sy(last_y):.1f}" r="2" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{width - pad_r - 2}" y="{pad_t + 12 + idx * 12}" '
            f'text-anchor="end" fill="{color}">{escape(name)} ({_fmt(last_y)})</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


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


def _finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


__all__ = ["line_chart_svg", "history_to_series"]
