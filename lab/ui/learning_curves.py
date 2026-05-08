"""Server-side SVG renderer for per-experiment learning curves.

Pure-Python SVG path generator (no matplotlib dep at request time, no
client-side JS framework). One `Curve` instance per series, with
deterministic colour + axis ticks. Used by the experiment + study
templates to embed inline SVG charts directly in the HTML response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Series:
    label: str
    points: tuple[tuple[float, float], ...]   # (x, y) per epoch
    color: str


def collect_series(
    history: Iterable[dict],
    primary_metric: str,
) -> list[Series]:
    """Pull the four canonical curves out of a history list.

    Two y-scales are mixed in one chart on purpose: losses live in
    [0, ~1] for BCE-with-logits and metrics live in [0, 1] -- both fit
    on the same axis without normalisation tricks. When old experiments
    have only `loss` (back-compat), we treat it as ``train_loss``.
    """
    history = list(history)
    if not history:
        return []

    def _column(field: str, fallback: str | None = None) -> tuple[tuple[float, float], ...]:
        out: list[tuple[float, float]] = []
        for h in history:
            ep = h.get("epoch")
            v = h.get(field)
            if v is None and fallback is not None:
                v = h.get(fallback)
            if isinstance(ep, (int, float)) and isinstance(v, (int, float)):
                out.append((float(ep), float(v)))
        return tuple(out)

    series: list[Series] = []
    train_loss = _column("train_loss", fallback="loss")
    val_loss = _column("val_loss")
    metric = _column(primary_metric)
    f1 = _column("f1_macro")
    if train_loss:
        series.append(Series("train_loss", train_loss, "#3a6cd6"))
    if val_loss:
        series.append(Series("val_loss", val_loss, "#d6843a"))
    if metric:
        series.append(Series(primary_metric, metric, "#7bd88f"))
    if f1 and primary_metric != "f1_macro":
        series.append(Series("f1_macro", f1, "#b894ff"))
    return series


def render_svg(
    series: Sequence[Series],
    *,
    width: int = 560,
    height: int = 240,
    pad_left: int = 48,
    pad_right: int = 16,
    pad_top: int = 18,
    pad_bottom: int = 32,
) -> str:
    """Return inline SVG markup for ``series``. Empty when no data."""
    if not series or not any(s.points for s in series):
        return ""

    xs = [x for s in series for x, _ in s.points]
    ys = [y for s in series for _, y in s.points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1
    # pad y a little so points don't ride the chart edges
    span = y_max - y_min
    y_min -= span * 0.05
    y_max += span * 0.05

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def x_to_px(x: float) -> float:
        return pad_left + (x - x_min) / (x_max - x_min) * plot_w

    def y_to_px(y: float) -> float:
        return pad_top + (1 - (y - y_min) / (y_max - y_min)) * plot_h

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="background:#0a0c10;border:1px solid #2a2f37;border-radius:6px;'
        f'font-family:ui-monospace,monospace;font-size:10px">'
    )

    # gridlines + y ticks
    n_y_ticks = 4
    for i in range(n_y_ticks + 1):
        y_val = y_min + (y_max - y_min) * i / n_y_ticks
        y_px = y_to_px(y_val)
        parts.append(
            f'<line x1="{pad_left}" y1="{y_px:.1f}" x2="{width - pad_right}" '
            f'y2="{y_px:.1f}" stroke="#1c2129" stroke-width="0.5" />'
        )
        parts.append(
            f'<text x="{pad_left - 4}" y="{y_px + 3:.1f}" fill="#7a8493" '
            f'text-anchor="end">{y_val:.3f}</text>'
        )

    # x ticks (epochs)
    epoch_ticks = sorted({int(x) for s in series for x, _ in s.points})
    if len(epoch_ticks) > 8:
        step = max(1, len(epoch_ticks) // 8)
        epoch_ticks = epoch_ticks[::step]
    for ep in epoch_ticks:
        x_px = x_to_px(float(ep))
        parts.append(
            f'<line x1="{x_px:.1f}" y1="{height - pad_bottom}" '
            f'x2="{x_px:.1f}" y2="{height - pad_bottom + 4}" stroke="#7a8493" />'
        )
        parts.append(
            f'<text x="{x_px:.1f}" y="{height - pad_bottom + 14}" fill="#7a8493" '
            f'text-anchor="middle">e{ep}</text>'
        )

    # axes
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
        f'y2="{height - pad_bottom}" stroke="#3a4150" />'
    )
    parts.append(
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" '
        f'y2="{height - pad_bottom}" stroke="#3a4150" />'
    )

    # series
    for s in series:
        if not s.points:
            continue
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{x_to_px(x):.1f},{y_to_px(y):.1f}"
            for i, (x, y) in enumerate(s.points)
        )
        parts.append(
            f'<path d="{d}" fill="none" stroke="{s.color}" stroke-width="1.5" />'
        )
        for x, y in s.points:
            parts.append(
                f'<circle cx="{x_to_px(x):.1f}" cy="{y_to_px(y):.1f}" r="2.5" '
                f'fill="{s.color}" />'
            )

    # legend
    legend_y = pad_top + 4
    legend_x = pad_left + 8
    for s in series:
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 8}" width="10" height="2" '
            f'fill="{s.color}" />'
        )
        parts.append(
            f'<text x="{legend_x + 14}" y="{legend_y}" fill="#cfd6df">{s.label}</text>'
        )
        legend_x += len(s.label) * 6 + 30

    parts.append("</svg>")
    return "".join(parts)


def last_epoch_summary(history: Iterable[dict], primary_metric: str) -> dict:
    """One-shot dict of last-epoch values for the summary card."""
    history = list(history)
    if not history:
        return {}
    last = history[-1]
    return {
        "epoch": last.get("epoch"),
        "train_loss": last.get("train_loss", last.get("loss")),
        "val_loss": last.get("val_loss"),
        "f1_macro": last.get("f1_macro"),
        primary_metric: last.get(primary_metric),
        "n_epochs": len(history),
    }
