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


_EXP_PALETTE: tuple[str, ...] = (
    "#3a6cd6", "#7bd88f", "#d6843a", "#b894ff", "#e85d75", "#f0c14b",
    "#5cd4d4", "#ff9b54", "#a0d468", "#ec87c0", "#9b59b6", "#1abc9c",
)


def _exp_color(index: int) -> str:
    return _EXP_PALETTE[index % len(_EXP_PALETTE)]


def study_loss_series(experiments) -> tuple[list[Series], list[tuple[float, str]]]:
    """Cross-experiment train/val loss timeline for a study.

    X-axis is a *global* epoch counter so continuation runs land
    immediately after their source experiment, making the trajectory
    look like one continuous training run instead of two separate
    snippets. Returns ``(series, dividers)`` where each divider is a
    ``(global_epoch_at_boundary, exp_label)`` pair the renderer uses to
    draw a vertical separator + experiment annotation.

    Train loss = solid line, val loss = same colour but dashed -- one
    pair per experiment so eight experiments produce sixteen lines on
    the chart, all sharing the y-axis. Overfitting is the moment the
    dashed line bends up while the solid line keeps falling.
    """
    train_series: list[Series] = []
    val_series: list[Series] = []
    dividers: list[tuple[float, str]] = []

    # Map experiment id -> end-of-history global x so continuation runs
    # offset by their source's epoch count.
    end_x_by_exp: dict[str, float] = {}
    cumulative_offset = 0.0

    for idx, exp in enumerate(experiments):
        if not exp.history:
            continue
        proposal = getattr(exp, "proposal", None)
        cont_from = getattr(proposal, "continue_from_experiment_id", None) if proposal else None
        offset = end_x_by_exp.get(cont_from, cumulative_offset) if cont_from else cumulative_offset

        train_pts: list[tuple[float, float]] = []
        val_pts: list[tuple[float, float]] = []
        for h in exp.history:
            ep = h.get("epoch")
            if not isinstance(ep, (int, float)):
                continue
            x = offset + float(ep)
            tl = h.get("train_loss", h.get("loss"))
            vl = h.get("val_loss")
            if isinstance(tl, (int, float)):
                train_pts.append((x, float(tl)))
            if isinstance(vl, (int, float)):
                val_pts.append((x, float(vl)))

        if not train_pts and not val_pts:
            continue

        color = _exp_color(idx)
        label = exp.id
        if train_pts:
            train_series.append(Series(f"{label} train", tuple(train_pts), color))
        if val_pts:
            val_series.append(Series(f"{label} val", tuple(val_pts), color))
        last_x = max(
            (p[0] for p in train_pts + val_pts),
            default=offset,
        )
        end_x_by_exp[exp.id] = last_x
        cumulative_offset = last_x
        dividers.append((last_x, label))

    return train_series + val_series, dividers


def render_study_loss_svg(
    series: Sequence[Series],
    dividers: Sequence[tuple[float, str]] = (),
    *,
    width: int = 880,
    height: int = 320,
    pad_left: int = 56,
    pad_right: int = 16,
    pad_top: int = 22,
    pad_bottom: int = 56,
) -> str:
    """Multi-experiment loss SVG.

    Visually distinguishes train (solid) vs val (dashed) by checking the
    series label suffix; assigns different colour per experiment via
    `study_loss_series`. ``dividers`` draws a faint vertical line +
    experiment id at each boundary so continuation runs are marked.
    """
    if not series:
        return ""

    xs = [x for s in series for x, _ in s.points]
    ys = [y for s in series for _, y in s.points]
    if not xs or not ys:
        return ""
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1
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
        f'font-family:ui-monospace,monospace;font-size:10px;width:100%;max-width:{width}px">'
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

    # axes
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
        f'y2="{height - pad_bottom}" stroke="#3a4150" />'
    )
    parts.append(
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" '
        f'y2="{height - pad_bottom}" stroke="#3a4150" />'
    )

    # experiment dividers + labels — vertical line between experiments
    # (skip the last divider since that's the right edge of the chart)
    inner_dividers = list(dividers)[:-1] if len(dividers) > 1 else []
    for boundary_x, _ in inner_dividers:
        bx = x_to_px(boundary_x)
        parts.append(
            f'<line x1="{bx:.1f}" y1="{pad_top}" x2="{bx:.1f}" '
            f'y2="{height - pad_bottom}" stroke="#3a4150" stroke-dasharray="2,3" />'
        )
    for boundary_x, label in dividers:
        bx = x_to_px(boundary_x)
        parts.append(
            f'<text x="{bx:.1f}" y="{height - pad_bottom + 14}" fill="#7a8493" '
            f'text-anchor="middle" font-size="9">{label}</text>'
        )

    # series — train solid, val dashed (label suffix decides)
    for s in series:
        if not s.points:
            continue
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{x_to_px(x):.1f},{y_to_px(y):.1f}"
            for i, (x, y) in enumerate(s.points)
        )
        dash = '' if s.label.endswith(" train") else 'stroke-dasharray="4,3" '
        parts.append(
            f'<path d="{d}" fill="none" stroke="{s.color}" stroke-width="1.5" '
            f'{dash}/>'
        )

    # legend — one swatch per unique colour (one entry per experiment)
    seen_colors: dict[str, str] = {}
    for s in series:
        if s.color in seen_colors:
            continue
        # strip the " train" / " val" suffix from the label
        exp_label = s.label.rsplit(" ", 1)[0]
        seen_colors[s.color] = exp_label

    legend_y = height - pad_bottom + 32
    legend_x = pad_left
    for color, label in seen_colors.items():
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 8}" width="10" height="2" '
            f'fill="{color}" />'
        )
        parts.append(
            f'<text x="{legend_x + 14}" y="{legend_y}" fill="#cfd6df">{label}</text>'
        )
        legend_x += len(label) * 6 + 36

    # overfitting hint legend (solid vs dashed)
    parts.append(
        f'<text x="{width - pad_right}" y="{pad_top + 10}" fill="#7a8493" '
        f'text-anchor="end" font-size="9">'
        f'solid = train_loss · dashed = val_loss</text>'
    )
    parts.append(
        f'<text x="{width - pad_right}" y="{pad_top + 22}" fill="#7a8493" '
        f'text-anchor="end" font-size="9">'
        f'overfit when dashed climbs while solid keeps falling</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


_METRIC_COLORS: dict[str, str] = {
    "train_loss": "#3a6cd6",
    "val_loss": "#d6843a",
    "roc_auc_macro": "#7bd88f",
    "f1_macro": "#b894ff",
}


def history_to_chart_data(history: Iterable[dict], primary_metric: str) -> dict:
    """Per-experiment chart payload for the JS renderer.

    Each metric is a list of ``[epoch, value]`` pairs. The renderer can
    toggle individual metrics on/off and recompute the y-axis from
    whatever is visible. ``loss`` (legacy) is mapped onto ``train_loss``
    so old experiments still draw."""
    history = list(history)
    out: dict[str, object] = {
        "primary_metric": primary_metric,
        "metrics": {},
        "colors": dict(_METRIC_COLORS),
    }
    metric_keys = ("train_loss", "val_loss", "roc_auc_macro", "f1_macro")
    for key in metric_keys:
        points: list[list[float]] = []
        for h in history:
            ep = h.get("epoch")
            value = h.get(key)
            if value is None and key == "train_loss":
                value = h.get("loss")
            if isinstance(ep, (int, float)) and isinstance(value, (int, float)):
                points.append([float(ep), float(value)])
        if points:
            out["metrics"][key] = points
    return out


def study_to_chart_data(experiments) -> dict:
    """Cross-experiment payload: one entry per experiment with its
    metric series, x_offset (epochs accumulated from earlier
    experiments / source of a continuation), and a stable colour.

    The JS renderer adds ``x_offset`` to each epoch for the global axis;
    when the user un-checks an experiment it gets removed from the
    rescale calculation too.
    """
    out: list[dict] = []
    end_x_by_exp: dict[str, float] = {}
    cumulative_offset = 0.0

    for idx, exp in enumerate(experiments):
        if not exp.history:
            continue
        proposal = getattr(exp, "proposal", None)
        cont_from = getattr(proposal, "continue_from_experiment_id", None) if proposal else None
        offset = end_x_by_exp.get(cont_from, cumulative_offset) if cont_from else cumulative_offset

        chart = history_to_chart_data(exp.history, exp.primary_metric)
        if not chart["metrics"]:
            continue

        # Every experiment carries every epoch present in its history;
        # use the max to advance the global cursor.
        last_local_epoch = max(
            (p[0] for points in chart["metrics"].values() for p in points),
            default=0.0,
        )

        out.append(
            {
                "id": exp.id,
                "label": exp.id,
                "x_offset": offset,
                "color": _exp_color(idx),
                "primary_metric": exp.primary_metric,
                "metrics": chart["metrics"],
                "continued_from": cont_from,
            }
        )
        end_x_by_exp[exp.id] = offset + last_local_epoch
        cumulative_offset = offset + last_local_epoch
    return {"experiments": out, "metric_colors": dict(_METRIC_COLORS)}


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
