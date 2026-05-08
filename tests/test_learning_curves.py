"""Server-side learning-curve renderer: SVG markup + last-epoch summary.

The SVG is generated as a plain string (no matplotlib at request time),
so we assert structural invariants — series presence, axis tick count,
and that empty histories return an empty string instead of a broken SVG.
"""

from __future__ import annotations

from lab.ui.learning_curves import (
    Series,
    collect_series,
    last_epoch_summary,
    render_svg,
)


def _hist(*epochs: dict) -> list[dict]:
    return list(epochs)


# ---------- collect_series ----------


def test_collect_series_returns_train_and_val_loss_when_present():
    history = _hist(
        {"epoch": 1, "train_loss": 0.7, "val_loss": 0.8, "roc_auc_macro": 0.55, "f1_macro": 0.50},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6, "roc_auc_macro": 0.65, "f1_macro": 0.58},
    )
    series = collect_series(history, "roc_auc_macro")
    labels = [s.label for s in series]
    assert "train_loss" in labels
    assert "val_loss" in labels
    assert "roc_auc_macro" in labels
    assert "f1_macro" in labels


def test_collect_series_falls_back_to_legacy_loss_field():
    """Old experiments only emit `loss`; treat it as `train_loss` so the
    chart still renders for runs from before the train/val split."""
    history = _hist(
        {"epoch": 1, "loss": 0.7, "roc_auc_macro": 0.55},
        {"epoch": 2, "loss": 0.5, "roc_auc_macro": 0.62},
    )
    series = collect_series(history, "roc_auc_macro")
    train = next((s for s in series if s.label == "train_loss"), None)
    assert train is not None
    assert train.points == ((1.0, 0.7), (2.0, 0.5))


def test_collect_series_skips_missing_metric_columns():
    history = _hist({"epoch": 1, "train_loss": 0.5})
    series = collect_series(history, "roc_auc_macro")
    labels = [s.label for s in series]
    assert labels == ["train_loss"]


def test_collect_series_empty_history_returns_empty_list():
    assert collect_series([], "roc_auc_macro") == []


def test_collect_series_omits_f1_when_it_is_the_primary_metric():
    """Avoid drawing the same series twice when f1_macro is both the
    primary metric and the legacy `f1_macro` column."""
    history = _hist(
        {"epoch": 1, "train_loss": 0.5, "f1_macro": 0.4},
        {"epoch": 2, "train_loss": 0.4, "f1_macro": 0.55},
    )
    series = collect_series(history, "f1_macro")
    labels = [s.label for s in series]
    # f1_macro appears once (as the primary), not twice.
    assert labels.count("f1_macro") == 1


# ---------- render_svg ----------


def test_render_svg_emits_one_path_per_series():
    series = [
        Series("train_loss", ((1, 0.7), (2, 0.5)), "#3a6cd6"),
        Series("val_loss", ((1, 0.8), (2, 0.6)), "#d6843a"),
    ]
    svg = render_svg(series)
    assert svg.count("<path ") == 2
    assert "#3a6cd6" in svg
    assert "#d6843a" in svg
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")


def test_render_svg_returns_empty_string_for_no_series():
    assert render_svg([]) == ""


def test_render_svg_returns_empty_string_when_all_series_have_no_points():
    assert render_svg([Series("x", (), "#000")]) == ""


def test_render_svg_includes_legend_label_text():
    series = [Series("train_loss", ((1, 0.5),), "#3a6cd6")]
    svg = render_svg(series)
    assert ">train_loss<" in svg


def test_render_svg_axis_ticks_match_unique_epochs():
    series = [Series("train_loss", ((1, 0.5), (2, 0.4), (3, 0.3)), "#3a6cd6")]
    svg = render_svg(series)
    # one x-tick per epoch (e1, e2, e3 in legend text)
    assert ">e1<" in svg
    assert ">e2<" in svg
    assert ">e3<" in svg


# ---------- last_epoch_summary ----------


def test_last_epoch_summary_returns_canonical_columns():
    history = _hist(
        {"epoch": 1, "train_loss": 0.7, "val_loss": 0.8, "roc_auc_macro": 0.55, "f1_macro": 0.5},
        {"epoch": 2, "train_loss": 0.4, "val_loss": 0.6, "roc_auc_macro": 0.72, "f1_macro": 0.65},
    )
    s = last_epoch_summary(history, "roc_auc_macro")
    assert s["epoch"] == 2
    assert s["train_loss"] == 0.4
    assert s["val_loss"] == 0.6
    assert s["roc_auc_macro"] == 0.72
    assert s["n_epochs"] == 2


def test_last_epoch_summary_falls_back_to_legacy_loss():
    history = _hist({"epoch": 5, "loss": 0.42, "roc_auc_macro": 0.7})
    s = last_epoch_summary(history, "roc_auc_macro")
    assert s["train_loss"] == 0.42


def test_last_epoch_summary_empty_history_is_empty_dict():
    assert last_epoch_summary([], "roc_auc_macro") == {}
