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


# ---------- cross-experiment study chart ----------


from dataclasses import dataclass

from lab.ui.learning_curves import render_study_loss_svg, study_loss_series


@dataclass
class _StubProposal:
    continue_from_experiment_id: str | None = None


@dataclass
class _StubExp:
    id: str
    history: list[dict]
    proposal: _StubProposal | None = None
    primary_metric: str = "roc_auc_macro"


def test_study_loss_series_concatenates_continuation_after_source():
    """A continued experiment's epochs offset by the source's epoch count
    so the chart looks like one continuous trajectory instead of two
    separate snippets — that's the whole point of the
    `continue_from_experiment_id` action."""
    a = _StubExp(
        "exp_a",
        [
            {"epoch": 1, "train_loss": 0.7, "val_loss": 0.8},
            {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6},
        ],
    )
    b = _StubExp(
        "exp_b",
        [
            {"epoch": 1, "train_loss": 0.4, "val_loss": 0.55},
            {"epoch": 2, "train_loss": 0.3, "val_loss": 0.5},
        ],
        proposal=_StubProposal(continue_from_experiment_id="exp_a"),
    )
    series, dividers = study_loss_series([a, b])

    # Pull exp_b's train series out — its xs should start at 3 (offset 2).
    b_train = next(s for s in series if s.label == "exp_b train")
    xs = [x for x, _ in b_train.points]
    assert xs == [3.0, 4.0]
    assert ("exp_a", 2.0) in {(label, last_x) for last_x, label in dividers} or any(
        label == "exp_a" for _, label in dividers
    )


def test_study_loss_series_independent_runs_concatenate_chronologically():
    """When experiments are NOT continuations, each one starts where the
    previous left off so all curves fit on a single x-axis without
    overlapping."""
    a = _StubExp("exp_a", [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6}])
    b = _StubExp("exp_b", [{"epoch": 1, "train_loss": 0.4, "val_loss": 0.5}])
    series, _ = study_loss_series([a, b])
    # exp_b should start at x=2 (after exp_a's only epoch at x=1)
    b_train = next(s for s in series if s.label == "exp_b train")
    assert b_train.points[0][0] == 2.0


def test_study_loss_series_emits_train_and_val_per_experiment():
    a = _StubExp("exp_a", [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
        {"epoch": 2, "train_loss": 0.3, "val_loss": 0.55},
    ])
    series, _ = study_loss_series([a])
    labels = sorted(s.label for s in series)
    assert labels == ["exp_a train", "exp_a val"]


def test_study_loss_series_skips_experiments_without_history():
    a = _StubExp("exp_a", [])
    b = _StubExp("exp_b", [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6}])
    series, dividers = study_loss_series([a, b])
    # only exp_b shows up in series + dividers
    assert all("exp_a" not in s.label for s in series)
    assert any("exp_b" == label for _, label in dividers)


def test_render_study_loss_svg_marks_train_solid_val_dashed():
    """Crucial for the overfit story: the user must be able to tell at
    a glance which line is which without consulting the legend."""
    series = [
        Series("exp_a train", ((1, 0.7), (2, 0.5)), "#3a6cd6"),
        Series("exp_a val", ((1, 0.8), (2, 0.6)), "#3a6cd6"),
    ]
    svg = render_study_loss_svg(series, dividers=[(2.0, "exp_a")])
    # Train path must NOT carry stroke-dasharray; val MUST.
    assert svg.count("stroke-dasharray=") >= 1   # at least val
    paths = [
        line for line in svg.split("<path ") if line.startswith('d="')
    ]
    assert len(paths) == 2
    # Find the train path (the one without stroke-dasharray)
    n_dashed_paths = sum(1 for p in paths if "stroke-dasharray" in p)
    assert n_dashed_paths == 1   # exactly the val path


def test_render_study_loss_svg_uses_distinct_colors_per_experiment():
    series, _ = study_loss_series([
        _StubExp("exp_a", [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6}]),
        _StubExp("exp_b", [{"epoch": 1, "train_loss": 0.4, "val_loss": 0.55}]),
    ])
    a_color = next(s.color for s in series if s.label.startswith("exp_a"))
    b_color = next(s.color for s in series if s.label.startswith("exp_b"))
    assert a_color != b_color


def test_render_study_loss_svg_empty_series_yields_empty_string():
    assert render_study_loss_svg([], []) == ""


def test_render_study_loss_svg_includes_overfit_legend_hint():
    series = [Series("exp_a train", ((1, 0.5),), "#3a6cd6")]
    svg = render_study_loss_svg(series, dividers=[(1.0, "exp_a")])
    assert "overfit" in svg.lower()
    assert "solid" in svg.lower() and "dashed" in svg.lower()


# ---------- chart-data payloads (for the JS renderer) ----------


from lab.ui.learning_curves import history_to_chart_data, study_to_chart_data


def test_history_to_chart_data_includes_every_present_metric():
    history = _hist(
        {"epoch": 1, "train_loss": 0.7, "val_loss": 0.8, "roc_auc_macro": 0.55, "f1_macro": 0.5},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6, "roc_auc_macro": 0.65, "f1_macro": 0.6},
    )
    out = history_to_chart_data(history, "roc_auc_macro")
    assert set(out["metrics"]) == {"train_loss", "val_loss", "roc_auc_macro", "f1_macro"}
    assert out["metrics"]["train_loss"] == [[1.0, 0.7], [2.0, 0.5]]
    assert "colors" in out


def test_history_to_chart_data_uses_legacy_loss_for_train_loss():
    history = _hist({"epoch": 1, "loss": 0.42, "roc_auc_macro": 0.6})
    out = history_to_chart_data(history, "roc_auc_macro")
    assert out["metrics"]["train_loss"] == [[1.0, 0.42]]


def test_history_to_chart_data_skips_metrics_without_points():
    """A history with only train_loss must NOT carry empty val_loss/etc.
    keys — the JS renderer would try to draw an empty series and the
    metric checkbox would always toggle a no-op."""
    history = _hist({"epoch": 1, "train_loss": 0.5})
    out = history_to_chart_data(history, "roc_auc_macro")
    assert set(out["metrics"]) == {"train_loss"}


def test_study_to_chart_data_emits_one_entry_per_experiment():
    a = _StubExp("exp_a", [
        {"epoch": 1, "train_loss": 0.7, "val_loss": 0.8},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6},
    ])
    b = _StubExp("exp_b", [
        {"epoch": 1, "train_loss": 0.4, "val_loss": 0.55},
    ])
    out = study_to_chart_data([a, b])
    ids = [e["id"] for e in out["experiments"]]
    assert ids == ["exp_a", "exp_b"]
    # Independent runs concatenate: b starts at x_offset = 2 (a's last epoch)
    a_entry = out["experiments"][0]
    b_entry = out["experiments"][1]
    assert a_entry["x_offset"] == 0.0
    assert b_entry["x_offset"] == 2.0
    # Distinct colours so the JS palette doesn't collapse
    assert a_entry["color"] != b_entry["color"]


def test_study_to_chart_data_continuation_offsets_to_source_end():
    a = _StubExp("exp_a", [
        {"epoch": 1, "train_loss": 0.7, "val_loss": 0.8},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6},
    ])
    b = _StubExp(
        "exp_b",
        [{"epoch": 1, "train_loss": 0.4, "val_loss": 0.55}],
        proposal=_StubProposal(continue_from_experiment_id="exp_a"),
    )
    out = study_to_chart_data([a, b])
    b_entry = next(e for e in out["experiments"] if e["id"] == "exp_b")
    # Continuation lands at the source's end (= a's last epoch = 2)
    assert b_entry["x_offset"] == 2.0
    assert b_entry["continued_from"] == "exp_a"


def test_study_to_chart_data_skips_experiments_without_history():
    out = study_to_chart_data([_StubExp("exp_a", [])])
    assert out["experiments"] == []
