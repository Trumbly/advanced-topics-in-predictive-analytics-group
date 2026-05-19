"""I-13 acceptance: figures + report generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.config import load_settings
from lab.core.models import Experiment, Proposal, Study, Task, TaskError, Verdict
from lab.reporting.figures import render_figures
from lab.reporting.generator import generate_report

REPO_ROOT = Path(__file__).resolve().parent.parent


def _study() -> Study:
    proposal_a = Proposal(
        architecture_name="EffNetB0",
        family="efficientnet_pretrained",
        lr=3e-4,
        lr_schedule="cosine",
        epochs=3,
    )
    proposal_b = Proposal(
        architecture_name="CnnSmall",
        family="cnn_scratch",
        lr=1e-3,
        lr_schedule="constant",
        epochs=2,
    )
    e1 = Experiment(
        id="exp_0001",
        index=0,
        status="JUDGED",
        proposal=proposal_a,
        primary_metric="roc_auc_macro",
        primary_score=0.55,
        history=[
            {"epoch": 1, "loss": 0.7, "roc_auc_macro": 0.50},
            {"epoch": 2, "loss": 0.5, "roc_auc_macro": 0.55},
        ],
        verdict=Verdict(verdict="keep", score=0.6, rationale="ok"),
    )
    e2 = Experiment(
        id="exp_0002",
        index=1,
        status="JUDGED",
        proposal=proposal_b,
        primary_metric="roc_auc_macro",
        primary_score=0.41,
        history=[
            {"epoch": 1, "loss": 0.8, "roc_auc_macro": 0.30},
            {"epoch": 2, "loss": 0.6, "roc_auc_macro": 0.41},
        ],
    )
    e3 = Experiment(
        id="exp_0003",
        index=2,
        status="FAILED",
        proposal=proposal_a,
        primary_metric="roc_auc_macro",
        tasks=[
            Task(
                name="execute",
                status="FAILED",
                error=TaskError(error_type="ShapeMismatch", message="x"),
            )
        ],
    )
    return Study(
        id="study_test_xxxx",
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[e1, e2, e3],
        best_experiment_id=e1.id,
        best_score=0.55,
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )


def test_render_figures_produces_all_required(tmp_path):
    paths = render_figures(_study(), tmp_path)
    assert "score_progression" in paths
    assert "best_learning_curve" in paths
    assert "failure_breakdown" in paths
    assert "family_performance" in paths
    for name, p in paths.items():
        assert p.exists() and p.stat().st_size > 0, name


def test_per_class_auc_skipped_when_metric_missing(tmp_path):
    paths = render_figures(_study(), tmp_path)
    assert "per_class_auc" not in paths


def test_per_class_auc_rendered_when_present(tmp_path):
    s = _study()
    # inject per_class_auc in best
    s.experiments[0].metrics = {"per_class_auc": [0.9, 0.6, 0.7, 0.8]}
    paths = render_figures(s, tmp_path)
    assert "per_class_auc" in paths
    assert paths["per_class_auc"].exists()


def test_generate_report_writes_md_and_html(tmp_path):
    settings = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = settings.paths.model_copy(
        update={"experiments_dir": str(tmp_path / "studies")}
    )
    s = settings.model_copy(update={"paths": new_paths})

    md_path = generate_report(_study(), s)
    assert md_path.exists()
    html_path = md_path.with_suffix(".html")
    assert html_path.exists()

    md = md_path.read_text()
    # The top-experiments table now shows human-readable labels like "Exp 1 (0001)"
    assert "Exp 1 (0001)" in md  # best experiment label
    assert "ShapeMismatch" in md  # failure summary


def test_report_html_renders_table_for_top_experiments(tmp_path):
    """Regression: the 'Top experiments' pipe-table must come out as a real
    <table> element. markdown-it's commonmark profile does not enable the
    table extension by default, so we explicitly enable it in the generator."""
    settings = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = settings.paths.model_copy(
        update={"experiments_dir": str(tmp_path / "studies")}
    )
    s = settings.model_copy(update={"paths": new_paths})

    md_path = generate_report(_study(), s)
    html = md_path.with_suffix(".html").read_text()
    assert "<table>" in html
    assert "<th>id</th>" in html
    # report.md.j2 routes experiment ids through exp_label_for, so the
    # rendered cell carries the human-readable label rather than the raw id.
    assert "<td>Exp 1 (0001)</td>" in html


def test_report_html_inlines_figures_as_data_uris(tmp_path):
    """Regression: <img src="score_progression.png"> would 404 when served
    by the UI route /reports/<id>. We inline every figure as a base64 data
    URI so the file is self-contained."""
    settings = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = settings.paths.model_copy(
        update={"experiments_dir": str(tmp_path / "studies")}
    )
    s = settings.model_copy(update={"paths": new_paths})

    md_path = generate_report(_study(), s)
    html = md_path.with_suffix(".html").read_text()
    # No relative image references survive into the served HTML
    assert 'src="score_progression.png"' not in html
    # Each figure is embedded as a data: URI
    assert 'src="data:image/png;base64,' in html


def test_report_html_is_self_contained_document(tmp_path):
    """Regression: the served HTML must include the <head><style> wrapper so
    the report doesn't render as an unstyled plain-serif page."""
    settings = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = settings.paths.model_copy(
        update={"experiments_dir": str(tmp_path / "studies")}
    )
    s = settings.model_copy(update={"paths": new_paths})

    md_path = generate_report(_study(), s)
    html = md_path.with_suffix(".html").read_text()
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "<style>" in html
    assert "</style>" in html
    assert "<body>" in html


def test_report_renders_with_no_score_history(tmp_path):
    settings = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = settings.paths.model_copy(
        update={"experiments_dir": str(tmp_path / "studies")}
    )
    s = settings.model_copy(update={"paths": new_paths})
    bare = Study(
        id="study_bare_xxxx",
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[],
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )
    md_path = generate_report(bare, s)
    assert md_path.exists()
    md = md_path.read_text()
    assert "No experiments" in md
