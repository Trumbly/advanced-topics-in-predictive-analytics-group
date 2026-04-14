"""StudySummary aggregation tests.

``iter_studies`` derives per-study best ROC-AUC / F1 from the experiments
inside each study.json, so the studies list can show both metrics (not
just the primary one).
"""
from __future__ import annotations

import json
from pathlib import Path

from lab.ui.loaders import iter_studies


def _write(experiments_dir: Path, study_id: str, experiments: list[dict]) -> None:
    d = experiments_dir / study_id
    d.mkdir(parents=True)
    (d / "study.json").write_text(json.dumps({
        "id": study_id,
        "name": study_id,
        "task_name": "track_b",
        "primary_metric": "f1_macro",
        "status": "completed",
        "best_score": max((e.get("metrics", {}).get("f1_macro", 0) for e in experiments), default=0),
        "experiments": experiments,
    }))


def test_study_summary_picks_best_roc_auc_and_f1_across_experiments(tmp_path):
    _write(tmp_path, "study_mix", [
        {"metrics": {"roc_auc_macro": 0.85, "f1_macro": 0.20}},
        {"metrics": {"roc_auc_macro": 0.95, "f1_macro": 0.10}},  # best ROC
        {"metrics": {"roc_auc_macro": 0.80, "f1_macro": 0.40}},  # best F1
    ])
    summaries = iter_studies(tmp_path)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.best_roc_auc == 0.95
    assert s.best_f1 == 0.40


def test_study_summary_tolerates_missing_metrics(tmp_path):
    _write(tmp_path, "study_nometrics", [
        {"metrics": {}},
        {"metrics": {"loss": 0.5}},
    ])
    s = iter_studies(tmp_path)[0]
    assert s.best_roc_auc is None
    assert s.best_f1 is None


def test_study_summary_accepts_binary_aliases(tmp_path):
    """Track A records f1_binary / roc_auc_binary — same family, must aggregate."""
    _write(tmp_path, "study_binary", [
        {"metrics": {"f1_binary": 0.77, "roc_auc_binary": 0.82}},
    ])
    s = iter_studies(tmp_path)[0]
    assert s.best_f1 == 0.77
    assert s.best_roc_auc == 0.82


def test_study_summary_returns_empty_for_missing_dir(tmp_path):
    assert iter_studies(tmp_path / "does_not_exist") == []
