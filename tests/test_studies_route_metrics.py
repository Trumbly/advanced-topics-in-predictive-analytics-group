from __future__ import annotations

from lab.core.models import Experiment, Study
from lab.ui.routes.studies import _metric_value_for_experiment, _recompute_best


def test_metric_value_for_experiment_prefers_metrics_dict():
    exp = Experiment(
        study_id="s",
        primary_metric="f1_macro",
        primary_score=0.2,
        metrics={"f1_macro": 0.2, "roc_auc_macro": 0.8},
        history=[{"roc_auc_macro": 0.7}],
    )
    assert _metric_value_for_experiment(exp, "roc_auc_macro") == 0.8


def test_metric_value_for_experiment_falls_back_to_history_best():
    exp = Experiment(
        study_id="s",
        primary_metric="f1_macro",
        primary_score=0.2,
        metrics={},
        history=[{"roc_auc_macro": 0.61}, {"roc_auc_macro": 0.72}],
    )
    assert _metric_value_for_experiment(exp, "roc_auc_macro") == 0.72


def test_recompute_best_uses_study_primary_metric_when_available():
    study = Study(task_name="track_b", primary_metric="roc_auc_macro")
    e1 = Experiment(study_id=study.id, id="exp1", primary_metric="f1_macro", primary_score=0.91)
    e2 = Experiment(study_id=study.id, id="exp2", primary_metric="roc_auc_macro", primary_score=0.73)
    e3 = Experiment(study_id=study.id, id="exp3", primary_metric="roc_auc_macro", primary_score=0.77)
    study.experiments = [e1, e2, e3]

    _recompute_best(study)

    assert study.best_experiment_id == "exp3"
    assert study.best_score == 0.77


def test_recompute_best_falls_back_to_any_metric_if_no_match():
    study = Study(task_name="track_b", primary_metric="roc_auc_macro")
    e1 = Experiment(study_id=study.id, id="exp1", primary_metric="f1_macro", primary_score=0.91)
    e2 = Experiment(study_id=study.id, id="exp2", primary_metric="f1_macro", primary_score=0.73)
    study.experiments = [e1, e2]

    _recompute_best(study)

    assert study.best_experiment_id == "exp1"
    assert study.best_score == 0.91
