from __future__ import annotations

from lab.core.memory import Memory
from lab.core.models import (
    Experiment,
    ExperimentStatus,
    Study,
    TaskError,
)


def _success(name: str, score: float, family: str = "cnn") -> Experiment:
    return Experiment(
        study_id="s1",
        architecture_name=name,
        architecture_family=family,
        status=ExperimentStatus.COMPLETED,
        primary_metric="f1_macro",
        primary_score=score,
    )


def _failure(name: str, err_type: str) -> Experiment:
    return Experiment(
        study_id="s1",
        architecture_name=name,
        architecture_family="cnn",
        status=ExperimentStatus.FAILED,
        primary_metric="f1_macro",
        error=TaskError(error_type=err_type, message=f"{err_type} happened"),
    )


def test_top_k_returns_sorted_by_score():
    m = Memory(score_metric="f1_macro")
    m.add(_success("a", 0.3))
    m.add(_success("b", 0.7))
    m.add(_success("c", 0.5))
    top = m.top_k(2)
    assert [e.architecture_name for e in top] == ["b", "c"]


def test_recent_failures_returns_last_n():
    m = Memory(score_metric="f1_macro")
    m.add(_failure("x", "ShapeMismatch"))
    m.add(_failure("y", "OOM"))
    m.add(_failure("z", "SyntaxError"))
    recent = m.recent_failures(2)
    assert [e.architecture_name for e in recent] == ["y", "z"]


def test_to_markdown_mentions_winners_and_failures():
    m = Memory(score_metric="f1_macro")
    m.add(_success("bilstm", 0.82))
    m.add(_failure("broken", "ShapeMismatch"))
    md = m.to_markdown(top_k=3)
    assert "bilstm" in md
    assert "0.8200" in md
    assert "broken" in md
    assert "ShapeMismatch" in md


def test_seed_from_predecessor_copies_entries(tmp_path):
    pred = Study(task_name="track_b", primary_metric="f1_macro")
    pred.experiments.append(_success("old", 0.6))
    m = Memory(score_metric="f1_macro")
    m.seed_from_predecessor(pred)
    assert len(m.entries) == 1
    assert m.entries[0].architecture_name == "old"
