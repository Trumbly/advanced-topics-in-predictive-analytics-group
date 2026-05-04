"""I-05 acceptance: memory top-K + recent failures + seeding."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.core.memory import Memory, summarize_curve
from lab.core.models import Experiment, Proposal, Study, Task, TaskError


def _proposal(name: str, family: str = "cnn_scratch") -> Proposal:
    return Proposal(
        architecture_name=name,
        family=family,
        lr=1e-3,
        lr_schedule="cosine",
        epochs=3,
    )


def _success(idx: int, score: float, *, history: list[dict] | None = None) -> Experiment:
    return Experiment(
        id=f"exp_ok_{idx:04d}",
        index=idx,
        status="JUDGED",
        proposal=_proposal(f"Arch{idx}"),
        primary_metric="roc_auc_macro",
        primary_score=score,
        history=history or [
            {"epoch": 1, "loss": 0.7, "roc_auc_macro": 0.6},
            {"epoch": 2, "loss": 0.5, "roc_auc_macro": 0.7},
            {"epoch": 3, "loss": 0.4, "roc_auc_macro": score},
        ],
    )


def _failure(idx: int, err_type: str = "ShapeMismatch", msg: str = "shape mismatch") -> Experiment:
    return Experiment(
        id=f"exp_fail_{idx:04d}",
        index=idx,
        status="FAILED",
        proposal=_proposal(f"BadArch{idx}", family="efficientnet_pretrained"),
        primary_metric="roc_auc_macro",
        primary_score=None,
        tasks=[
            Task(
                name="execute",
                status="FAILED",
                error=TaskError(error_type=err_type, message=msg),
            )
        ],
    )


def _study(experiments: list[Experiment]) -> Study:
    return Study(
        id=f"study_20260504_120000_xxxx",
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=experiments,
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )


# -------- top-K + recent failures --------

def test_top_k_keeps_only_best_by_score(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=5, path=tmp_path / "mem.json")
    for i, score in enumerate([0.4, 0.7, 0.5, 0.9, 0.6]):
        m.add(_success(i, score))
    ids = [e.id for e in m.wins]
    scores = [e.primary_score for e in m.wins]
    assert scores == [0.9, 0.7, 0.6]
    assert len(ids) == 3


def test_recent_failures_keeps_last_n(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=2, path=tmp_path / "mem.json")
    for i in range(5):
        m.add(_failure(i))
    assert [e.index for e in m.failures] == [3, 4]


def test_to_markdown_under_budget(tmp_path: Path):
    m = Memory(top_k=20, recent_failures=20, path=tmp_path / "mem.json")
    for i in range(20):
        m.add(_success(i, 0.5 + 0.01 * i))
    for i in range(20):
        m.add(_failure(i, msg="x" * 200))
    md = m.to_markdown()
    assert len(md.encode("utf-8")) <= 3072


# -------- seeding --------

def test_seed_from_predecessor_pulls_in_experiments(tmp_path: Path):
    pred_exps = [_success(0, 0.5), _success(1, 0.8), _failure(2)]
    pred = _study(pred_exps)
    m = Memory(top_k=5, recent_failures=5, path=tmp_path / "mem.json")
    m.seed_from_predecessor(pred)
    assert len(m.wins) == 2
    assert len(m.failures) == 1


def test_seed_from_agent_memory_filters_by_task(tmp_path: Path):
    studies_root = tmp_path / "studies"
    studies_root.mkdir()

    s1 = _study([_success(0, 0.7)])
    s1.id = "study_s1"
    s1.save(studies_root)

    s2 = _study([_success(0, 0.9)])
    s2.id = "study_s2"
    s2.task_name = "track_other"
    s2.save(studies_root)

    m = Memory(top_k=5, recent_failures=5, path=tmp_path / "mem.json")
    m.seed_from_agent_memory(studies_root, task="track_b")
    assert len(m.wins) == 1
    assert m.wins[0].primary_score == 0.7


# -------- save/load --------

def test_save_and_load_round_trip(tmp_path: Path):
    m = Memory(top_k=3, recent_failures=3, path=tmp_path / "mem.json")
    m.add(_success(0, 0.7))
    m.add(_failure(1))
    m.save()

    loaded = Memory.load(tmp_path / "mem.json", top_k=3, recent_failures=3)
    assert len(loaded.wins) == 1
    assert len(loaded.failures) == 1
    assert loaded.wins[0].primary_score == 0.7


def test_load_returns_empty_when_file_missing(tmp_path: Path):
    m = Memory.load(tmp_path / "nope.json", top_k=3, recent_failures=3)
    assert m.wins == [] and m.failures == []


# -------- summarize_curve trend classification --------

def test_curve_trend_improving():
    history = [
        {"epoch": 1, "loss": 0.9, "roc_auc_macro": 0.5},
        {"epoch": 2, "loss": 0.6, "roc_auc_macro": 0.65},
        {"epoch": 3, "loss": 0.4, "roc_auc_macro": 0.85},
    ]
    s = summarize_curve(history, "roc_auc_macro")
    assert "improving" in s


def test_curve_trend_flat():
    history = [{"epoch": i, "roc_auc_macro": 0.5} for i in range(1, 5)]
    s = summarize_curve(history, "roc_auc_macro")
    assert "flat" in s


def test_curve_trend_regressing():
    history = [
        {"epoch": 1, "roc_auc_macro": 0.8},
        {"epoch": 2, "roc_auc_macro": 0.6},
        {"epoch": 3, "roc_auc_macro": 0.3},
    ]
    s = summarize_curve(history, "roc_auc_macro")
    assert "regressing" in s


def test_summarize_curve_empty_history():
    assert summarize_curve([], "roc_auc_macro") == "no history"


def test_has_checkpoint_for(tmp_path: Path):
    m = Memory(top_k=2, recent_failures=2, path=tmp_path / "mem.json")
    exp = _success(0, 0.7)
    exp.checkpoint_path = "/tmp/x.pt"
    m.add(exp)
    assert m.has_checkpoint_for(exp.id)
    assert not m.has_checkpoint_for("does-not-exist")
