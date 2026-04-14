"""Unit tests for `agent.memory.ExperimentMemory`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.memory import ExperimentMemory
from agent.models import (
    Experiment,
    ExperimentStatus,
    ModelConfig,
    TrainingResults,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


BASE_TIME = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)


def _make_experiment(
    eid: str,
    *,
    score: float | None,
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
    minutes: int = 0,
) -> Experiment:
    results = None
    if score is not None:
        results = TrainingResults(
            metrics={"f1_macro": score, "roc_auc_macro": score, "loss": 1.0 - score},
            duration_seconds=60.0,
        )
    return Experiment(
        experiment_id=eid,
        study_id="study_test",
        llm_model="gemma4:e4b",
        status=status,
        config=ModelConfig(
            architecture=f"arch_{eid}",
            hyperparams={"lr": 1e-3, "epochs": 5},
        ),
        results=results,
        created_at=BASE_TIME + timedelta(minutes=minutes),
    )


@pytest.fixture
def memory(tmp_path: Path) -> ExperimentMemory:
    return ExperimentMemory(study_dir=tmp_path / "study")


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


class TestBasics:
    def test_starts_empty(self, memory: ExperimentMemory) -> None:
        assert memory.is_empty()
        assert len(memory) == 0
        assert memory.last() is None
        assert memory.best() is None
        assert memory.best_score() is None

    def test_append_and_last(self, memory: ExperimentMemory) -> None:
        exp = _make_experiment("exp_001", score=0.5)
        memory.append(exp)
        assert not memory.is_empty()
        assert len(memory) == 1
        assert memory.last() is not None
        assert memory.last().experiment_id == "exp_001"  # type: ignore[union-attr]

    def test_get_by_id(self, memory: ExperimentMemory) -> None:
        memory.append(_make_experiment("exp_001", score=0.5))
        memory.append(_make_experiment("exp_002", score=0.6))
        assert memory.get("exp_002") is not None
        assert memory.get("exp_999") is None

    def test_append_replaces_existing(self, memory: ExperimentMemory) -> None:
        memory.append(_make_experiment("exp_001", score=0.5))
        memory.append(_make_experiment("exp_001", score=0.8))
        assert len(memory) == 1
        assert memory.best_score() == 0.8


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        study_dir = tmp_path / "study"
        mem1 = ExperimentMemory(study_dir=study_dir)
        mem1.append(_make_experiment("exp_001", score=0.5))
        mem1.append(_make_experiment("exp_002", score=0.7))

        mem2 = ExperimentMemory(study_dir=study_dir)
        assert len(mem2) == 2
        assert mem2.best_score() == 0.7

    def test_save_writes_both_json_and_markdown(
        self, memory: ExperimentMemory
    ) -> None:
        memory.append(_make_experiment("exp_001", score=0.5))
        assert memory.json_path.exists()
        assert memory.markdown_path.exists()
        md = memory.markdown_path.read_text()
        assert "exp_001" in md


# ---------------------------------------------------------------------------
# Queries: top_k, failures, best
# ---------------------------------------------------------------------------


class TestQueries:
    def test_top_k_sorts_by_score_desc(self, memory: ExperimentMemory) -> None:
        memory.append(_make_experiment("a", score=0.3))
        memory.append(_make_experiment("b", score=0.9))
        memory.append(_make_experiment("c", score=0.6))
        top = memory.top_k(2)
        assert [e.experiment_id for e in top] == ["b", "c"]

    def test_top_k_excludes_failures(self, memory: ExperimentMemory) -> None:
        memory.append(_make_experiment("a", score=0.3))
        memory.append(
            _make_experiment(
                "fail_1", score=None, status=ExperimentStatus.FAILED
            )
        )
        memory.append(_make_experiment("b", score=0.7))
        top = memory.top_k(5)
        ids = [e.experiment_id for e in top]
        assert "fail_1" not in ids
        assert ids == ["b", "a"]

    def test_recent_failures(self, memory: ExperimentMemory) -> None:
        memory.append(_make_experiment("ok", score=0.5, minutes=1))
        memory.append(
            _make_experiment(
                "fail_1", score=None, status=ExperimentStatus.FAILED, minutes=2
            )
        )
        memory.append(
            _make_experiment(
                "fail_2", score=None, status=ExperimentStatus.TIMEOUT, minutes=3
            )
        )
        fails = memory.recent_failures(5)
        assert len(fails) == 2
        assert fails[0].experiment_id == "fail_2"  # most recent first
        assert fails[1].experiment_id == "fail_1"

    def test_best_returns_highest_score(self, memory: ExperimentMemory) -> None:
        memory.append(_make_experiment("a", score=0.5))
        memory.append(_make_experiment("b", score=0.8))
        memory.append(_make_experiment("c", score=0.3))
        best = memory.best()
        assert best is not None
        assert best.experiment_id == "b"
        assert memory.best_score() == 0.8


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestMarkdown:
    def test_empty_returns_empty_string(self, memory: ExperimentMemory) -> None:
        assert memory.to_markdown() == ""

    def test_includes_top_k_section(self, memory: ExperimentMemory) -> None:
        memory.append(_make_experiment("exp_001", score=0.5))
        memory.append(_make_experiment("exp_002", score=0.7))
        md = memory.to_markdown(top_k=5)
        assert "Top-K Successful Experiments" in md
        assert "exp_001" in md
        assert "exp_002" in md
        assert "f1_macro" in md

    def test_includes_failures_section(self, memory: ExperimentMemory) -> None:
        memory.append(_make_experiment("exp_good", score=0.5))
        memory.append(
            _make_experiment(
                "exp_bad", score=None, status=ExperimentStatus.FAILED
            )
        )
        md = memory.to_markdown(top_k=5)
        assert "Recent Failures" in md
        assert "exp_bad" in md
