"""I-18 acceptance: cross-study prompt scoring + use-best selection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.core.models import Experiment, Proposal, Study
from lab.prompts.scoring import (
    PromptScoreStats,
    aggregate_prompt_scores,
    get_best_version,
)


def _proposal() -> Proposal:
    return Proposal(
        architecture_name="A",
        family="cnn_scratch",
        lr=1e-3,
        lr_schedule="cosine",
        epochs=2,
    )


def _exp(score: float, idx: int = 0) -> Experiment:
    return Experiment(
        id=f"exp_{idx:04d}",
        index=idx,
        status="JUDGED",
        proposal=_proposal(),
        primary_metric="roc_auc_macro",
        primary_score=score,
    )


def _study_with_version(
    sid: str, version: str, scores: list[float], task: str = "track_b"
) -> Study:
    return Study(
        id=sid,
        task_name=task,
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        prompt_template_paths={
            "propose_architecture": Path(f"config/prompts/propose_architecture/{version}.yaml")
        },
        experiments=[_exp(s, i) for i, s in enumerate(scores)],
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )


def test_aggregate_scores_two_versions(tmp_path):
    a = _study_with_version("s_a_1", "v1", [0.3, 0.4])
    b = _study_with_version("s_a_2", "v1", [0.2, 0.3])
    c = _study_with_version("s_b_1", "v2", [0.5, 0.6])
    for s in (a, b, c):
        s.save(tmp_path)

    table = aggregate_prompt_scores(tmp_path)
    assert "track_b" in table
    assert set(table["track_b"].keys()) == {"v1", "v2"}
    v1 = table["track_b"]["v1"]
    v2 = table["track_b"]["v2"]
    assert v1.count == 4
    assert v2.count == 2
    assert v1.mean < v2.mean


def test_get_best_version_picks_higher_mean(tmp_path):
    for i in range(4):
        _study_with_version(f"s_v1_{i}", "v1", [0.3]).save(tmp_path)
    for i in range(4):
        _study_with_version(f"s_v2_{i}", "v2", [0.5]).save(tmp_path)

    assert get_best_version("track_b", tmp_path, min_runs=3) == "v2"


def test_min_runs_filters_low_count_versions(tmp_path):
    _study_with_version("s_lucky", "v9", [0.99]).save(tmp_path)  # 1 run, very high
    for i in range(4):
        _study_with_version(f"s_v1_{i}", "v1", [0.5]).save(tmp_path)

    # v9 has only 1 run - filtered out by min_runs=3
    assert get_best_version("track_b", tmp_path, min_runs=3) == "v1"


def test_no_eligible_versions_returns_none(tmp_path):
    _study_with_version("s", "v1", [0.5]).save(tmp_path)  # only 1 run
    assert get_best_version("track_b", tmp_path, min_runs=3) is None


def test_studies_without_prompt_path_are_skipped(tmp_path):
    s = _study_with_version("s_v1", "v1", [0.5, 0.6, 0.7])
    s.prompt_template_paths = {}
    s.save(tmp_path)
    table = aggregate_prompt_scores(tmp_path)
    assert "track_b" not in table or "v1" not in table.get("track_b", {})


def test_aggregate_returns_pydantic_stats(tmp_path):
    _study_with_version("s1", "v1", [0.3, 0.4, 0.5]).save(tmp_path)
    table = aggregate_prompt_scores(tmp_path)
    stats = table["track_b"]["v1"]
    assert isinstance(stats, PromptScoreStats)
    assert stats.best == 0.5
    assert stats.mean == pytest.approx(0.4)
