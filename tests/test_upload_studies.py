"""Tests for the selective-upload filter logic."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.upload_studies import select_studies


def _mk(tmp: Path, study_id: str, *, task: str = "track_a",
        publish: bool = False, tags: list[str] | None = None,
        best_score: float | None = None) -> None:
    d = tmp / study_id
    d.mkdir(parents=True)
    (d / "study.json").write_text(json.dumps({
        "id": study_id,
        "task_name": task,
        "publish": publish,
        "tags": tags or [],
        "best_score": best_score,
        "experiments": [],
    }))


def test_published_only_filters(tmp_path):
    _mk(tmp_path, "a", publish=True)
    _mk(tmp_path, "b", publish=False)
    selected = select_studies(tmp_path, published_only=True)
    assert [d.name for d in selected] == ["a"]


def test_ids_filter(tmp_path):
    _mk(tmp_path, "a")
    _mk(tmp_path, "b")
    selected = select_studies(tmp_path, ids=["b"])
    assert [d.name for d in selected] == ["b"]


def test_tag_filter_is_any_match(tmp_path):
    _mk(tmp_path, "a", tags=["baseline"])
    _mk(tmp_path, "b", tags=["exp"])
    _mk(tmp_path, "c", tags=["baseline", "exp"])
    selected = {d.name for d in select_studies(tmp_path, tags=["baseline"])}
    assert selected == {"a", "c"}


def test_best_n_applied_last(tmp_path):
    _mk(tmp_path, "a", publish=True, best_score=0.5)
    _mk(tmp_path, "b", publish=True, best_score=0.9)
    _mk(tmp_path, "c", publish=True, best_score=0.7)
    _mk(tmp_path, "d", publish=False, best_score=1.0)
    selected = [d.name for d in select_studies(tmp_path, published_only=True, best=2)]
    assert selected == ["b", "c"]


def test_task_filter(tmp_path):
    _mk(tmp_path, "a", task="track_a")
    _mk(tmp_path, "b", task="track_b")
    selected = [d.name for d in select_studies(tmp_path, task="track_b")]
    assert selected == ["b"]
