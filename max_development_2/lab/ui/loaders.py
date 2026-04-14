"""Read-only helpers: walk ``experiments/studies/*`` and load studies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.core.models import Study


@dataclass
class StudySummary:
    id: str
    name: str
    task_name: str
    status: str
    created_at: str | None
    primary_metric: str
    best_score: float | None
    experiments_count: int
    publish: bool
    tags: list[str]
    predecessor_id: str | None = None


def iter_studies(experiments_dir: Path) -> list[StudySummary]:
    if not experiments_dir.exists():
        return []
    out: list[StudySummary] = []
    for d in sorted(experiments_dir.iterdir(), reverse=True):
        sj = d / "study.json"
        if not sj.exists():
            continue
        try:
            data = json.loads(sj.read_text())
        except json.JSONDecodeError:
            continue
        out.append(StudySummary(
            id=data.get("id", d.name),
            name=data.get("name", d.name),
            task_name=data.get("task_name", ""),
            status=data.get("status", "unknown"),
            created_at=data.get("created_at"),
            primary_metric=data.get("primary_metric", ""),
            best_score=data.get("best_score"),
            experiments_count=len(data.get("experiments", [])),
            publish=bool(data.get("publish", False)),
            tags=list(data.get("tags", [])),
            predecessor_id=data.get("predecessor_id"),
        ))
    return out


def load_study(experiments_dir: Path, study_id: str) -> Study | None:
    study_dir = experiments_dir / study_id
    if not (study_dir / "study.json").exists():
        return None
    return Study.load(study_dir)


def save_study(study: Study, experiments_dir: Path) -> None:
    study.save(experiments_dir)


def load_experiment_raw(experiments_dir: Path, study_id: str, exp_id: str) -> dict[str, Any] | None:
    study = load_study(experiments_dir, study_id)
    if not study:
        return None
    for e in study.experiments:
        if e.id == exp_id:
            return e.model_dump(mode="json")
    return None


def read_live_stdout(sandbox_root: Path, exp_id: str, *, tail: int = 400) -> str:
    path = sandbox_root / exp_id / "stdout.log"
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-tail:])


__all__ = [
    "StudySummary",
    "iter_studies",
    "load_study",
    "save_study",
    "load_experiment_raw",
    "read_live_stdout",
]
