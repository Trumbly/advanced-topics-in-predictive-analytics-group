"""Pydantic data model for studies, experiments, tasks.

These are task-agnostic: no audio fields, no hardcoded metric names.
Metric scores live in a free-form dict keyed by the metric name declared
in the task config.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums & small value types
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class StudyStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class TaskError(BaseModel):
    error_type: str
    message: str
    traceback: str | None = None


class Task(BaseModel):
    """One step in the experiment pipeline (propose, generate, validate, ...)."""
    name: str
    status: TaskStatus = TaskStatus.PENDING
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    code_used: str | None = None
    error: TaskError | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    prompt_paths: dict[str, str] = Field(default_factory=dict)


class Metric(BaseModel):
    name: str
    value: float


class Experiment(BaseModel):
    id: str = Field(default_factory=lambda: f"exp_{uuid4().hex[:10]}")
    study_id: str
    index: int = 0
    status: ExperimentStatus = ExperimentStatus.PENDING

    architecture_name: str | None = None
    architecture_family: str | None = None
    architecture_proposal: str | None = None
    code: str | None = None

    primary_metric: str = ""
    primary_score: float | None = None
    metrics: dict[str, float|None] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)

    error: TaskError | None = None
    tasks: list[Task] = Field(default_factory=list)

    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    sandbox_path: str | None = None

    # Path (relative to repo root) of the best-epoch checkpoint archived
    # after a successful run. ``None`` for failed/aborted experiments or
    # when the skeleton produced no state_dict. Used by the continue-from-
    # checkpoint path: the next proposal may reference this experiment id
    # and the orchestrator translates it to ``AGENT_INIT_FROM_CHECKPOINT``.
    checkpoint_path: str | None = None


class DatasetProfile(BaseModel):
    """Task-owned dataset profile. Fields depend on the task kind."""
    task_name: str
    kind: str                          # "text_classification_binary", ...
    num_classes: int | None = None
    num_train_samples: int | None = None
    num_val_samples: int | None = None
    num_test_samples: int | None = None
    # Free-form task-specific details.
    extras: dict[str, Any] = Field(default_factory=dict)


class Study(BaseModel):
    id: str = Field(default_factory=lambda: f"study_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:4]}")
    name: str = ""
    task_name: str
    status: StudyStatus = StudyStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    primary_metric: str = ""
    best_experiment_id: str | None = None
    best_score: float | None = None

    experiments: list[Experiment] = Field(default_factory=list)

    # Prompt versions that were active at the time the study ran.
    prompt_template_paths: dict[str, str] = Field(default_factory=dict)

    # Study continuation support
    predecessor_id: str | None = None

    # Selective publishing
    publish: bool = False
    tags: list[str] = Field(default_factory=list)

    # Where the study physically ran. Frozen at study start so the UI
    # can show "ran on local CPU" vs "ran on Kaggle GPU" forever after.
    executor_backend: str = "local"
    executor_infrastructure: dict[str, Any] = Field(default_factory=dict)

    notes: str = ""

    def save(self, experiments_dir: Path) -> Path:
        out = experiments_dir / self.id
        out.mkdir(parents=True, exist_ok=True)
        (out / "study.json").write_text(self.model_dump_json(indent=2))
        return out

    @classmethod
    def load(cls, study_dir: Path) -> "Study":
        return cls.model_validate_json((study_dir / "study.json").read_text())
