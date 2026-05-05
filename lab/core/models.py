"""Pydantic data models for every on-disk artifact (ADR-015).

All models default to `extra="ignore"` so JSON written by an older or newer
version of the codebase still round-trips through `Study.load`.
"""

from __future__ import annotations

import secrets
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums (string literals for stable JSON form)
# ---------------------------------------------------------------------------

TaskName = Literal[
    "propose", "generate", "validate", "execute", "recover", "analyze", "judge"
]
TaskStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "ABORTED"]
ExperimentStatus = Literal[
    "PROPOSED",
    "GENERATING",
    "VALIDATING",
    "EXECUTING",
    "RECOVERING",
    "JUDGED",
    "COMPLETED",
    "FAILED",
    "ABORTED",
]
StudyStatus = Literal["STARTED", "RUNNING", "COMPLETED", "ABORTED", "FAILED"]
LRSchedule = Literal["constant", "cosine", "onecycle"]
Personality = Literal["exploratory", "conservative"]
VerdictKind = Literal["promote", "keep", "discard", "abort_study"]

ErrorType = Literal[
    "OOM",
    "Timeout",
    "ShapeMismatch",
    "ValueError",
    "RuntimeError",
    "FileNotFound",
    "ImportError",
    "ForbiddenImport",
    "BadSignature",
    "UnknownTorchNN",
    "SmokeFailed",
    "Syntax",
    "Other",
]

_FORWARD_COMPAT = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


class StudyNotFoundError(FileNotFoundError):
    """Raised when `Study.load(root, study_id)` cannot find the on-disk artifact."""


# ---------------------------------------------------------------------------
# Task / experiment building blocks
# ---------------------------------------------------------------------------


class TaskError(BaseModel):
    model_config = _FORWARD_COMPAT

    error_type: ErrorType
    message: str
    traceback: str | None = None
    autofix_hint: str | None = None


class ValidationResult(BaseModel):
    model_config = _FORWARD_COMPAT

    ok: bool
    error_type: str | None = None
    message: str | None = None
    findings: list[str] = Field(default_factory=list)
    autofix_hint: str | None = None


class ExecutionResult(BaseModel):
    model_config = _FORWARD_COMPAT

    succeeded: bool
    primary_score: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    error: TaskError | None = None


class DatasetProfile(BaseModel):
    model_config = _FORWARD_COMPAT

    num_classes: int
    num_train: int
    input_tensor_shape: tuple[int, ...]
    class_imbalance: dict[str, int] | None = None


class EDAReport(BaseModel):
    model_config = _FORWARD_COMPAT

    markdown: str
    num_classes: int
    num_train: int
    imbalance_ratio: float
    input_tensor_shape: tuple[int, ...]
    notes: list[str] = Field(default_factory=list)


class Proposal(BaseModel):
    model_config = _FORWARD_COMPAT

    architecture_name: str
    family: str
    lr: float
    lr_schedule: LRSchedule
    epochs: int
    init_from_experiment_id: str | None = None
    weight_decay: float | None = None  # L2 regularization (#40)


class Verdict(BaseModel):
    model_config = _FORWARD_COMPAT

    verdict: VerdictKind
    score: float
    rationale: str = Field(max_length=500)
    suggested_next: str | None = None


class Task(BaseModel):
    model_config = _FORWARD_COMPAT

    name: TaskName
    status: TaskStatus
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    prompt_paths: dict[str, Path] = Field(default_factory=dict)
    error: TaskError | None = None


class Experiment(BaseModel):
    model_config = _FORWARD_COMPAT

    id: str
    index: int
    status: ExperimentStatus
    proposal: Proposal | None = None
    code: str | None = None
    primary_metric: str
    primary_score: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    verdict: Verdict | None = None
    duration_seconds: float | None = None
    sandbox_path: str | None = None
    checkpoint_path: str | None = None


class Study(BaseModel):
    model_config = _FORWARD_COMPAT

    id: str
    task_name: str
    status: StudyStatus
    personality: Personality
    agent_memory_enabled: bool
    predecessor_id: str | None = None
    prompt_template_paths: dict[str, Path] = Field(default_factory=dict)
    experiments: list[Experiment] = Field(default_factory=list)
    best_experiment_id: str | None = None
    best_score: float | None = None
    study_verdict: Verdict | None = None
    created_at: datetime
    finished_at: datetime | None = None
    llm_model: str | None = None  # provider:model used to drive the loop

    # ----- I/O helpers -----

    def save(self, root: Path) -> None:
        out_dir = Path(root) / self.id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "study.json").write_text(
            self.model_dump_json(indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, root: Path, study_id: str) -> "Study":
        path = Path(root) / study_id / "study.json"
        if not path.exists():
            raise StudyNotFoundError(f"study not found: {path}")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Study ID generator
# ---------------------------------------------------------------------------

_ID_ALPHABET = string.ascii_lowercase + string.digits


def new_study_id(now: datetime | None = None) -> str:
    """Return a unique study id of the form ``study_YYYYMMDD_HHMMSS_xxxx``."""
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    suffix = "".join(secrets.choice(_ID_ALPHABET) for _ in range(4))
    return f"study_{ts}_{suffix}"
