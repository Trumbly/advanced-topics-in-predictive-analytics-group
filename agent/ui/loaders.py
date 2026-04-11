"""Read-only disk access for the UI.

Every function here takes a `studies_root` argument — by convention
`experiments/studies/` — and reads fully-materialized Pydantic models
from the JSON files the orchestrator/logger wrote during a run. We
deliberately never write to disk from the UI layer.

Schema reminder (from `agent.logger.ExperimentLogger`):

    experiments/studies/<study_id>/
        study.json
        study.md
        memory.json
        memory.md
        experiments/
            <experiment_id>/
                experiment.json
                experiment.md
                tasks/
                    <task_id>.json
                    <task_id>.md
        report/
            report.md
            figures/*.png

    sandbox/<study_id>/<experiment_id>/
        code.py
        stdout.log
        stderr.log
        results.json

The loaders below surface this as typed `StudySummary`, `ExperimentCard`,
`TaskView` dataclasses so the template layer doesn't poke at raw dicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.models import Experiment, Study, Task


# ---------------------------------------------------------------------------
# Value types returned to templates
# ---------------------------------------------------------------------------


@dataclass
class StudySummary:
    """Index-page row for one study."""

    study_id: str
    name: str
    status: str
    hypothesis: str
    experiment_count: int
    completed_count: int
    failed_count: int
    best_experiment_id: str | None
    best_score: float | None
    created_at: str
    updated_at: str
    has_report: bool


@dataclass
class FailureBreakdown:
    """Counts of failed tasks grouped by error_type."""

    total_failed: int
    by_error_type: dict[str, int]


@dataclass
class StudyDetail:
    """Everything the study detail page needs."""

    study: Study
    experiments: list[Experiment]
    completed_count: int
    failed_count: int
    score_metric: str
    score_progression: list[dict[str, Any]]  # [{exp_id, score, status}]
    failure_breakdown: FailureBreakdown
    has_report: bool
    report_path: Path | None


@dataclass
class TaskView:
    """One row in the experiment detail task chain."""

    task_id: str
    task_name: str
    task_type: str
    status: str
    error_type: str | None
    error_message: str | None
    llm_response: str | None
    code_used: str | None
    prompt_used: str | None
    duration_seconds: float | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentDetail:
    """Everything the experiment detail page needs."""

    study_id: str
    experiment: Experiment
    tasks: list[TaskView]
    code: str | None
    stdout_tail: str | None
    stderr_tail: str | None
    training_curves: dict[str, list[float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SCORE_METRIC = "roc_auc_macro"
_LOG_TAIL_LINES = 200


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _tail(path: Path, *, max_lines: int = _LOG_TAIL_LINES) -> str | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "...[truncated]...\n" + "\n".join(lines[-max_lines:])


def _fmt_dt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return value.isoformat(sep=" ", timespec="seconds")
    except Exception:  # noqa: BLE001
        return str(value)


# ---------------------------------------------------------------------------
# Index: list of studies
# ---------------------------------------------------------------------------


def list_studies(studies_root: Path) -> list[StudySummary]:
    """Return a list of `StudySummary` sorted newest-first.

    Any directory under `studies_root` that contains a parseable
    `study.json` shows up. Broken directories are skipped silently.
    """
    if not studies_root.exists():
        return []

    rows: list[StudySummary] = []
    for study_dir in sorted(studies_root.iterdir(), reverse=True):
        if not study_dir.is_dir():
            continue
        data = _read_json(study_dir / "study.json")
        if data is None:
            continue
        try:
            study = Study.model_validate(data)
        except Exception:  # noqa: BLE001
            continue

        completed = 0
        failed = 0
        experiments_dir = study_dir / "experiments"
        if experiments_dir.exists():
            for exp_dir in experiments_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                exp_data = _read_json(exp_dir / "experiment.json")
                if exp_data is None:
                    continue
                status = exp_data.get("status", "")
                if status == "completed":
                    completed += 1
                elif status == "failed" or status == "timeout":
                    failed += 1

        rows.append(
            StudySummary(
                study_id=study.study_id,
                name=study.name,
                status=(
                    study.status.value
                    if hasattr(study.status, "value")
                    else str(study.status)
                ),
                hypothesis=study.hypothesis,
                experiment_count=len(study.experiment_ids),
                completed_count=completed,
                failed_count=failed,
                best_experiment_id=study.best_experiment_id,
                best_score=study.best_score,
                created_at=_fmt_dt(study.created_at),
                updated_at=_fmt_dt(study.updated_at),
                has_report=(study_dir / "report" / "report.md").exists(),
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Study detail
# ---------------------------------------------------------------------------


def _load_experiment(exp_dir: Path) -> Experiment | None:
    data = _read_json(exp_dir / "experiment.json")
    if data is None:
        return None
    try:
        return Experiment.model_validate(data)
    except Exception:  # noqa: BLE001
        return None


def _failure_breakdown(
    study_dir: Path, experiments: list[Experiment]
) -> FailureBreakdown:
    """Walk every task file and bucket failures by error_type."""
    counts: dict[str, int] = {}
    total = 0
    exp_root = study_dir / "experiments"
    for exp in experiments:
        tasks_dir = exp_root / exp.experiment_id / "tasks"
        if not tasks_dir.exists():
            continue
        for task_file in tasks_dir.glob("*.json"):
            task_data = _read_json(task_file)
            if task_data is None:
                continue
            if task_data.get("status") != "failed":
                continue
            err = task_data.get("error") or {}
            err_type = err.get("error_type") or "Unknown"
            counts[err_type] = counts.get(err_type, 0) + 1
            total += 1
    return FailureBreakdown(total_failed=total, by_error_type=counts)


def load_study_detail(
    studies_root: Path, study_id: str
) -> StudyDetail | None:
    """Return everything needed to render the study detail page."""
    study_dir = studies_root / study_id
    study_data = _read_json(study_dir / "study.json")
    if study_data is None:
        return None
    try:
        study = Study.model_validate(study_data)
    except Exception:  # noqa: BLE001
        return None

    experiments: list[Experiment] = []
    exp_root = study_dir / "experiments"
    if exp_root.exists():
        for exp_id in study.experiment_ids:
            exp = _load_experiment(exp_root / exp_id)
            if exp is not None:
                experiments.append(exp)

    score_progression: list[dict[str, Any]] = []
    running_best: float | None = None
    completed_count = 0
    failed_count = 0
    for exp in experiments:
        status_str = (
            exp.status.value
            if hasattr(exp.status, "value")
            else str(exp.status)
        )
        if status_str == "completed":
            completed_count += 1
        elif status_str in ("failed", "timeout"):
            failed_count += 1

        score = None
        if exp.results and exp.results.metrics:
            score = exp.results.metrics.get(_SCORE_METRIC)
        if score is not None:
            running_best = score if running_best is None else max(running_best, score)
        score_progression.append(
            {
                "experiment_id": exp.experiment_id,
                "score": score,
                "best_so_far": running_best,
                "status": status_str,
                "architecture": (
                    exp.config.architecture if exp.config else None
                ),
            }
        )

    failure_breakdown = _failure_breakdown(study_dir, experiments)
    report_path = study_dir / "report" / "report.md"

    return StudyDetail(
        study=study,
        experiments=experiments,
        completed_count=completed_count,
        failed_count=failed_count,
        score_metric=_SCORE_METRIC,
        score_progression=score_progression,
        failure_breakdown=failure_breakdown,
        has_report=report_path.exists(),
        report_path=report_path if report_path.exists() else None,
    )


# ---------------------------------------------------------------------------
# Experiment detail
# ---------------------------------------------------------------------------


def load_experiment_detail(
    studies_root: Path,
    sandbox_root: Path,
    study_id: str,
    experiment_id: str,
) -> ExperimentDetail | None:
    study_dir = studies_root / study_id
    exp_dir = study_dir / "experiments" / experiment_id
    exp = _load_experiment(exp_dir)
    if exp is None:
        return None

    tasks_dir = exp_dir / "tasks"
    tasks: list[TaskView] = []
    if tasks_dir.exists():
        # Sort by task_id so the task chain renders in order.
        for task_file in sorted(tasks_dir.glob("*.json")):
            task_data = _read_json(task_file)
            if task_data is None:
                continue
            try:
                task = Task.model_validate(task_data)
            except Exception:  # noqa: BLE001
                continue
            err = task.error
            duration = None
            if task.started_at and task.completed_at:
                duration = (
                    task.completed_at - task.started_at
                ).total_seconds()
            tasks.append(
                TaskView(
                    task_id=task.task_id,
                    task_name=task.task_name or "?",
                    task_type=(
                        task.task_type.value
                        if hasattr(task.task_type, "value")
                        else str(task.task_type)
                    ),
                    status=(
                        task.status.value
                        if hasattr(task.status, "value")
                        else str(task.status)
                    ),
                    error_type=err.error_type if err else None,
                    error_message=err.message if err else None,
                    llm_response=task.llm_response,
                    code_used=task.code_used,
                    prompt_used=task.prompt_used,
                    duration_seconds=duration,
                    extra=task.output or {},
                )
            )

    sandbox_dir = sandbox_root / study_id / experiment_id
    code_path = sandbox_dir / "code.py"
    code: str | None = None
    if code_path.exists():
        try:
            code = code_path.read_text(errors="replace")
        except OSError:
            code = None

    stdout_tail = _tail(sandbox_dir / "stdout.log")
    stderr_tail = _tail(sandbox_dir / "stderr.log")

    training_curves: dict[str, list[float]] = {}
    if exp.results and exp.results.training_curves:
        training_curves = {
            key: [float(v) for v in values]
            for key, values in exp.results.training_curves.items()
        }

    return ExperimentDetail(
        study_id=study_id,
        experiment=exp,
        tasks=tasks,
        code=code,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        training_curves=training_curves,
    )


__all__ = [
    "StudySummary",
    "StudyDetail",
    "ExperimentDetail",
    "TaskView",
    "FailureBreakdown",
    "list_studies",
    "load_study_detail",
    "load_experiment_detail",
]
