"""Predefined handler: export the current experiment as a Kaggle notebook.

This handler is meant to be invoked outside the main pipeline (e.g. after
the orchestrator loop completes) or from the CLI `submit` command.
It wraps the SubmissionExporter with the Task/TaskError interface the
orchestrator expects from predefined handlers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.models import Experiment, Study, Task, TaskError, TaskStatus
from agent.submission import (
    NoBestExperimentError,
    SubmissionExporter,
    SubmissionValidationError,
)


def run(
    task: Task,
    *,
    config: dict[str, Any] | None = None,
    study: Study | None = None,
    experiment: Experiment | None = None,
    code: str | None = None,
    output_path: Path | None = None,
) -> Task:
    """Wrap SubmissionExporter.export for the orchestrator's task interface."""
    task.started_at = _now()
    task.status = TaskStatus.RUNNING

    if study is None or experiment is None or code is None or output_path is None:
        return _fail(
            task,
            "MissingArguments",
            "generate_submission handler requires study, experiment, code, and output_path",
        )

    cfg = config or {}
    exporter = SubmissionExporter(
        target_runtime_seconds=int(cfg.get("target_runtime_seconds", 5400)),
        cpu_only=bool(cfg.get("cpu_only", True)),
    )

    try:
        written = exporter.export(
            study=study,
            experiment=experiment,
            code=code,
            output_path=output_path,
        )
    except (NoBestExperimentError, SubmissionValidationError) as exc:
        return _fail(task, type(exc).__name__, str(exc))

    task.output = {"submission_path": str(written)}
    task.status = TaskStatus.COMPLETED
    task.completed_at = _now()
    return task


def _fail(task: Task, error_type: str, message: str) -> Task:
    task.status = TaskStatus.FAILED
    task.error = TaskError(error_type=error_type, message=message)
    task.completed_at = _now()
    return task


def _now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["run"]
