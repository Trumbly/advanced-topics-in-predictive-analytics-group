"""Predefined handler: execute the validated code via the CodeExecutor.

This handler is a thin adapter between the pipeline step config and the
`CodeExecutor` class. It:

1. Reads `task.code_used` (populated by the preceding `generate_code` step).
2. Instantiates a CodeExecutor from the pipeline `config:` block
   (or uses the injected executor from the orchestrator).
3. Runs the code and stores the ExecutionResult on the Task output.
4. Sets the Task status to COMPLETED, FAILED, or TIMEOUT based on the
   outcome.

The actual TrainingResults parsing happens in the next step (`capture_metrics`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.executor import CodeExecutor, ExecutionResult
from agent.models import Task, TaskError, TaskStatus


def run(
    task: Task,
    *,
    config: dict[str, Any] | None = None,
    executor: CodeExecutor | None = None,
    sandbox_root: Path | str = Path("sandbox"),
) -> Task:
    """Run the code in `task.code_used` via the CodeExecutor."""
    task.started_at = _now()
    task.status = TaskStatus.RUNNING

    code = task.code_used or ""
    if not code.strip():
        task.status = TaskStatus.FAILED
        task.error = TaskError(
            error_type="NoCode",
            message="Task.code_used is empty — nothing to execute",
        )
        task.completed_at = _now()
        return task

    cfg = config or {}
    if executor is None:
        executor = CodeExecutor(
            sandbox_root=Path(sandbox_root),
            timeout_seconds=int(cfg.get("timeout_seconds", 600)),
        )

    result: ExecutionResult = executor.run(code, experiment_id=task.experiment_id)

    task.output = {
        "exit_code": result.exit_code,
        "duration_seconds": result.duration_seconds,
        "workdir": str(result.workdir),
        "results_json_path": (
            str(result.results_json_path) if result.results_json_path else None
        ),
        "timed_out": result.timed_out,
    }

    if result.timed_out:
        task.status = TaskStatus.TIMEOUT
        task.error = result.error
    elif result.succeeded:
        task.status = TaskStatus.COMPLETED
    else:
        task.status = TaskStatus.FAILED
        task.error = result.error

    task.completed_at = _now()
    return task


def _now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["run"]
