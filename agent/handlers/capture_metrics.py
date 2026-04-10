"""Predefined handler: parse the results.json from the previous training run.

Runs after `execute_training` has populated `task.output["results_json_path"]`
(the previous task in the pipeline). Uses the MetricsCollector to parse
the file into a TrainingResults object and attaches it to the output
dict for the orchestrator to read.

If parsing fails (missing file, bad JSON, missing required metrics) the
task is marked FAILED with a `MetricsParseError`-derived TaskError so
the LLM can react.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.metrics import MetricsCollector, MetricsParseError
from agent.models import Task, TaskError, TaskStatus


def run(
    task: Task,
    *,
    config: dict[str, Any] | None = None,
    collector: MetricsCollector | None = None,
    previous_task_output: dict[str, Any] | None = None,
) -> Task:
    """Parse results.json from the preceding training task.

    Args:
        task: the capture_metrics task to update.
        config: pipeline step config (`required_metrics`, etc.)
        collector: optional injected MetricsCollector (tests use this)
        previous_task_output: the output dict from the preceding task
            (typically execute_training). Must contain `results_json_path`.
    """
    task.started_at = _now()
    task.status = TaskStatus.RUNNING

    cfg = config or {}
    required: tuple[str, ...] = tuple(
        cfg.get("required_metrics") or ("roc_auc_macro", "loss")
    )
    collector = collector or MetricsCollector(required_metrics=required)

    prev_output = previous_task_output or {}
    results_path = prev_output.get("results_json_path")

    if not results_path:
        return _fail(
            task,
            "NoResults",
            "Previous task did not produce a results.json path",
        )

    try:
        training_results = collector.parse_results(Path(results_path))
    except MetricsParseError as e:
        return _fail(task, "MetricsParseError", str(e))

    # Store the parsed results for the orchestrator to lift onto the Experiment
    task.output = {
        "results_json_path": str(results_path),
        "training_results": training_results.model_dump(mode="json"),
    }
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
