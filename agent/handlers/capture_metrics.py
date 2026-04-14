"""Predefined handler: parse the results.json from the previous training run.

Runs after `execute_training` has populated `task.output["results_json_path"]`
(the previous task in the pipeline). Uses the MetricsCollector to parse
the file into a TrainingResults object and attaches it to the output
dict for the orchestrator to read.

If parsing fails (missing file, bad JSON, missing required metrics) the
task is marked FAILED with a `MetricsParseError`-derived TaskError so
the LLM can react.

Whenever we successfully read the raw JSON, we also stash the full file
content in `task.output["raw_results"]`. That way the experiment log
contains everything the training script reported — including the error
message if the script hit its own except branch — without having to
reach into the sandbox directory later.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.metrics import MetricsCollector, MetricsParseError, ScriptReportedError
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
        cfg.get("required_metrics") or ("f1_macro", "loss")
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

    results_path_obj = Path(results_path)
    raw_results = _safe_load_raw(results_path_obj)

    # Embed the raw contents in task.output so the experiment log captures
    # everything the script reported, even when parsing fails.
    task.output = {
        "results_json_path": str(results_path),
        "raw_results": raw_results,
    }

    try:
        training_results = collector.parse_results(results_path_obj)
    except ScriptReportedError as e:
        # The LLM script caught its own exception and wrote {"error": "..."}.
        # Surface that cleanly with the script's message, not a confusing
        # "missing required metrics" error.
        return _fail(task, "ScriptReportedError", e.script_error)
    except MetricsParseError as e:
        return _fail(task, "MetricsParseError", str(e))

    # Add the parsed results on top of the raw payload
    task.output["training_results"] = training_results.model_dump(mode="json")
    task.status = TaskStatus.COMPLETED
    task.completed_at = _now()
    return task


def _safe_load_raw(path: Path) -> dict[str, Any] | str | None:
    """Best-effort load of results.json for logging.

    Returns the parsed dict if possible, a raw text blob if JSON is broken,
    or None if the file does not exist. Never raises.
    """
    if not path.exists():
        return None
    try:
        text = path.read_text()
    except OSError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Truncate very long non-JSON content so we don't bloat the log
        return text[:2000] + ("..." if len(text) > 2000 else "")


def _fail(task: Task, error_type: str, message: str) -> Task:
    task.status = TaskStatus.FAILED
    task.error = TaskError(error_type=error_type, message=message)
    task.completed_at = _now()
    return task


def _now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["run"]
