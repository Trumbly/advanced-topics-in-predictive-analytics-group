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
from datetime import datetime, timezone
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
    agent_status: str | None  # live status from orchestrator.log (active studies only)


@dataclass
class FailureBreakdown:
    """Counts of failed tasks grouped by error_type."""

    total_failed: int
    by_error_type: dict[str, int]


@dataclass
class SubmissionRecord:
    """One entry in the submission history for a study."""

    filename: str
    path: Path
    size_bytes: int
    created_at: str  # ISO string, mtime of the .ipynb file


@dataclass
class RunningExperiment:
    """A snapshot of an experiment that appears to be currently running.

    We detect 'running' from the filesystem, not from a running
    orchestrator process — that way the UI works even if the user
    started the study in a separate terminal. An experiment is
    'running' when either:

      - Its `experiment.json` exists and has `status == "running"`, OR
      - Its sandbox `stdout.log` mtime is within the last
        `_RUNNING_HEARTBEAT_SECONDS` seconds.

    The second rule catches the common case where `experiment.json`
    is written only at the end of the experiment, so the file either
    doesn't exist yet or is stale.
    """

    study_id: str
    experiment_id: str
    stdout_tail: str
    stdout_last_modified_iso: str | None
    seconds_since_last_write: float | None
    agent_status: str  # e.g. "exp_001 → generate_code (llm) ..."


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
    submissions: list[SubmissionRecord]


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


_SCORE_METRIC = "f1_macro"
_LOG_TAIL_LINES = 200
# How fresh the stdout.log mtime has to be for us to consider a
# sandboxed experiment "currently running". Must be comfortably
# larger than the orchestrator heartbeat interval (~10s).
_RUNNING_HEARTBEAT_SECONDS = 60.0


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

        status_str = (
            study.status.value
            if hasattr(study.status, "value")
            else str(study.status)
        )
        # For active/running studies, show what the orchestrator is doing
        agent_status = (
            parse_agent_status(study_dir)
            if status_str in ("active", "running")
            else None
        )

        rows.append(
            StudySummary(
                study_id=study.study_id,
                name=study.name,
                status=status_str,
                hypothesis=study.hypothesis,
                experiment_count=len(study.experiment_ids),
                completed_count=completed,
                failed_count=failed,
                best_experiment_id=study.best_experiment_id,
                best_score=study.best_score,
                created_at=_fmt_dt(study.created_at),
                updated_at=_fmt_dt(study.updated_at),
                has_report=(study_dir / "report" / "report.md").exists(),
                agent_status=agent_status,
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
    submissions = _list_submissions(study_dir)

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
        submissions=submissions,
    )


import re as _re

# Patterns the orchestrator prints to its log. We scan backwards from
# the end to find the LAST matching line — that's the current status.
_RE_EXPERIMENT_HEADER = _re.compile(
    r"──── Experiment (\d+)/(\d+) ────"
)
_RE_TASK_START = _re.compile(
    r"\[(\w+)\] (\w+) \((\w+)\) \.\.\."
)
_RE_TASK_DONE = _re.compile(
    r"[✓✗] (\w+) \("
)
_RE_RECOVERY = _re.compile(
    r"\[(\w+)\] attempting error recovery (\d+)/(\d+)"
)
_RE_PROMOTION = _re.compile(
    r"──── Promoting (\w+)"
)
_RE_STUDY_FINISHED = _re.compile(
    r"Study finished"
)


@dataclass
class AgentProgress:
    """Structured progress info parsed from orchestrator.log."""

    status_line: str  # e.g. "exp_001 → generate_code (llm) ..."
    current_experiment: str | None  # e.g. "exp_001"
    current_experiment_num: int | None  # e.g. 1
    total_experiments: int | None  # e.g. 25
    current_step: str | None  # e.g. "generate_code"
    completed_steps: list[str]  # e.g. ["propose_architecture"]
    all_steps: list[str]  # e.g. ["propose_architecture", "generate_code", ...]
    is_recovery: bool


_PIPELINE_STEPS = [
    "propose_architecture",
    "generate_code",
    "validate_code",
    "execute_training",
    "capture_metrics",
    "analyze_results",
]


def parse_agent_progress(study_dir: Path) -> AgentProgress:
    """Rich progress info from orchestrator.log for the progress bar UI."""
    log_path = study_dir / "orchestrator.log"
    if not log_path.exists():
        return AgentProgress(
            status_line="(starting...)",
            current_experiment=None,
            current_experiment_num=None,
            total_experiments=None,
            current_step=None,
            completed_steps=[],
            all_steps=_PIPELINE_STEPS,
            is_recovery=False,
        )

    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return AgentProgress(
            status_line="(starting...)",
            current_experiment=None,
            current_experiment_num=None,
            total_experiments=None,
            current_step=None,
            completed_steps=[],
            all_steps=_PIPELINE_STEPS,
            is_recovery=False,
        )

    lines = text.splitlines()

    current_exp: str | None = None
    exp_num: int | None = None
    total_exp: int | None = None
    current_step: str | None = None
    completed_steps: list[str] = []
    is_recovery = False
    status_line = "(running...)"

    for line in reversed(lines[-300:]):
        line = line.strip()
        if not line:
            continue

        m = _RE_STUDY_FINISHED.search(line)
        if m:
            status_line = "Study finished"
            break

        m = _RE_PROMOTION.search(line)
        if m and not current_step:
            status_line = f"Promoting {m.group(1)} ..."
            current_step = "promotion"
            break

        m = _RE_RECOVERY.search(line)
        if m and not current_step:
            current_exp = m.group(1)
            is_recovery = True
            status_line = f"{m.group(1)} → error recovery {m.group(2)}/{m.group(3)}"
            current_step = "error_recovery"
            break

        m = _RE_TASK_START.search(line)
        if m and not current_step:
            current_exp = m.group(1)
            current_step = m.group(2)
            status_line = f"{m.group(1)} → {m.group(2)} ({m.group(3)}) ..."
            # Don't break — keep scanning for experiment header

        m = _RE_TASK_DONE.search(line)
        if m and current_step and m.group(1) != current_step:
            # A different task completed before the current one started
            completed_steps.append(m.group(1))

        m = _RE_EXPERIMENT_HEADER.search(line)
        if m:
            exp_num = int(m.group(1))
            total_exp = int(m.group(2))
            if not current_step:
                status_line = f"Experiment {exp_num}/{total_exp} starting..."
            break

    # Build the completed_steps list from the pipeline order
    if current_step and current_step in _PIPELINE_STEPS:
        idx = _PIPELINE_STEPS.index(current_step)
        completed_steps = _PIPELINE_STEPS[:idx]

    return AgentProgress(
        status_line=status_line,
        current_experiment=current_exp,
        current_experiment_num=exp_num,
        total_experiments=total_exp,
        current_step=current_step,
        completed_steps=completed_steps,
        all_steps=_PIPELINE_STEPS,
        is_recovery=is_recovery,
    )


def parse_agent_status(study_dir: Path) -> str:
    """Read the orchestrator.log and extract a one-line status string.

    Returns something like:
      "Experiment 2/25 → generate_code (llm) ..."
      "Experiment 1/5 → execute_training ✓ (12.3s)"
      "Experiment 3/10 → error recovery 1/2"
      "Promoting exp_007 ..."
      "Study finished"
      "(starting...)"

    On any error or missing log, returns "(starting...)".
    """
    log_path = study_dir / "orchestrator.log"
    if not log_path.exists():
        return "(starting...)"
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return "(starting...)"

    lines = text.splitlines()
    if not lines:
        return "(starting...)"

    # Scan backwards — first match wins (most recent status)
    current_experiment = ""
    for line in reversed(lines[-200:]):  # last 200 lines
        line = line.strip()
        if not line:
            continue

        m = _RE_STUDY_FINISHED.search(line)
        if m:
            return "Study finished"

        m = _RE_PROMOTION.search(line)
        if m:
            return f"Promoting {m.group(1)} ..."

        m = _RE_RECOVERY.search(line)
        if m:
            return f"{m.group(1)} → error recovery {m.group(2)}/{m.group(3)}"

        m = _RE_TASK_START.search(line)
        if m:
            exp_id, task_name, task_type = m.groups()
            return f"{exp_id} → {task_name} ({task_type}) ..."

        m = _RE_TASK_DONE.search(line)
        if m:
            # We have a completed task but don't know which experiment
            # unless we find the header. Just report the task name.
            task_name = m.group(1)
            return f"... {task_name} done"

        m = _RE_EXPERIMENT_HEADER.search(line)
        if m:
            return f"Experiment {m.group(1)}/{m.group(2)} starting..."

    return "(running...)"


def find_running_experiment(
    studies_root: Path,
    sandbox_root: Path,
    study_id: str,
) -> RunningExperiment | None:
    """Return a snapshot of the currently-running experiment, if any.

    Walks `sandbox/<study_id>/*/stdout.log`, picks the one with the
    freshest mtime, and — if that mtime is within
    `_RUNNING_HEARTBEAT_SECONDS` of now — returns a `RunningExperiment`
    with a short tail of the log. Returns None when nothing looks live.

    Also respects an `experiment.json` with `status == "running"`: such
    an experiment is always surfaced even if its log is briefly stale.
    """
    sandbox_dir = sandbox_root / study_id
    if not sandbox_dir.exists():
        # Maybe the orchestrator has set status=running but not yet
        # written any sandbox output — fall through to the JSON scan.
        return _find_running_from_experiment_json(studies_root, study_id)

    candidates: list[tuple[float, Path]] = []
    for exp_dir in sandbox_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        log = exp_dir / "stdout.log"
        if not log.exists():
            continue
        try:
            mtime = log.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, exp_dir))

    if not candidates:
        return _find_running_from_experiment_json(studies_root, study_id)

    candidates.sort(reverse=True)  # newest first
    best_mtime, best_exp_dir = candidates[0]

    now = datetime.now(tz=timezone.utc).timestamp()
    age = now - best_mtime

    # Always surface if experiment.json explicitly says running —
    # otherwise require a fresh heartbeat.
    explicit_running = _experiment_status(
        studies_root, study_id, best_exp_dir.name
    ) == "running"
    if not explicit_running and age > _RUNNING_HEARTBEAT_SECONDS:
        return None

    stdout_tail = _tail(best_exp_dir / "stdout.log") or ""
    agent_status = parse_agent_status(studies_root / study_id)
    return RunningExperiment(
        study_id=study_id,
        experiment_id=best_exp_dir.name,
        stdout_tail=stdout_tail,
        stdout_last_modified_iso=datetime.fromtimestamp(
            best_mtime, tz=timezone.utc
        ).isoformat(sep=" ", timespec="seconds"),
        seconds_since_last_write=round(age, 1),
        agent_status=agent_status,
    )


def _find_running_from_experiment_json(
    studies_root: Path, study_id: str
) -> RunningExperiment | None:
    """Fall back: read experiment.json files and return the first one
    whose status is 'running'. Used when the sandbox log is missing or
    hasn't been written yet."""
    exp_root = studies_root / study_id / "experiments"
    if not exp_root.exists():
        return None
    for exp_dir in sorted(exp_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        data = _read_json(exp_dir / "experiment.json")
        if data is None:
            continue
        if data.get("status") == "running":
            return RunningExperiment(
                study_id=study_id,
                experiment_id=exp_dir.name,
                stdout_tail="(no sandbox output yet)",
                stdout_last_modified_iso=None,
                seconds_since_last_write=None,
                agent_status=parse_agent_status(studies_root / study_id),
            )
    return None


def _experiment_status(
    studies_root: Path, study_id: str, experiment_id: str
) -> str | None:
    data = _read_json(
        studies_root
        / study_id
        / "experiments"
        / experiment_id
        / "experiment.json"
    )
    if data is None:
        return None
    return data.get("status")


def read_stdout_tail(
    sandbox_root: Path,
    study_id: str,
    experiment_id: str,
    *,
    max_lines: int = _LOG_TAIL_LINES,
) -> str:
    """Return the last `max_lines` lines of the experiment's stdout.log.
    Empty string when the file is missing — the UI handles that gracefully.
    """
    return (
        _tail(
            sandbox_root / study_id / experiment_id / "stdout.log",
            max_lines=max_lines,
        )
        or ""
    )


def _list_submissions(study_dir: Path) -> list[SubmissionRecord]:
    """Return the submission history for one study, newest first."""
    submissions_dir = study_dir / "submissions"
    if not submissions_dir.exists():
        return []
    records: list[SubmissionRecord] = []
    for path in submissions_dir.glob("*.ipynb"):
        try:
            stat = path.stat()
        except OSError:
            continue
        records.append(
            SubmissionRecord(
                filename=path.name,
                path=path,
                size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(sep=" ", timespec="seconds"),
            )
        )
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records


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
    "SubmissionRecord",
    "RunningExperiment",
    "TaskView",
    "FailureBreakdown",
    "list_studies",
    "load_study_detail",
    "load_experiment_detail",
    "find_running_experiment",
    "read_stdout_tail",
]
