"""Process lifecycle for UI-launched agent studies.

When the user clicks "Start study" in the dashboard we spawn the CLI
as a detached subprocess, persist a small launch record, and poll for
liveness by PID. Stop sends SIGINT to the process group so the agent
finishes the current experiment before exiting (matches Ctrl+C).

Launch records live under ``experiments/launches/<launch_id>.json`` and
the combined stdout/stderr stream is tailed to ``.log`` next to it.

We track launches (not studies) here because a study id is only known
*after* the orchestrator has started — the CLI passes ``--launch-id``
and the orchestrator writes the study id back into the launch JSON
via :func:`set_study_id` as soon as it is generated.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4


LAUNCH_DIR_REL = "experiments/launches"
_TERMINAL_STUDY_STATUSES = frozenset({"completed", "failed", "aborted"})


@dataclass
class Launch:
    id: str
    pid: int
    command: list[str]
    task: str
    study_id: str | None = None
    started_at: str = ""
    stopped_at: str | None = None
    status: str = "running"        # running | completed | stopped
    log_path: str = ""
    name: str = ""
    tags: list[str] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _launches_dir(repo_root: Path) -> Path:
    p = repo_root / LAUNCH_DIR_REL
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path(launch_id: str, repo_root: Path) -> Path:
    return _launches_dir(repo_root) / f"{launch_id}.json"


def _write(launch: Launch, repo_root: Path) -> None:
    data = asdict(launch)
    _path(launch.id, repo_root).write_text(json.dumps(data, indent=2))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def _study_status(study_id: str, repo_root: Path) -> str | None:
    """Best-effort load of `study.json` status for launch reconciliation."""
    path = repo_root / "experiments" / "studies" / study_id / "study.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    status = data.get("status")
    if not isinstance(status, str):
        return None
    return status.strip().lower()


def _refresh_status(launch: Launch, repo_root: Path) -> str:
    """Turn a stored ``running`` into ``completed`` when the PID is gone.
    Terminal states (``completed``, ``stopped``) are returned as-is."""
    if launch.status != "running":
        return launch.status
    # Primary source of truth: if the linked study has a terminal status,
    # the launch should no longer appear as running even if the PID check
    # is stale (PID reuse) or a post-processing step kept the process alive.
    if launch.study_id:
        s = _study_status(launch.study_id, repo_root)
        if s in _TERMINAL_STUDY_STATUSES:
            return "completed"
    return "running" if _pid_alive(launch.pid) else "completed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def spawn(
    repo_root: Path,
    *,
    task: str,
    name: str = "",
    tags: str = "",
    predecessor: str = "",
    report: bool = True,
    prompt_overrides: dict[str, str] | None = None,
    executor_backend: str = "",
    primary_metrics: str = "",
) -> Launch:
    launch_id = (
        "launch_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_" + uuid4().hex[:4]
    )
    ldir = _launches_dir(repo_root)
    log_path = ldir / f"{launch_id}.log"

    cmd: list[str] = [sys.executable, "-m", "lab"]
    if task:
        cmd.extend(["--task", task])
    cmd.append("run")
    if name:
        cmd.extend(["--name", name])
    if tags:
        cmd.extend(["--tags", tags])
    if predecessor:
        cmd.extend(["--resume", predecessor])
    if report:
        cmd.append("--report")
    cmd.extend(["--launch-id", launch_id])
    if executor_backend:
        cmd.extend(["--executor", executor_backend])
    if primary_metrics.strip():
        cmd.extend(["--primary-metrics", primary_metrics.strip()])
    for k, v in (prompt_overrides or {}).items():
        cmd.extend(["--prompt", f"{k}={v}"])

    # Detached subprocess, with stdout+stderr merged into the launch log.
    # start_new_session makes the child the leader of its own process group
    # so SIGINT can later be sent to the whole tree — same trick the
    # executor uses for the sandbox.
    log_fh = log_path.open("w")
    popen_kwargs: dict[str, object] = {
        "cwd": str(repo_root),
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    launch = Launch(
        id=launch_id,
        pid=proc.pid,
        command=cmd,
        task=task,
        name=name,
        tags=tag_list,
        started_at=_now_iso(),
        log_path=str(log_path.relative_to(repo_root)),
    )
    _write(launch, repo_root)
    return launch


def list_launches(repo_root: Path) -> list[Launch]:
    out: list[Launch] = []
    for p in sorted(_launches_dir(repo_root).glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        try:
            launch = Launch(**data)
        except TypeError:
            # Schema drift — skip rather than crash.
            continue
        refreshed = _refresh_status(launch, repo_root)
        if refreshed != launch.status:
            launch.status = refreshed
            try:
                _write(launch, repo_root)
            except OSError:
                pass
        out.append(launch)
    return out


def load_launch(launch_id: str, repo_root: Path) -> Launch | None:
    p = _path(launch_id, repo_root)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        launch = Launch(**data)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    refreshed = _refresh_status(launch, repo_root)
    if refreshed != launch.status:
        launch.status = refreshed
        try:
            _write(launch, repo_root)
        except OSError:
            pass
    return launch


def stop_launch(launch_id: str, repo_root: Path) -> bool:
    launch = load_launch(launch_id, repo_root)
    if launch is None:
        return False
    if not _pid_alive(launch.pid):
        # Already dead — reconcile the stored status and succeed idempotently.
        if launch.status == "running":
            launch.status = "completed"
            _write(launch, repo_root)
        return True
    try:
        if os.name == "nt":
            os.kill(launch.pid, signal.SIGTERM)
        else:
            # Graceful: orchestrator catches SIGINT, finishes the current
            # experiment, then exits cleanly. SIGTERM would hard-kill.
            os.killpg(os.getpgid(launch.pid), signal.SIGINT)
    except (OSError, ProcessLookupError):
        return False
    launch.stopped_at = _now_iso()
    launch.status = "stopped"
    _write(launch, repo_root)
    return True


def set_study_id(launch_id: str, study_id: str, repo_root: Path) -> None:
    """Called by the agent subprocess once it has a study id.

    Best-effort — a missing launch record just means the user launched
    from the CLI, which is fine.
    """
    launch = load_launch(launch_id, repo_root)
    if launch is None:
        return
    launch.study_id = study_id
    _write(launch, repo_root)


def active_launches(repo_root: Path) -> list[Launch]:
    return [l for l in list_launches(repo_root) if l.status == "running"]


def launch_for_study(study_id: str, repo_root: Path) -> Launch | None:
    for l in list_launches(repo_root):
        if l.study_id == study_id and l.status == "running":
            return l
    return None


__all__ = [
    "Launch", "spawn", "stop_launch", "list_launches", "load_launch",
    "active_launches", "launch_for_study", "set_study_id",
]
