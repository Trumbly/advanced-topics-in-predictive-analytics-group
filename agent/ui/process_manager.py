"""Lightweight process manager for the UI 'Start / Stop' controls.

We don't maintain a long-lived registry of subprocess handles — that
would break on server restart. Instead:

  1. **Start**: spawn `agent start ...` as a detached subprocess with
     stdout/stderr redirected to a log file. Write the PID to a
     `.pid` file inside the study directory.

  2. **Status**: read the `.pid` file and probe liveness with
     `os.kill(pid, 0)`. If the process is gone, clean up the pidfile.

  3. **Stop**: read the `.pid` file, send SIGTERM, wait a short
     grace period, then SIGKILL if necessary.

The log file lives at `<study_dir>/orchestrator.log` and can be tailed
by the UI for a coarse "what is the orchestrator doing" view. The
per-experiment stdout.log (under sandbox/) is the fine-grained view
the live-monitoring HTMX partials already use.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Snapshot of a running study process."""

    pid: int
    study_id: str
    alive: bool


_PIDFILE_NAME = ".orchestrator.pid"
_LOGFILE_NAME = "orchestrator.log"
_STOP_GRACE_SECONDS = 5


def pidfile_path(study_dir: Path) -> Path:
    return study_dir / _PIDFILE_NAME


def logfile_path(study_dir: Path) -> Path:
    return study_dir / _LOGFILE_NAME


def start_study(
    *,
    python_executable: str,
    study_dir: Path,
    name: str,
    hypothesis: str,
    max_experiments: int = 20,
    model: str | None = None,
    repo_root: Path | None = None,
    extra_cli_args: list[str] | None = None,
) -> ProcessInfo:
    """Spawn `agent start ...` as a detached subprocess.

    Returns a `ProcessInfo` with the PID of the new process.
    Raises `RuntimeError` if a process is already running for this
    study directory (detected via the pidfile).
    """
    existing = get_status(study_dir)
    if existing is not None and existing.alive:
        raise RuntimeError(
            f"Study is already running (PID {existing.pid}). "
            f"Stop it first with stop_study()."
        )

    study_dir.mkdir(parents=True, exist_ok=True)
    log_path = logfile_path(study_dir)

    # Build the CLI command. Pass `--study-id` so the CLI uses the
    # EXACT same directory as the one we create for the pidfile + log.
    # Without this, the CLI generates its own study_id (with a UTC
    # timestamp) while the process manager uses the caller's local
    # timestamp → two different directories → study.json lands in the
    # wrong one and the UI loader can't find it.
    cmd: list[str] = [
        python_executable,
        "-m",
        "agent.cli",
        "start",
        "--study",
        name,
        "--study-id",
        study_dir.name,  # force CLI to use our directory name
        "--hypothesis",
        hypothesis,
        "--max-experiments",
        str(max_experiments),
    ]
    if model:
        cmd.extend(["--model", model])
    if extra_cli_args:
        cmd.extend(extra_cli_args)

    log_fh = log_path.open("w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach from our process group
        cwd=str(repo_root or study_dir.parent.parent.parent),
    )

    # Write pidfile.
    pidfile_path(study_dir).write_text(str(proc.pid))
    logger.info("Started study %s (PID %d)", study_dir.name, proc.pid)

    return ProcessInfo(pid=proc.pid, study_id=study_dir.name, alive=True)


def get_status(study_dir: Path) -> ProcessInfo | None:
    """Check if a study process is alive by probing the pidfile.

    Returns None if no pidfile exists. Returns a `ProcessInfo` with
    `alive=False` if the pidfile exists but the process has exited
    (and cleans up the pidfile in that case).
    """
    pf = pidfile_path(study_dir)
    if not pf.exists():
        return None

    try:
        pid = int(pf.read_text().strip())
    except (ValueError, OSError):
        pf.unlink(missing_ok=True)
        return None

    if _is_alive(pid):
        return ProcessInfo(pid=pid, study_id=study_dir.name, alive=True)

    # Process is gone — clean up stale pidfile.
    pf.unlink(missing_ok=True)
    return ProcessInfo(pid=pid, study_id=study_dir.name, alive=False)


def stop_study(study_dir: Path) -> bool:
    """Send SIGTERM to the study process, wait briefly, SIGKILL if needed.

    After the process dies, updates `study.json` to `status: aborted`
    so the UI no longer shows the study as active/running.

    Returns True if the process was successfully stopped. Returns False
    if no running process was found.
    """
    info = get_status(study_dir)
    if info is None or not info.alive:
        return False

    pid = info.pid
    logger.info("Stopping study %s (PID %d)", study_dir.name, pid)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pidfile_path(study_dir).unlink(missing_ok=True)
        _mark_study_aborted(study_dir)
        return True

    # Wait for graceful shutdown.
    for _ in range(int(_STOP_GRACE_SECONDS * 10)):
        if not _is_alive(pid):
            pidfile_path(study_dir).unlink(missing_ok=True)
            _mark_study_aborted(study_dir)
            return True
        time.sleep(0.1)

    # Escalate to SIGKILL.
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    pidfile_path(study_dir).unlink(missing_ok=True)
    _mark_study_aborted(study_dir)
    return True


def _mark_study_aborted(study_dir: Path) -> None:
    """Update study.json status to 'aborted' after killing the process.

    The orchestrator normally sets this itself on graceful shutdown, but
    when we SIGTERM/SIGKILL it from the UI, it dies before it can
    persist the final state. We patch the JSON directly.
    """
    import json as _json

    study_json = study_dir / "study.json"
    if not study_json.exists():
        return
    try:
        data = _json.loads(study_json.read_text())
        if data.get("status") in ("active", "running"):
            data["status"] = "aborted"
            study_json.write_text(_json.dumps(data, indent=2, default=str))
            logger.info("Marked %s as aborted in study.json", study_dir.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not mark study as aborted: %s", exc)


def _is_alive(pid: int) -> bool:
    """Check if a PID is alive (without blocking)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still alive.
        return True


__all__ = [
    "ProcessInfo",
    "start_study",
    "get_status",
    "stop_study",
    "pidfile_path",
    "logfile_path",
]
