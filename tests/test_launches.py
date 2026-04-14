"""Launch manager tests — spawn real subprocesses and verify lifecycle."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from lab.ui import launches


def _long_runner_script(tmp: Path) -> Path:
    """A tiny script that sleeps in a loop and handles SIGINT gracefully.

    We use this instead of the real agent so tests don't need an LLM.
    """
    p = tmp / "runner.py"
    p.write_text(
        "import signal, sys, time\n"
        "print('started', flush=True)\n"
        "def h(*a): print('got sigint', flush=True); sys.exit(0)\n"
        "signal.signal(signal.SIGINT, h)\n"
        "for i in range(30):\n"
        "    print(f'tick {i}', flush=True)\n"
        "    time.sleep(0.2)\n"
    )
    return p


def _spawn_dummy(tmp: Path) -> launches.Launch:
    """Use launches.spawn directly but swap the command via a subclass-like
    trick: we call spawn then monkeypatch the Launch's command after the fact
    isn't possible — so we skip spawn and build a Launch ourselves here.
    """
    raise NotImplementedError  # placeholder — tests below use a lower-level path


def test_spawn_writes_record_and_process_runs(tmp_path):
    """Full round-trip: spawn → list → stop → reconciles."""
    # We monkey-patch only the command builder: spawn internally uses
    # ``sys.executable -m lab run`` which would require the whole env.
    # For a unit test, call the same APIs against a hand-built record.
    script = _long_runner_script(tmp_path)
    launches_dir = tmp_path / "experiments" / "launches"
    launches_dir.mkdir(parents=True)
    log_path = launches_dir / "launch_test.log"
    log_fh = log_path.open("w")

    popen_kwargs = {
        "stdout": log_fh, "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL, "cwd": str(tmp_path),
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        [sys.executable, str(script)], **popen_kwargs,  # type: ignore[arg-type]
    )

    record = launches.Launch(
        id="launch_test",
        pid=proc.pid,
        command=[sys.executable, str(script)],
        task="dummy",
        started_at="2026-04-14T00:00:00+00:00",
        log_path=str(log_path.relative_to(tmp_path)),
    )
    launches._write(record, tmp_path)

    # List sees the record as running
    listed = launches.list_launches(tmp_path)
    assert len(listed) == 1
    assert listed[0].status == "running"

    # Stop sends SIGINT, script exits cleanly
    assert launches.stop_launch("launch_test", tmp_path) is True
    proc.wait(timeout=5)

    # Reconciled status
    refreshed = launches.load_launch("launch_test", tmp_path)
    assert refreshed is not None
    assert refreshed.status == "stopped"


def test_stop_is_idempotent_when_pid_already_dead(tmp_path):
    launches_dir = tmp_path / "experiments" / "launches"
    launches_dir.mkdir(parents=True)
    record = launches.Launch(
        id="launch_dead",
        pid=999999,  # very unlikely to exist
        command=["true"],
        task="dummy",
    )
    launches._write(record, tmp_path)
    assert launches.stop_launch("launch_dead", tmp_path) is True
    # Running → completed reconcile
    refreshed = launches.load_launch("launch_dead", tmp_path)
    assert refreshed.status == "completed"


def test_set_study_id_attaches_to_launch(tmp_path):
    launches_dir = tmp_path / "experiments" / "launches"
    launches_dir.mkdir(parents=True)
    record = launches.Launch(
        id="launch_s", pid=os.getpid(), command=[], task="track_a",
    )
    launches._write(record, tmp_path)
    launches.set_study_id("launch_s", "study_x", tmp_path)
    refreshed = launches.load_launch("launch_s", tmp_path)
    assert refreshed.study_id == "study_x"


def test_launch_for_study_returns_running_only(tmp_path):
    launches_dir = tmp_path / "experiments" / "launches"
    launches_dir.mkdir(parents=True)
    r1 = launches.Launch(id="l1", pid=os.getpid(), command=[], task="t", study_id="s", status="running")
    r2 = launches.Launch(id="l2", pid=999999, command=[], task="t", study_id="s", status="stopped")
    launches._write(r1, tmp_path)
    launches._write(r2, tmp_path)
    found = launches.launch_for_study("s", tmp_path)
    assert found is not None and found.id == "l1"
