"""Watchdog: stale studies marked FAILED."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.core.models import Study
from lab.core.watchdog import Watchdog


def _seed_study(root: Path, study_id: str) -> None:
    study = Study(
        id=study_id,
        task_name="track_b",
        status="RUNNING",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[],
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )
    study.save(root)


def _touch_log(study_dir: Path, content: str = "{}") -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    log = study_dir / "run.log.jsonl"
    log.write_text(content + "\n", encoding="utf-8")


def test_watchdog_marks_failed_when_log_silent(tmp_path):
    sid = "study_silent_xxxx"
    _seed_study(tmp_path, sid)
    _touch_log(tmp_path / sid, json.dumps({"event": "study_start"}))

    # Backdate the log mtime so it is "too old".
    log = tmp_path / sid / "run.log.jsonl"
    old = time.time() - 999
    import os
    os.utime(log, (old, old))

    wd = Watchdog(sid, tmp_path, timeout_s=10)
    wd.start()
    # Watchdog polls at min(POLL_CEILING, timeout/4) — give it a bit.
    deadline = time.time() + 8
    while time.time() < deadline and not wd.stalled:
        time.sleep(0.2)
    wd.stop()
    assert wd.stalled

    reloaded = Study.load(tmp_path, sid)
    assert reloaded.status == "FAILED"
    last_log = log.read_text().strip().splitlines()[-1]
    payload = json.loads(last_log)
    assert payload["event"] == "error"
    assert payload["fields"]["reason"] == "stalled"


def test_watchdog_idle_when_log_fresh(tmp_path):
    sid = "study_fresh_xxxx"
    _seed_study(tmp_path, sid)
    _touch_log(tmp_path / sid, json.dumps({"event": "study_start"}))

    wd = Watchdog(sid, tmp_path, timeout_s=60)
    wd.start()
    time.sleep(1.0)
    wd.stop()
    assert not wd.stalled

    reloaded = Study.load(tmp_path, sid)
    assert reloaded.status == "RUNNING"


def test_watchdog_does_not_clobber_terminal_status(tmp_path):
    sid = "study_terminal_xxxx"
    study = Study(
        id=sid,
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[],
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )
    study.save(tmp_path)
    _touch_log(tmp_path / sid)
    log = tmp_path / sid / "run.log.jsonl"
    import os
    old = time.time() - 999
    os.utime(log, (old, old))

    wd = Watchdog(sid, tmp_path, timeout_s=10)
    wd.start()
    deadline = time.time() + 8
    while time.time() < deadline and not wd.stalled:
        time.sleep(0.2)
    wd.stop()
    # Watchdog detects stall but must NOT override a terminal status
    reloaded = Study.load(tmp_path, sid)
    assert reloaded.status == "COMPLETED"
