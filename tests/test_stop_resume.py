"""Stop button + resume routes.

The stop endpoint flips the abort flag on the in-flight runner via the
process-wide registry; the resume endpoint kicks off a new study with
the source as predecessor so memory carries over."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from lab.config import load_settings
from lab.core.lifecycle import _RUNNING_RUNNERS, get_running_runner
from lab.core.models import Study
from lab.ui.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


def _study(sid: str, status: str = "RUNNING") -> Study:
    return Study(
        id=sid,
        task_name="track_b",
        status=status,
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[],
        created_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
    )


@pytest.fixture
def client(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(
        update={"experiments_dir": str(tmp_path / "studies")}
    )
    s = s.model_copy(update={"paths": new_paths})
    Path(new_paths.experiments_dir).mkdir(parents=True)
    yield TestClient(create_app(s)), s
    _RUNNING_RUNNERS.clear()


# ---------- registry ----------


def test_get_running_runner_returns_none_when_unregistered():
    _RUNNING_RUNNERS.clear()
    assert get_running_runner("nope") is None


def test_get_running_runner_returns_registered_instance():
    _RUNNING_RUNNERS.clear()

    class _Stub:
        def abort(self): ...

    stub = _Stub()
    _RUNNING_RUNNERS["study_x"] = stub
    try:
        assert get_running_runner("study_x") is stub
    finally:
        _RUNNING_RUNNERS.clear()


# ---------- stop endpoint ----------


def test_stop_endpoint_calls_abort_on_registered_runner(client):
    c, _s = client
    aborted = []

    class _Stub:
        def abort(self):
            aborted.append(True)

    _RUNNING_RUNNERS["study_running"] = _Stub()
    _study("study_running").save(Path(_s.paths.experiments_dir))

    r = c.post("/studies/study_running/stop", follow_redirects=False)
    assert r.status_code == 303
    assert aborted == [True]


def test_stop_endpoint_marks_stale_running_study_as_aborted(client):
    """Stale UI scenario: study.json says RUNNING but no runner is
    registered (UI was restarted, or CLI-launched study finished without
    the UI noticing). Stop click must NOT 404 — instead flip the saved
    status to ABORTED so the page reload shows the resume button."""
    c, s = client
    _study("study_stale", status="RUNNING").save(Path(s.paths.experiments_dir))

    r = c.post("/studies/study_stale/stop", follow_redirects=False)
    assert r.status_code == 303

    # Reload from disk and confirm the status was flipped.
    reloaded = Study.load(Path(s.paths.experiments_dir), "study_stale")
    assert reloaded.status == "ABORTED"
    assert reloaded.finished_at is not None


def test_stop_endpoint_returns_404_when_study_does_not_exist(client):
    c, _s = client
    r = c.post("/studies/study_unknown/stop")
    assert r.status_code == 404
    assert "study not found" in r.json()["detail"].lower()


def test_stop_endpoint_leaves_finished_study_untouched(client):
    """A study already marked COMPLETED/ABORTED on disk should redirect
    without rewriting the file (no spurious finished_at update)."""
    c, s = client
    finished = _study("study_done", status="COMPLETED")
    finished.save(Path(s.paths.experiments_dir))
    original_finished_at = Study.load(
        Path(s.paths.experiments_dir), "study_done"
    ).finished_at

    r = c.post("/studies/study_done/stop", follow_redirects=False)
    assert r.status_code == 303
    reloaded = Study.load(Path(s.paths.experiments_dir), "study_done")
    assert reloaded.status == "COMPLETED"
    assert reloaded.finished_at == original_finished_at


# ---------- resume endpoint ----------


def test_resume_endpoint_launches_new_study_with_predecessor(client):
    c, s = client
    src = _study("study_done", status="ABORTED")
    src.save(Path(s.paths.experiments_dir))

    launched_args: list = []

    def _fake_cmd_run(args):
        launched_args.append(args)
        return 0

    with patch("lab.cli.cmd_run", _fake_cmd_run):
        r = c.post("/studies/study_done/resume", follow_redirects=False)

    assert r.status_code == 303
    assert r.headers["location"] == "/studies?launched=1"
    # Wait briefly for the daemon thread to call our patched cmd_run.
    import time as _time

    for _ in range(50):
        if launched_args:
            break
        _time.sleep(0.02)
    assert launched_args, "background thread never called cmd_run"
    args = launched_args[0]
    assert args.predecessor == "study_done"
    assert args.task == "track_b"
    assert args.agent_memory is True


def test_resume_endpoint_404_when_source_missing(client):
    c, _s = client
    r = c.post("/studies/missing_id/resume")
    assert r.status_code == 404


# ---------- study page renders the right button ----------


def test_study_page_shows_stop_button_when_running(client):
    c, s = client
    _study("study_running", status="RUNNING").save(Path(s.paths.experiments_dir))
    r = c.get("/studies/study_running")
    assert r.status_code == 200
    assert "Stop study" in r.text
    assert "Resume" not in r.text


def test_study_page_shows_resume_button_when_finished(client):
    c, s = client
    _study("study_done", status="COMPLETED").save(Path(s.paths.experiments_dir))
    r = c.get("/studies/study_done")
    assert r.status_code == 200
    assert "Resume" in r.text
    assert "Stop study" not in r.text
