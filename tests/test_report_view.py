"""Tests for the auto-build report view (GET /reports/{study_id})."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab.config import load_settings
from lab.core.models import Experiment, Proposal, Study, Verdict
from lab.ui.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


def _study(sid: str = "study_test_rept") -> Study:
    proposal = Proposal(
        architecture_name="EffNetB0",
        family="efficientnet_pretrained",
        lr=3e-4,
        lr_schedule="cosine",
        epochs=2,
    )
    exp = Experiment(
        id="exp_0001",
        index=0,
        status="JUDGED",
        proposal=proposal,
        primary_metric="roc_auc_macro",
        primary_score=0.55,
        history=[{"epoch": 1, "loss": 0.7, "roc_auc_macro": 0.55}],
        verdict=Verdict(verdict="keep", score=0.6, rationale="ok"),
    )
    return Study(
        id=sid,
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[exp],
        best_experiment_id=exp.id,
        best_score=0.55,
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
        llm_model="ollama:gemma4:e4b",
    )


@pytest.fixture(autouse=True)
def _clear_report_builds():
    from lab.ui import app as app_module

    app_module._report_builds.clear()
    yield
    app_module._report_builds.clear()


@pytest.fixture
def studies_root(tmp_path) -> Path:
    root = tmp_path / "studies"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def app_client(studies_root):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(
        update={"experiments_dir": str(studies_root)}
    )
    s = s.model_copy(update={"paths": new_paths})
    _study().save(studies_root)
    return TestClient(create_app(s))


# ---------------------------------------------------------------------------
# Test 1 — existing report is served as HTML
# ---------------------------------------------------------------------------

def test_existing_report_served(studies_root, app_client):
    report_path = studies_root / "study_test_rept" / "report.html"
    report_path.write_text("<html><body>My report content</body></html>", encoding="utf-8")

    r = app_client.get("/reports/study_test_rept")

    assert r.status_code == 200
    assert r.text == "<html><body>My report content</body></html>"


# ---------------------------------------------------------------------------
# Test 2 — missing report kicks off build + shows loading page
# ---------------------------------------------------------------------------

def test_missing_report_triggers_build_and_shows_loading(studies_root, monkeypatch):
    """First GET returns 202 loading page; after build finishes, second GET returns 200."""
    study_dir = studies_root / "study_test_rept"

    def _fake_generate_report(study, settings):
        time.sleep(0.05)
        (study_dir / "report.html").write_text(
            "<html><body>Generated report</body></html>", encoding="utf-8"
        )

    import lab.reporting.generator as gen_mod
    monkeypatch.setattr(gen_mod, "generate_report", _fake_generate_report)

    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"experiments_dir": str(studies_root)})
    s = s.model_copy(update={"paths": new_paths})
    _study().save(studies_root)
    client = TestClient(create_app(s))

    r1 = client.get("/reports/study_test_rept")
    assert r1.status_code == 202
    assert "Building report" in r1.text
    assert 'http-equiv="refresh"' in r1.text

    # Wait for background thread to finish writing report.html
    deadline = time.time() + 5.0
    while not (study_dir / "report.html").exists() and time.time() < deadline:
        time.sleep(0.05)

    r2 = client.get("/reports/study_test_rept")
    assert r2.status_code == 200
    assert "Generated report" in r2.text


# ---------------------------------------------------------------------------
# Test 3 — unknown study returns 404
# ---------------------------------------------------------------------------

def test_unknown_study_returns_404(app_client):
    r = app_client.get("/reports/study_does_not_exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 4 — failed build shows the error banner on next request
# ---------------------------------------------------------------------------

def test_failed_build_shows_error_banner(studies_root, monkeypatch):
    def _failing_generate_report(study, settings):
        raise RuntimeError("synthetic failure")

    import lab.reporting.generator as gen_mod
    monkeypatch.setattr(gen_mod, "generate_report", _failing_generate_report)

    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"experiments_dir": str(studies_root)})
    s = s.model_copy(update={"paths": new_paths})
    _study().save(studies_root)
    client = TestClient(create_app(s))

    # First request kicks off the (failing) build
    r1 = client.get("/reports/study_test_rept")
    assert r1.status_code == 202

    # Wait for the background thread to record the failure
    from lab.ui import app as app_module

    deadline = time.time() + 5.0
    while time.time() < deadline:
        with app_module._report_builds_lock:
            state = app_module._report_builds.get("study_test_rept")
            if state is not None and state.error is not None:
                break
        time.sleep(0.05)

    r2 = client.get("/reports/study_test_rept")
    assert r2.status_code == 500
    assert "synthetic failure" in r2.text


# ---------------------------------------------------------------------------
# Test 5 — concurrent requests don't start a second build
# ---------------------------------------------------------------------------

def test_concurrent_requests_single_build(studies_root, monkeypatch):
    call_count = {"n": 0}

    def _slow_generate_report(study, settings):
        call_count["n"] += 1
        time.sleep(0.5)
        (studies_root / "study_test_rept" / "report.html").write_text(
            "<html><body>done</body></html>", encoding="utf-8"
        )

    import lab.reporting.generator as gen_mod
    monkeypatch.setattr(gen_mod, "generate_report", _slow_generate_report)

    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"experiments_dir": str(studies_root)})
    s = s.model_copy(update={"paths": new_paths})
    _study().save(studies_root)
    client = TestClient(create_app(s))

    # Fire two sequential GETs; the second should observe the in-flight state.
    r1 = client.get("/reports/study_test_rept")
    r2 = client.get("/reports/study_test_rept")

    assert r1.status_code == 202
    assert r2.status_code == 202
    assert call_count["n"] == 1
