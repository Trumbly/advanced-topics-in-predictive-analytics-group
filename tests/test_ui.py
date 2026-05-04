"""I-16 acceptance: FastAPI routes respond + read-mostly dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab.config import load_settings
from lab.core.models import Experiment, Proposal, Study, Verdict
from lab.ui.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


def _study(sid: str = "study_test_xxxx") -> Study:
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
    )


@pytest.fixture
def client(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(
        update={"experiments_dir": str(tmp_path / "studies")}
    )
    s = s.model_copy(update={"paths": new_paths})

    Path(new_paths.experiments_dir).mkdir(parents=True)
    _study().save(Path(new_paths.experiments_dir))

    return TestClient(create_app(s))


def test_studies_list_renders(client):
    r = client.get("/studies")
    assert r.status_code == 200
    assert "study_test_xxxx" in r.text


def test_study_detail_renders_and_no_500_on_completed(client):
    r = client.get("/studies/study_test_xxxx")
    assert r.status_code == 200
    assert "exp_0001" in r.text


def test_experiment_detail_renders(client):
    r = client.get("/experiments/study_test_xxxx/exp_0001")
    assert r.status_code == 200
    assert "EffNetB0" in r.text
    assert "keep" in r.text


def test_404_on_missing_study(client):
    r = client.get("/studies/nope_xxxx")
    assert r.status_code == 404


def test_prompts_dashboard_lists_all_tasks(client):
    r = client.get("/prompts")
    assert r.status_code == 200
    for task in (
        "propose_architecture",
        "generate_code",
        "recover_from_error",
        "analyze_result",
        "judge_experiment",
        "judge_study",
    ):
        assert task in r.text


def test_api_studies_returns_json_list(client):
    r = client.get("/api/studies")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list)
    assert payload[0]["id"] == "study_test_xxxx"


def test_api_study_detail_returns_json(client):
    r = client.get("/api/studies/study_test_xxxx")
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == "study_test_xxxx"
    assert payload["best_experiment_id"] == "exp_0001"


def test_prompts_activate_redirects(client, tmp_path):
    r = client.post(
        "/prompts/propose_architecture/activate",
        params={"version": "v1"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/prompts"


def test_root_redirects_to_studies(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Studies" in r.text or "studies" in r.text
