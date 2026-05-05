"""dashboard KPI aggregation + UI route."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab.config import load_settings
from lab.core.dashboard import compute_kpis
from lab.core.models import (
    Experiment,
    Proposal,
    Study,
    Task,
    TaskError,
)
from lab.ui.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


def _proposal(name: str, family: str = "cnn_scratch") -> Proposal:
    return Proposal(
        architecture_name=name,
        family=family,
        lr=1e-3,
        lr_schedule="cosine",
        epochs=2,
    )


def _success(idx: int, score: float, *, family: str = "cnn_scratch", arch: str = "Tiny") -> Experiment:
    return Experiment(
        id=f"exp_ok_{idx:04d}",
        index=idx,
        status="JUDGED",
        proposal=_proposal(arch, family),
        primary_metric="roc_auc_macro",
        primary_score=score,
        metrics={"roc_auc_macro": score, "f1_macro": score - 0.05},
    )


def _failure(idx: int) -> Experiment:
    return Experiment(
        id=f"exp_fail_{idx:04d}",
        index=idx,
        status="FAILED",
        proposal=_proposal("Bad"),
        primary_metric="roc_auc_macro",
        tasks=[
            Task(
                name="execute",
                status="FAILED",
                error=TaskError(error_type="ShapeMismatch", message="x"),
            )
        ],
    )


def _study(
    sid: str,
    experiments: list[Experiment],
    *,
    status: str = "COMPLETED",
    propose_version: str = "v1",
) -> Study:
    return Study(
        id=sid,
        task_name="track_b",
        status=status,
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=experiments,
        prompt_template_paths={
            "propose_architecture": Path(f"config/prompts/propose_architecture/{propose_version}.yaml"),
            "generate_code": Path(f"config/prompts/generate_code/{propose_version}.yaml"),
            "recover_from_error": Path("config/prompts/recover_from_error/v1.yaml"),
            "analyze_result": Path("config/prompts/analyze_result/v1.yaml"),
            "judge_experiment": Path("config/prompts/judge_experiment/v1.yaml"),
            "judge_study": Path("config/prompts/judge_study/v1.yaml"),
        },
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )


# ---- KPI math ----

def test_compute_kpis_empty(tmp_path):
    kpis = compute_kpis(tmp_path)
    assert kpis.total_studies == 0
    assert kpis.total_experiments == 0
    assert kpis.experiment_error_rate == 0.0
    assert kpis.study_error_rate == 0.0
    assert kpis.best_roc_auc is None
    assert kpis.best_model is None
    assert kpis.best_prompts == {}


def test_compute_kpis_counts_and_rates(tmp_path):
    _study("s1", [_success(0, 0.5), _success(1, 0.7), _failure(2)]).save(tmp_path)
    _study("s2", [_failure(0)], status="FAILED").save(tmp_path)
    kpis = compute_kpis(tmp_path)
    assert kpis.total_studies == 2
    assert kpis.total_experiments == 4
    # 2 failures of 4 experiments
    assert kpis.experiment_error_rate == pytest.approx(0.5)
    # s2 is FAILED + has no scored experiments → error
    assert kpis.study_error_rate == pytest.approx(0.5)


def test_compute_kpis_best_metrics_and_model(tmp_path):
    _study(
        "s1",
        [
            _success(0, 0.55, family="cnn_scratch", arch="Tiny"),
            _success(1, 0.82, family="efficientnet_pretrained", arch="EffNetB0"),
        ],
    ).save(tmp_path)
    kpis = compute_kpis(tmp_path)
    assert kpis.best_roc_auc == pytest.approx(0.82)
    assert kpis.best_f1 == pytest.approx(0.77)
    assert kpis.best_model.architecture_name == "EffNetB0"
    assert kpis.best_model.family == "efficientnet_pretrained"
    assert kpis.best_model.experiment_id == "exp_ok_0001"


def test_compute_kpis_best_prompts_per_task(tmp_path):
    # Two studies on v1, two on v2 with higher scores → v2 wins for the
    # tasks that switched, v1 for the tasks that stayed.
    for i in range(3):
        _study(f"s_v1_{i}", [_success(0, 0.40), _success(1, 0.42)], propose_version="v1").save(tmp_path)
    for i in range(3):
        _study(f"s_v2_{i}", [_success(0, 0.70), _success(1, 0.72)], propose_version="v2").save(tmp_path)

    kpis = compute_kpis(tmp_path, min_runs=3)
    propose = kpis.best_prompts["propose_architecture"]
    versions = {v.version: v for v in propose.versions}
    assert set(versions) == {"v1", "v2"}
    assert versions["v2"].mean > versions["v1"].mean
    assert propose.best_version == "v2"

    # recover_from_error stayed on v1 in both → only v1 listed
    recover = kpis.best_prompts["recover_from_error"]
    assert {v.version for v in recover.versions} == {"v1"}


def test_compute_kpis_min_runs_filters(tmp_path):
    _study("s_lucky", [_success(0, 0.99)], propose_version="v9").save(tmp_path)
    for i in range(4):
        _study(f"s_v1_{i}", [_success(0, 0.5)]).save(tmp_path)
    kpis = compute_kpis(tmp_path, min_runs=3)
    propose = kpis.best_prompts["propose_architecture"]
    # v9 has 1 run → not eligible as best despite 0.99
    assert propose.best_version == "v1"


# ---- UI route ----

@pytest.fixture
def client(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"experiments_dir": str(tmp_path / "studies")})
    s = s.model_copy(update={"paths": new_paths})
    Path(new_paths.experiments_dir).mkdir(parents=True)
    _study("s1", [_success(0, 0.6), _success(1, 0.7), _failure(2)]).save(Path(new_paths.experiments_dir))
    return TestClient(create_app(s))


def test_dashboard_html_route_renders(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "Dashboard" in r.text
    assert "Total studies" in r.text
    assert "Best ROC-AUC" in r.text
    assert "Best prompts" in r.text


def test_dashboard_api_route_returns_kpis(client):
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    data = r.json()
    assert data["total_studies"] == 1
    assert data["total_experiments"] == 3
    assert data["best_model"]["architecture_name"] == "Tiny"
    assert "best_prompts" in data


def test_dashboard_nav_link_in_base(client):
    """The nav on every page exposes /dashboard."""
    r = client.get("/studies")
    assert 'href="/dashboard"' in r.text
