"""Per-experiment + per-LLM codegen quality KPIs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab.config import load_settings
from lab.core.codegen_rates import (
    aggregate_model_codegen,
    experiment_rates,
    study_rates,
)
from lab.core.dashboard import compute_kpis
from lab.core.models import (
    Experiment,
    Proposal,
    Study,
    Task,
    TaskError,
    Verdict,
)
from lab.ui.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


def _proposal() -> Proposal:
    return Proposal(
        architecture_name="X",
        family="cnn_scratch",
        lr=1e-3,
        lr_schedule="cosine",
        epochs=2,
    )


def _exp_with_attempts(
    *, validates: list[str], executes: list[str]
) -> Experiment:
    e = Experiment(
        id="exp_x",
        index=0,
        status="JUDGED",
        proposal=_proposal(),
        primary_metric="roc_auc_macro",
        primary_score=0.5,
    )
    for i, status in enumerate(validates):
        e.tasks.append(
            Task(
                name="validate",
                status=status,
                input={"attempt": i},
                error=None
                if status == "SUCCEEDED"
                else TaskError(error_type="SmokeFailed", message="x"),
            )
        )
    for i, status in enumerate(executes):
        e.tasks.append(
            Task(
                name="execute",
                status=status,
                input={"attempt": i},
                error=None
                if status == "SUCCEEDED"
                else TaskError(error_type="ValueError", message="x"),
            )
        )
    return e


# ---- per-experiment rates ----

def test_experiment_rates_first_try_pass():
    e = _exp_with_attempts(
        validates=["SUCCEEDED"], executes=["SUCCEEDED"]
    )
    r = experiment_rates(e)
    assert r.validation_pass_rate == 1.0
    assert r.execution_pass_rate == 1.0
    assert r.validate_first_try_ok is True
    assert r.execute_first_try_ok is True


def test_experiment_rates_with_retries():
    e = _exp_with_attempts(
        validates=["FAILED", "FAILED", "SUCCEEDED"],
        executes=["FAILED", "SUCCEEDED"],
    )
    r = experiment_rates(e)
    assert r.validate_attempts == 3
    assert r.validate_succeeded == 1
    assert r.validation_pass_rate == pytest.approx(1 / 3)
    assert r.validate_first_try_ok is False
    assert r.execute_attempts == 2
    assert r.execution_pass_rate == 0.5
    assert r.execute_first_try_ok is False


def test_experiment_rates_no_attempts():
    e = Experiment(
        id="exp_empty",
        index=0,
        status="PROPOSED",
        proposal=None,
        primary_metric="roc_auc_macro",
    )
    r = experiment_rates(e)
    assert r.validate_attempts == 0
    assert r.validation_pass_rate == 0.0
    assert r.validate_first_try_ok is None


def test_study_rates_returns_one_per_experiment():
    s = Study(
        id="s",
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[
            _exp_with_attempts(validates=["SUCCEEDED"], executes=["SUCCEEDED"]),
            _exp_with_attempts(validates=["FAILED", "SUCCEEDED"], executes=["FAILED"]),
        ],
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )
    rates = study_rates(s)
    assert len(rates) == 2
    assert rates[0].validate_first_try_ok is True
    assert rates[1].validate_first_try_ok is False


# ---- per-model aggregation ----

def _study(sid: str, *, model: str, exps: list[Experiment]) -> Study:
    return Study(
        id=sid,
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=exps,
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
        llm_model=model,
    )


def test_aggregate_model_codegen_groups_by_model(tmp_path):
    _study(
        "s_g4_1",
        model="ollama:gemma4",
        exps=[
            _exp_with_attempts(validates=["FAILED", "SUCCEEDED"], executes=["SUCCEEDED"]),
            _exp_with_attempts(validates=["SUCCEEDED"], executes=["FAILED", "SUCCEEDED"]),
        ],
    ).save(tmp_path)
    _study(
        "s_qw_1",
        model="ollama:qwen3-coder",
        exps=[
            _exp_with_attempts(validates=["SUCCEEDED"], executes=["SUCCEEDED"]),
            _exp_with_attempts(validates=["SUCCEEDED"], executes=["SUCCEEDED"]),
        ],
    ).save(tmp_path)

    table = aggregate_model_codegen(tmp_path)
    assert set(table) == {"ollama:gemma4", "ollama:qwen3-coder"}

    g4 = table["ollama:gemma4"]
    assert g4.experiments == 2
    assert g4.validate_attempts == 3
    assert g4.validate_failed == 1
    assert g4.bad_code_rate == pytest.approx(1 / 3)
    assert g4.first_try_validate_pass_rate == pytest.approx(0.5)  # 1 of 2 exps
    assert g4.execution_failure_rate == pytest.approx(1 / 3)  # 1 fail of 3 attempts

    qw = table["ollama:qwen3-coder"]
    assert qw.bad_code_rate == 0.0
    assert qw.first_try_validate_pass_rate == 1.0
    assert qw.execution_failure_rate == 0.0


def test_aggregate_uses_unknown_when_model_missing(tmp_path):
    _study(
        "s_no_model",
        model=None,  # type: ignore[arg-type]
        exps=[_exp_with_attempts(validates=["SUCCEEDED"], executes=["SUCCEEDED"])],
    ).save(tmp_path)
    table = aggregate_model_codegen(tmp_path)
    assert "unknown" in table


# ---- dashboard wiring ----

def test_dashboard_kpis_include_model_codegen(tmp_path):
    _study(
        "s1",
        model="ollama:gemma4",
        exps=[_exp_with_attempts(validates=["FAILED", "SUCCEEDED"], executes=["SUCCEEDED"])],
    ).save(tmp_path)
    kpis = compute_kpis(tmp_path)
    assert "ollama:gemma4" in kpis.model_codegen
    assert kpis.model_codegen["ollama:gemma4"].validate_attempts == 2


# ---- UI wiring ----

@pytest.fixture
def client(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(
        update={"experiments_dir": str(tmp_path / "studies")}
    )
    s = s.model_copy(update={"paths": new_paths})
    Path(new_paths.experiments_dir).mkdir(parents=True)
    _study(
        "study_test_xxxx",
        model="ollama:gemma4",
        exps=[_exp_with_attempts(validates=["FAILED", "SUCCEEDED"], executes=["SUCCEEDED"])],
    ).save(Path(new_paths.experiments_dir))
    return TestClient(create_app(s))


def test_study_detail_shows_per_experiment_pass_rates(client):
    r = client.get("/studies/study_test_xxxx")
    assert r.status_code == 200
    assert "validate pass rate" in r.text
    # 1 of 2 -> 50%
    assert "50%" in r.text


def test_dashboard_shows_per_model_table(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "Code-generation quality by LLM model" in r.text
    assert "ollama:gemma4" in r.text
    assert "bad-code rate" in r.text
