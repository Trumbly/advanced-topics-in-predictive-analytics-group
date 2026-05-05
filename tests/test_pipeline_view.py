"""UI pipeline-state derivation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lab.core.models import (
    Experiment,
    Proposal,
    Study,
    Task,
    TaskError,
    Verdict,
)
from lab.ui.pipeline import (
    PHASES,
    current_step,
    derive_experiment_pipeline,
    derive_study_pipelines,
)


def _proposal() -> Proposal:
    return Proposal(
        architecture_name="Tiny",
        family="cnn_scratch",
        lr=1e-3,
        lr_schedule="cosine",
        epochs=2,
    )


def _exp(**overrides) -> Experiment:
    base = dict(
        id="exp_0000",
        index=0,
        status="PROPOSED",
        proposal=None,
        primary_metric="roc_auc_macro",
    )
    base.update(overrides)
    return Experiment(**base)


def test_phases_constant_in_order():
    assert PHASES == ("propose", "generate", "validate", "execute", "judge")


def test_pending_pipeline_for_fresh_experiment():
    pl = derive_experiment_pipeline(_exp(), is_active=False)
    assert [s for _, s in pl] == ["pending"] * 5


def test_active_experiment_first_pending_marked_running():
    pl = derive_experiment_pipeline(_exp(), is_active=True)
    states = dict(pl)
    assert states["propose"] == "running"
    assert states["generate"] == "pending"


def test_propose_done_after_succeeded_task():
    e = _exp()
    e.tasks.append(Task(name="propose", status="SUCCEEDED"))
    pl = dict(derive_experiment_pipeline(e, is_active=True))
    assert pl["propose"] == "done"
    assert pl["generate"] == "running"  # next pending


def test_validate_failure_reflects_in_state():
    e = _exp()
    e.tasks.append(Task(name="propose", status="SUCCEEDED"))
    e.code = "..."
    e.tasks.append(
        Task(
            name="validate",
            status="FAILED",
            error=TaskError(error_type="UnknownTorchNN", message="x"),
        )
    )
    pl = dict(derive_experiment_pipeline(e, is_active=True))
    # validate is the failed pill (red), generate is now running because the
    # next agent action is a recover→regen.
    assert pl["validate"] == "failed"
    assert pl["generate"] == "running"
    assert pl["execute"] == "pending"  # not jumped ahead


def test_validate_failed_routes_running_back_to_generate():
    """Regression for the bug where validate FAILED made execute pulse."""
    e = _exp(status="VALIDATING")
    e.tasks.append(Task(name="propose", status="SUCCEEDED"))
    e.code = "..."
    e.tasks.append(
        Task(
            name="validate",
            status="FAILED",
            error=TaskError(error_type="SmokeFailed", message="x"),
        )
    )
    pl = dict(derive_experiment_pipeline(e, is_active=True))
    assert pl["execute"] == "pending"
    assert pl["generate"] == "running"


def test_recover_llm_reprompt_routes_to_generate():
    e = _exp(status="VALIDATING")
    e.tasks.append(Task(name="propose", status="SUCCEEDED"))
    e.code = "..."
    e.tasks.append(Task(name="validate", status="FAILED"))
    e.tasks.append(
        Task(
            name="recover",
            status="SUCCEEDED",
            input={"attempt": 0, "kind": "llm_reprompt"},
        )
    )
    pl = dict(derive_experiment_pipeline(e, is_active=True))
    assert pl["generate"] == "running"


def test_recover_autofix_routes_to_validate():
    e = _exp(status="VALIDATING")
    e.tasks.append(Task(name="propose", status="SUCCEEDED"))
    e.code = "..."
    e.tasks.append(Task(name="validate", status="FAILED"))
    e.tasks.append(
        Task(
            name="recover",
            status="SUCCEEDED",
            input={"attempt": 0, "kind": "autofix"},
        )
    )
    pl = dict(derive_experiment_pipeline(e, is_active=True))
    assert pl["validate"] == "running"


def test_validate_succeeded_then_execute_running():
    e = _exp(status="EXECUTING")
    e.tasks.append(Task(name="propose", status="SUCCEEDED"))
    e.code = "..."
    e.tasks.append(Task(name="validate", status="SUCCEEDED"))
    pl = dict(derive_experiment_pipeline(e, is_active=True))
    assert pl["validate"] == "done"
    assert pl["execute"] == "running"


def test_execute_failed_routes_to_generate():
    e = _exp(status="EXECUTING")
    e.tasks.append(Task(name="propose", status="SUCCEEDED"))
    e.code = "..."
    e.tasks.append(Task(name="validate", status="SUCCEEDED"))
    e.tasks.append(Task(name="execute", status="FAILED"))
    pl = dict(derive_experiment_pipeline(e, is_active=True))
    assert pl["execute"] == "failed"
    assert pl["generate"] == "running"


def test_judge_done_when_verdict_present():
    e = _exp(status="JUDGED")
    e.tasks.extend(
        [
            Task(name="propose", status="SUCCEEDED"),
            Task(name="validate", status="SUCCEEDED"),
            Task(name="execute", status="SUCCEEDED"),
        ]
    )
    e.code = "..."
    e.verdict = Verdict(verdict="keep", score=0.5, rationale="ok")
    pl = dict(derive_experiment_pipeline(e, is_active=False))
    assert pl["judge"] == "done"
    assert pl["execute"] == "done"


def test_derive_study_pipelines_marks_only_last_running(tmp_path):
    e0 = _exp(status="JUDGED")
    e0.tasks.extend([Task(name="propose", status="SUCCEEDED")])
    e0.code = "..."
    e0.verdict = Verdict(verdict="keep", score=0.5, rationale="ok")

    e1 = _exp(id="exp_0001", index=1)  # in-flight
    study = Study(
        id="study_x",
        task_name="track_b",
        status="RUNNING",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[e0, e1],
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )
    pipelines = derive_study_pipelines(study)
    # e0 finished — no running
    assert all(s != "running" for _, s in pipelines["exp_0000"])
    # e1 active — has running on first pending phase
    assert any(s == "running" for _, s in pipelines["exp_0001"])


def test_current_step_returns_phase(tmp_path):
    e = _exp()
    e.tasks.append(Task(name="propose", status="SUCCEEDED"))
    study = Study(
        id="study_x",
        task_name="track_b",
        status="RUNNING",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[e],
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )
    cur = current_step(study)
    assert cur == ("exp_0000", "generate")


def test_current_step_none_for_terminal_study():
    e = _exp(status="JUDGED")
    study = Study(
        id="study_x",
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[e],
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )
    assert current_step(study) is None
