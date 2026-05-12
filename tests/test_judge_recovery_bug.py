"""Regression: experiments that recovered from a codegen failure must
not be reported to the judge as failed.

Before this fix the orchestrator persisted one ``Task`` per validate
attempt (status FAILED + TaskError attached on each broken attempt),
plus one final SUCCEEDED validate after the recovery layer patched the
code. ``Judge._experiment_slots`` then walked the task list forward and
surfaced the *first* error to the judge prompt — i.e. the recovered
codegen failure — even when training afterwards completed normally and
produced a real ``primary_score``. The judge's v2 verdict rules treat
any "errored OR score==n/a" run as ``discard``, so a perfectly fine
experiment was being marked as failed in the dashboard.

The fix lives in ``lab.core.memory.terminal_error`` and the call sites
in ``lab.core.judge``: an experiment with a non-None ``primary_score``
no longer surfaces any error to the judge, and failed experiments
report their most recent (terminal) error instead of the first one.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lab.core.judge import _failure_breakdown
from lab.core.memory import _first_error, terminal_error
from lab.core.models import (
    Experiment,
    Proposal,
    Study,
    Task,
    TaskError,
)


def _proposal() -> Proposal:
    return Proposal(
        architecture_name="EffNetB1",
        family="efficientnet_pretrained",
        lr=1e-4,
        lr_schedule="cosine",
        epochs=5,
    )


def _exp_recovered_then_scored(score: float = 0.74) -> Experiment:
    """Realistic task chain: a failed validate attempt, an LLM recover
    that succeeded, a second validate that passed, an execute that
    returned a real score. This is the shape that used to be misread."""
    return Experiment(
        id="exp_recovered",
        index=0,
        status="JUDGED",
        proposal=_proposal(),
        primary_metric="roc_auc_macro",
        primary_score=score,
        tasks=[
            Task(name="propose", status="SUCCEEDED"),
            Task(name="generate", status="SUCCEEDED"),
            Task(
                name="validate",
                status="FAILED",
                input={"attempt": 0},
                error=TaskError(
                    error_type="BadSignature",
                    message="build_model missing num_classes argument",
                ),
            ),
            Task(
                name="recover",
                status="SUCCEEDED",
                input={"attempt": 0, "kind": "search_replace"},
            ),
            Task(name="validate", status="SUCCEEDED", input={"attempt": 1}),
            Task(name="execute", status="SUCCEEDED"),
        ],
        history=[
            {"epoch": 1, "loss": 0.6, "roc_auc_macro": 0.65},
            {"epoch": 2, "loss": 0.4, "roc_auc_macro": score},
        ],
    )


def _exp_failed_after_partial_recovery() -> Experiment:
    """Recovered from one validate failure, then died in execute. The
    terminal error should be the execute error, not the patched
    validate error."""
    return Experiment(
        id="exp_failed",
        index=1,
        status="FAILED",
        proposal=_proposal(),
        primary_metric="roc_auc_macro",
        tasks=[
            Task(name="propose", status="SUCCEEDED"),
            Task(name="generate", status="SUCCEEDED"),
            Task(
                name="validate",
                status="FAILED",
                input={"attempt": 0},
                error=TaskError(
                    error_type="BadSignature",
                    message="early codegen miss",
                ),
            ),
            Task(name="recover", status="SUCCEEDED", input={"attempt": 0}),
            Task(name="validate", status="SUCCEEDED", input={"attempt": 1}),
            Task(
                name="execute",
                status="FAILED",
                error=TaskError(
                    error_type="OOM",
                    message="CUDA out of memory at epoch 1",
                ),
            ),
        ],
    )


def test_terminal_error_ignores_recovered_failures():
    exp = _exp_recovered_then_scored()
    # The forward-scan helper used to return the BadSignature error.
    # Now that the experiment ended with a score, callers must see no
    # error at all.
    assert terminal_error(exp) is None
    assert _first_error(exp) is None  # legacy alias mirrors the fix


def test_terminal_error_returns_last_error_on_real_failure():
    exp = _exp_failed_after_partial_recovery()
    term = terminal_error(exp)
    assert term is not None
    # NOT the BadSignature from the recovered validate attempt.
    assert term.error_type == "OOM"


def test_judge_slot_for_recovered_run_is_none(tmp_path):
    """The judge prompt slot for a recovered-then-scored run must be
    ``(none)`` so the v2 verdict rules pick promote/keep, not discard."""
    from lab.core.judge import Judge
    from lab.core.memory import Memory

    exp = _exp_recovered_then_scored(0.81)
    mem = Memory(top_k=3, recent_failures=2, path=tmp_path / "memory.json")
    slots = Judge._experiment_slots(exp, mem)
    assert slots["error"] == "(none)"
    assert slots["primary_score"] == "0.8100"


def test_judge_slot_for_truly_failed_run_reports_terminal_error(tmp_path):
    from lab.core.judge import Judge
    from lab.core.memory import Memory

    exp = _exp_failed_after_partial_recovery()
    mem = Memory(top_k=3, recent_failures=2, path=tmp_path / "memory.json")
    slots = Judge._experiment_slots(exp, mem)
    assert "OOM" in slots["error"]
    assert "BadSignature" not in slots["error"]
    assert slots["primary_score"] == "n/a"


def test_failure_breakdown_skips_recovered_runs():
    """Study-level failure breakdown must not count experiments that
    recovered + finished with a real score. Pre-fix this returned
    'BadSignature: 1' even though the run produced 0.81 ROC-AUC."""
    study = Study(
        id="study_test",
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[
            _exp_recovered_then_scored(0.81),
            _exp_failed_after_partial_recovery(),
        ],
        created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
    )
    breakdown = _failure_breakdown(study)
    # Only the truly-failed experiment shows up; the recovered one is
    # invisible to the breakdown.
    assert breakdown == "OOM: 1"


def test_failure_breakdown_none_when_all_recovered():
    study = Study(
        id="study_all_ok",
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[
            _exp_recovered_then_scored(0.79),
            _exp_recovered_then_scored(0.83),
        ],
        created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
    )
    assert _failure_breakdown(study) == "(none)"
