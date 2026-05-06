"""Dashboard error rate must treat verdict='discard' experiments as
unsuccessful, even when they produced a primary_score.

Before the fix, an experiment that crashed mid-training (channel
mismatch, NaN loss, ...) but happened to write a results.json before
dying still inflated `best_*` and the success-rate KPI. Judge usually
returns 'discard' for those — the dashboard now respects that.
"""

from __future__ import annotations

from datetime import datetime, timezone

from lab.core.dashboard import _is_scored
from lab.core.models import Experiment, Verdict


def _exp(*, status: str, score: float | None, verdict_kind: str | None = None) -> Experiment:
    verdict = (
        Verdict(verdict=verdict_kind, score=0.4, rationale="t")
        if verdict_kind
        else None
    )
    return Experiment(
        id="exp_test",
        index=0,
        status=status,
        primary_metric="roc_auc_macro",
        primary_score=score,
        verdict=verdict,
    )


def test_keep_verdict_counts_as_scored():
    assert _is_scored(_exp(status="JUDGED", score=0.78, verdict_kind="keep")) is True


def test_promote_verdict_counts_as_scored():
    assert _is_scored(_exp(status="JUDGED", score=0.65, verdict_kind="promote")) is True


def test_discard_verdict_does_not_count_as_scored():
    assert _is_scored(_exp(status="JUDGED", score=0.5, verdict_kind="discard")) is False


def test_abort_study_verdict_does_not_count_as_scored():
    assert _is_scored(_exp(status="JUDGED", score=0.5, verdict_kind="abort_study")) is False


def test_failed_status_never_scored():
    assert _is_scored(_exp(status="FAILED", score=0.7, verdict_kind="keep")) is False


def test_no_score_never_scored():
    assert _is_scored(_exp(status="JUDGED", score=None, verdict_kind="keep")) is False


def test_no_verdict_with_score_still_counts():
    """Experiment that produced a score but never got judged
    (e.g. judge crashed). Treat as scored — the discard guard is opt-in."""
    assert _is_scored(_exp(status="JUDGED", score=0.7, verdict_kind=None)) is True
