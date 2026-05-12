"""Per-experiment submission: build artifacts for a chosen experiment
instead of the study's best one.

The user wants to keep "build submission" defaulting to the best
experiment of a study (the historical behaviour) while also being able
to submit any successful experiment individually — e.g. a non-best run
of a different family that they prefer manually. Failed experiments
must not be submittable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.config import Settings, load_settings
from lab.core.models import (
    Experiment,
    Proposal,
    Study,
    Task,
    TaskError,
)
from lab.submission.builder import (
    SubmissionValidationError,
    _resolve_target_experiment,
    _submission_out_dir,
    build_submission_for_study,
)


_BUILD_BLOCK = """
import torch
import torch.nn as nn

def build_model(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(1, num_classes),
    )
""".strip()


def _proposal(name: str = "arch_a") -> Proposal:
    return Proposal(
        architecture_name=name,
        family="cnn_scratch",
        lr=1e-3,
        lr_schedule="constant",
        epochs=2,
    )


def _success_exp(exp_id: str, score: float, arch: str = "arch_a") -> Experiment:
    code = (
        "# --- AGENT_BUILD_MODEL_START ---\n"
        + _BUILD_BLOCK
        + "\n# --- AGENT_BUILD_MODEL_END ---\n"
    )
    return Experiment(
        id=exp_id,
        index=0,
        status="JUDGED",
        proposal=_proposal(arch),
        primary_metric="roc_auc_macro",
        primary_score=score,
        code=code,
        tasks=[Task(name="execute", status="SUCCEEDED")],
    )


def _failed_exp(exp_id: str) -> Experiment:
    """Failed experiment that nonetheless reached the execute stage
    (so it has code on disk). Failure manifests as ``primary_score is
    None`` + a TaskError on execute. Resolving such an id must be
    rejected for submission."""
    code = (
        "# --- AGENT_BUILD_MODEL_START ---\n"
        + _BUILD_BLOCK
        + "\n# --- AGENT_BUILD_MODEL_END ---\n"
    )
    return Experiment(
        id=exp_id,
        index=1,
        status="FAILED",
        proposal=_proposal("broken"),
        primary_metric="roc_auc_macro",
        code=code,
        tasks=[
            Task(
                name="execute",
                status="FAILED",
                error=TaskError(error_type="OOM", message="cuda oom"),
            )
        ],
    )


def _study(
    experiments,
    best_id: str | None,
    *,
    sid: str = "study_x",
) -> Study:
    return Study(
        id=sid,
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=list(experiments),
        best_experiment_id=best_id,
        best_score=(
            max(
                (e.primary_score for e in experiments if e.primary_score is not None),
                default=None,
            )
            if best_id
            else None
        ),
        created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
    )


# --------------------------------------------------------------------------
# _resolve_target_experiment
# --------------------------------------------------------------------------


def test_resolve_target_defaults_to_best():
    e1 = _success_exp("exp_0001", 0.70)
    e2 = _success_exp("exp_0002", 0.81)
    s = _study([e1, e2], best_id="exp_0002")
    assert _resolve_target_experiment(s, None).id == "exp_0002"


def test_resolve_target_accepts_explicit_successful_experiment():
    e1 = _success_exp("exp_0001", 0.70)
    e2 = _success_exp("exp_0002", 0.81)
    s = _study([e1, e2], best_id="exp_0002")
    assert _resolve_target_experiment(s, "exp_0001").id == "exp_0001"


def test_resolve_target_rejects_failed_experiment():
    ok = _success_exp("exp_ok", 0.71)
    bad = _failed_exp("exp_bad")
    s = _study([ok, bad], best_id="exp_ok")
    with pytest.raises(SubmissionValidationError) as exc:
        _resolve_target_experiment(s, "exp_bad")
    assert "did not complete successfully" in str(exc.value)


def test_resolve_target_rejects_unknown_id():
    e = _success_exp("exp_0001", 0.71)
    s = _study([e], best_id="exp_0001")
    with pytest.raises(SubmissionValidationError) as exc:
        _resolve_target_experiment(s, "exp_missing")
    assert "not found in study" in str(exc.value)


def test_resolve_target_raises_when_no_best_and_no_explicit_id():
    s = _study([], best_id=None)
    with pytest.raises(SubmissionValidationError) as exc:
        _resolve_target_experiment(s, None)
    assert "best_experiment_id" in str(exc.value)


# --------------------------------------------------------------------------
# _submission_out_dir
# --------------------------------------------------------------------------


def _settings_with_root(tmp_path: Path) -> Settings:
    base = load_settings("track_b")
    return base.model_copy(
        update={"paths": base.paths.model_copy(update={"experiments_dir": str(tmp_path)})}
    )


def test_out_dir_for_best_writes_to_study_root(tmp_path: Path):
    s = _study(
        [_success_exp("exp_best", 0.81)],
        best_id="exp_best",
        sid="study_legacy",
    )
    out = _submission_out_dir(s, None, _settings_with_root(tmp_path))
    assert out == tmp_path / "study_legacy"


def test_out_dir_for_per_experiment_writes_to_subfolder(tmp_path: Path):
    s = _study(
        [_success_exp("exp_a", 0.81), _success_exp("exp_b", 0.7)],
        best_id="exp_a",
        sid="study_x",
    )
    out = _submission_out_dir(s, "exp_b", _settings_with_root(tmp_path))
    assert out == tmp_path / "study_x" / "experiments" / "exp_b"


def test_out_dir_treats_explicit_best_as_legacy_layout(tmp_path: Path):
    """When the caller passes the best id explicitly, we should still
    write to the study root — no point creating a duplicate subfolder."""
    s = _study(
        [_success_exp("exp_a", 0.81)],
        best_id="exp_a",
        sid="study_y",
    )
    out = _submission_out_dir(s, "exp_a", _settings_with_root(tmp_path))
    assert out == tmp_path / "study_y"


# --------------------------------------------------------------------------
# build_submission_for_study end-to-end
# --------------------------------------------------------------------------


def test_build_submission_for_best_writes_to_study_root(tmp_path: Path):
    e_best = _success_exp("exp_best", 0.81)
    e_other = _success_exp("exp_other", 0.70, arch="arch_other")
    s = _study([e_best, e_other], best_id="exp_best", sid="study_default")
    s.save(tmp_path)

    out = build_submission_for_study(s, _settings_with_root(tmp_path))
    assert out == tmp_path / "study_default" / "submission.ipynb"
    assert out.exists()
    payload = json.loads(out.read_text())
    notebook_text = json.dumps(payload)
    assert "exp_best" in notebook_text


def test_build_submission_for_explicit_experiment_writes_to_subfolder(
    tmp_path: Path,
):
    e_best = _success_exp("exp_best", 0.81, arch="arch_best")
    e_other = _success_exp("exp_other", 0.70, arch="arch_other")
    s = _study([e_best, e_other], best_id="exp_best", sid="study_split")
    s.save(tmp_path)

    out = build_submission_for_study(
        s, _settings_with_root(tmp_path), experiment_id="exp_other"
    )
    assert out == (
        tmp_path / "study_split" / "experiments" / "exp_other" / "submission.ipynb"
    )
    assert out.exists()
    payload = json.loads(out.read_text())
    notebook_text = json.dumps(payload)
    # Manual-selection breadcrumb in the env cell points at the chosen
    # experiment, not the best one.
    assert "exp_other" in notebook_text
    assert "Experiment (manual selection)" in notebook_text

    # The study-level best submission must remain untouched.
    assert not (tmp_path / "study_split" / "submission.ipynb").exists()


def test_build_submission_refuses_failed_experiment(tmp_path: Path):
    e_best = _success_exp("exp_best", 0.81)
    e_bad = _failed_exp("exp_bad")
    s = _study([e_best, e_bad], best_id="exp_best", sid="study_reject")
    s.save(tmp_path)

    with pytest.raises(SubmissionValidationError):
        build_submission_for_study(
            s, _settings_with_root(tmp_path), experiment_id="exp_bad"
        )
