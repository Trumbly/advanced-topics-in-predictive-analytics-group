"""I-14 acceptance: submission builder + submission-mode validator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.config import load_settings
from lab.core.models import Experiment, Proposal, Study
from lab.submission.builder import (
    SubmissionValidationError,
    build_submission_for_study,
)
from lab.tasks.skeleton import render_skeleton, splice_build_model

REPO_ROOT = Path(__file__).resolve().parent.parent

_GOOD_BUILD_BLOCK = """\
def build_model(num_classes: int) -> nn.Module:
    in_chan = INPUT_SHAPE[0]
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_chan * INPUT_SHAPE[1] * INPUT_SHAPE[2], num_classes),
    )
"""

_NETWORK_BLOCK = """\
import urllib.request
def build_model(num_classes: int) -> nn.Module:
    return nn.Linear(10, num_classes)
"""


def _settings_with_smaller_input(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"experiments_dir": str(tmp_path / "studies")})
    new_task = s.task.model_copy(
        update={"expected_num_classes": 4, "input_tensor_shape": [1, 8, 8]}
    )
    return s.model_copy(update={"paths": new_paths, "task": new_task})


def _study_with_code(settings, code: str, *, study_id="study_sub_xxxx") -> Study:
    proposal = Proposal(
        architecture_name="A",
        family="cnn_scratch",
        lr=1e-3,
        lr_schedule="cosine",
        epochs=2,
    )
    exp = Experiment(
        id="exp_best",
        index=0,
        status="JUDGED",
        proposal=proposal,
        primary_metric="roc_auc_macro",
        primary_score=0.7,
        code=code,
    )
    return Study(
        id=study_id,
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=[exp],
        best_experiment_id=exp.id,
        best_score=0.7,
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )


def test_builder_writes_valid_notebook(tmp_path):
    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _GOOD_BUILD_BLOCK)
    study = _study_with_code(settings, code)
    nb_path = build_submission_for_study(study, settings)
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 4
    # build_model block survives into cell 2
    code_cell = nb["cells"][1]["source"]
    assert "build_model" in code_cell


def test_builder_rejects_network_imports(tmp_path):
    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _NETWORK_BLOCK)
    study = _study_with_code(settings, code)
    with pytest.raises(SubmissionValidationError) as exc:
        build_submission_for_study(study, settings)
    assert "ForbiddenImport" in str(exc.value) or "urllib" in str(exc.value)
    assert exc.value.remediation


def test_builder_rejects_when_no_best_experiment(tmp_path):
    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _GOOD_BUILD_BLOCK)
    study = _study_with_code(settings, code)
    study.best_experiment_id = None
    with pytest.raises(SubmissionValidationError):
        build_submission_for_study(study, settings)


def test_builder_writes_csv_cell_with_required_columns(tmp_path):
    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _GOOD_BUILD_BLOCK)
    study = _study_with_code(settings, code)
    nb_path = build_submission_for_study(study, settings)
    csv_cell = json.loads(nb_path.read_text())["cells"][3]["source"]
    for col in ("row_id", "species_id", "probability"):
        assert col in csv_cell
    assert "submission.csv" in csv_cell
