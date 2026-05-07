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
    # 5 cells: env / prologue / build_model / helpers+init / csv
    assert len(nb["cells"]) == 5

    env_cell = nb["cells"][0]["source"]
    prologue_cell = nb["cells"][1]["source"]
    build_cell = nb["cells"][2]["source"]
    inference_cell = nb["cells"][3]["source"]

    # NUM_CLASSES must be defined BEFORE build_model is invoked, otherwise
    # the notebook crashes with `NameError: name 'NUM_CLASSES' is not defined`
    # like the user reported on Kaggle.
    assert "NUM_CLASSES" in prologue_cell
    assert "import torch" in prologue_cell
    # build_model lives in its own cell
    assert "def build_model" in build_cell
    # Inference cell instantiates the model + loads weights
    assert "build_model(num_classes=NUM_CLASSES)" in inference_cell
    assert "load_state_dict" in inference_cell
    # Env cell points users at AGENT_WEIGHTS_PATH for the Kaggle Dataset
    assert "AGENT_WEIGHTS_PATH" in inference_cell or "AGENT_WEIGHTS_PATH" in env_cell


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


def test_builder_writes_wide_csv_cell(tmp_path):
    """Kaggle BirdCLEF+ 2026 expects wide-format submission (one row per
    5s window, one column per species). The CSV cell must read the
    canonical column order from sample_submission.csv and write one
    column per species, NOT the old long-format (row_id, species_id,
    probability) triplet."""
    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _GOOD_BUILD_BLOCK)
    study = _study_with_code(settings, code)
    nb_path = build_submission_for_study(study, settings)
    csv_cell = json.loads(nb_path.read_text())["cells"][4]["source"]
    assert "sample_submission.csv" in csv_cell
    assert "species_columns" in csv_cell
    assert "submission.csv" in csv_cell
    # Wide-format: per-row dict keyed by species column header.
    assert "writerows" in csv_cell
    # Long-format columns must NOT be present anymore.
    assert "'species_id'" not in csv_cell
    assert "'probability'" not in csv_cell


def test_builder_inference_cell_loads_ogg_with_matching_mel_params(tmp_path):
    """The notebook's inference path must mirror the training mel cache:
    sr=32k, n_fft=2048, hop=512, n_mels=128. Drift here means train/inference
    get different spectrograms and the model collapses on Kaggle."""
    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _GOOD_BUILD_BLOCK)
    study = _study_with_code(settings, code)
    nb_path = build_submission_for_study(study, settings)
    csv_cell = json.loads(nb_path.read_text())["cells"][4]["source"]
    assert "32000" in csv_cell
    assert "n_fft" in csv_cell.lower()
    assert "hop_length" in csv_cell.lower() or "HOP_LENGTH" in csv_cell
    assert "test_soundscapes" in csv_cell


def test_builder_env_cell_defaults_to_gpu_when_available(tmp_path):
    """Env cell should let the notebook auto-pick CUDA when Kaggle hands us
    a T4, falling back to CPU. Hard-coded `cpu` like before silently halves
    the inference budget on the GPU runtime."""
    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _GOOD_BUILD_BLOCK)
    study = _study_with_code(settings, code)
    nb_path = build_submission_for_study(study, settings)
    env_cell = json.loads(nb_path.read_text())["cells"][0]["source"]
    assert "cuda" in env_cell
    assert "AGENT_DEVICE" in env_cell
    # Doc the inference-only workflow so users understand they upload a
    # checkpoint and run inference, not training.
    assert "INFERENCE-ONLY" in env_cell or "code competition" in env_cell.lower()


def test_local_csv_raises_when_no_test_dir(tmp_path):
    """When data/raw/test_soundscapes is missing, the local CSV builder
    surfaces a clear error with remediation hints — the UI can show
    those instead of a generic 500."""
    from lab.submission.builder import build_local_csv_for_study

    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _GOOD_BUILD_BLOCK)
    study = _study_with_code(settings, code)

    with pytest.raises(SubmissionValidationError) as exc:
        build_local_csv_for_study(study, settings)
    msg = str(exc.value)
    assert "test_soundscapes" in msg or "test audio" in msg or "checkpoint" in msg
    assert exc.value.remediation


def test_local_csv_raises_when_no_best_experiment(tmp_path):
    from lab.submission.builder import build_local_csv_for_study

    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _GOOD_BUILD_BLOCK)
    study = _study_with_code(settings, code)
    study.best_experiment_id = None
    with pytest.raises(SubmissionValidationError) as exc:
        build_local_csv_for_study(study, settings)
    assert "best_experiment_id" in str(exc.value) or "best" in str(exc.value).lower()


def test_copy_weights_creates_weights_pt(tmp_path):
    """The weights download is the artifact users actually upload to
    Kaggle as a Dataset. Builder must copy the sandbox checkpoint into
    the study folder so the UI route can serve it."""
    from lab.submission.builder import copy_weights_for_study

    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _GOOD_BUILD_BLOCK)
    study = _study_with_code(settings, code)

    # Stage a fake checkpoint in the sandbox + wire it onto the
    # experiment. Real runs do this via AGENT_CHECKPOINT_OUT.
    sandbox_ckpt = tmp_path / "sandbox" / "ckpt.pt"
    sandbox_ckpt.parent.mkdir(parents=True)
    sandbox_ckpt.write_bytes(b"\x80\x02fake-state-dict")
    study.experiments[0].checkpoint_path = str(sandbox_ckpt)

    out = copy_weights_for_study(study, settings)
    assert out.name == "weights.pt"
    assert out.read_bytes() == sandbox_ckpt.read_bytes()


def test_copy_weights_raises_when_checkpoint_missing(tmp_path):
    from lab.submission.builder import copy_weights_for_study

    settings = _settings_with_smaller_input(tmp_path)
    rendered = render_skeleton(settings)
    code = splice_build_model(rendered, _GOOD_BUILD_BLOCK)
    study = _study_with_code(settings, code)
    study.experiments[0].checkpoint_path = "/nope/ckpt.pt"
    with pytest.raises(SubmissionValidationError) as exc:
        copy_weights_for_study(study, settings)
    assert "checkpoint" in str(exc.value).lower()
