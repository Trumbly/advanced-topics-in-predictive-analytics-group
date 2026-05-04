"""I-10 acceptance: TaskAdapter ABC + BirdclefAdapter + dynamic loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.config import load_settings
from lab.core.models import DatasetProfile
from lab.tasks import TaskAdapter, get_task_adapter
from lab.tasks.track_b_birdclef import BirdclefAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings():
    return load_settings("track_b", repo_root=REPO_ROOT)


def test_get_task_adapter_resolves_to_birdclef(settings):
    adapter = get_task_adapter(settings)
    assert isinstance(adapter, BirdclefAdapter)
    assert adapter.name == "track_b"
    assert adapter.kind == "audio_multilabel"
    assert adapter.primary_metric == "roc_auc_macro"


def test_get_task_adapter_rejects_bad_path(settings):
    bad = settings.model_copy(
        update={"task": settings.task.model_copy(update={"adapter": "bogus_format_no_colon"})}
    )
    with pytest.raises(ValueError):
        get_task_adapter(bad)


def test_profile_falls_back_to_config_when_metadata_missing(settings, tmp_path):
    # Point processed_data_dir at an empty dir to force fallback.
    new_task = settings.task.model_copy(update={"processed_data_dir": str(tmp_path)})
    s = settings.model_copy(update={"task": new_task})

    adapter = BirdclefAdapter(s)
    profile = adapter.profile()
    assert isinstance(profile, DatasetProfile)
    assert profile.num_classes == 234
    assert profile.input_tensor_shape == tuple(s.task.input_tensor_shape)


def test_profile_reads_json_sidecar(settings, tmp_path):
    sidecar_dir = tmp_path / "mels"
    sidecar_dir.mkdir()
    (sidecar_dir / "metadata.json").write_text(
        json.dumps(
            {
                "num_classes": 234,
                "num_train": 15320,
                "input_tensor_shape": [1, 128, 313],
                "class_imbalance": {"a": 5, "b": 50},
            }
        )
    )
    new_task = settings.task.model_copy(update={"processed_data_dir": str(sidecar_dir)})
    s = settings.model_copy(update={"task": new_task})

    adapter = BirdclefAdapter(s)
    profile = adapter.profile()
    assert profile.num_train == 15320
    assert profile.class_imbalance == {"a": 5, "b": 50}


def test_profile_rejects_class_count_drift(settings, tmp_path):
    sidecar_dir = tmp_path / "mels"
    sidecar_dir.mkdir()
    (sidecar_dir / "metadata.json").write_text(
        json.dumps(
            {
                "num_classes": 99,
                "num_train": 100,
                "input_tensor_shape": [1, 128, 313],
                "class_imbalance": None,
            }
        )
    )
    new_task = settings.task.model_copy(update={"processed_data_dir": str(sidecar_dir)})
    s = settings.model_copy(update={"task": new_task})
    with pytest.raises(ValueError) as exc:
        BirdclefAdapter(s).profile()
    assert "234" in str(exc.value)


def test_prompt_slots_have_required_keys(settings):
    adapter = get_task_adapter(settings)
    slots = adapter.prompt_slots()
    required = {
        "task_description",
        "num_classes",
        "input_tensor_shape",
        "valid_architecture_families",
        "code_skeleton_content",
    }
    assert required.issubset(slots.keys())
    assert slots["num_classes"] == "234"


def test_signature_and_spawn_calls(settings):
    adapter = get_task_adapter(settings)
    assert adapter.model_block_signature() == ("build_model", "num_classes")
    spawn = list(adapter.spawn_triggering_calls())
    assert "DataLoader" in spawn


def test_build_submission_returns_path(settings, tmp_path):
    adapter = get_task_adapter(settings)
    out = adapter.build_submission(code="...", experiment_id="exp_xxx", out_dir=tmp_path)
    assert out.exists()
    assert out.name == "submission.ipynb"
