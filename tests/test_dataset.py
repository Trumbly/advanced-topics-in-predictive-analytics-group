"""ensure_dataset_present: synthetic stand-ins live in a separate dir."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lab.config import load_settings
from lab.tasks.dataset import (
    ensure_dataset_present,
    synthetic_dir_for,
    write_synthetic_shards,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_task = s.task.model_copy(
        update={
            "processed_data_dir": str(tmp_path / "mels"),
            "expected_num_classes": 4,
            "input_tensor_shape": [1, 8, 8],
        }
    )
    return s.model_copy(update={"task": new_task})


def test_synthetic_dir_is_sibling_with_suffix(tmp_path):
    real = tmp_path / "mels"
    out = synthetic_dir_for(real)
    assert out == tmp_path / "mels_synthetic"


def test_writes_synthetic_into_separate_dir_when_missing(settings, tmp_path):
    real = Path(settings.task.processed_data_dir)
    synthetic = synthetic_dir_for(real)
    assert not real.exists()
    assert not synthetic.exists()

    new_settings, used = ensure_dataset_present(settings, n_train=20, n_val=5)
    assert used is True
    assert new_settings.task.processed_data_dir == str(synthetic)

    # Real dir untouched
    assert not (real / "train.pt").exists()
    # Synthetic dir populated
    assert (synthetic / "train.pt").exists()
    assert (synthetic / "val.pt").exists()
    blob = torch.load(synthetic / "train.pt", map_location="cpu", weights_only=False)
    assert blob["x"].shape == (20, 1, 8, 8)
    assert blob["y"].shape == (20, 4)


def test_real_data_takes_precedence(settings, tmp_path):
    real = Path(settings.task.processed_data_dir)
    real.mkdir(parents=True)
    sentinel = {"x": torch.zeros(3, 1, 8, 8), "y": torch.zeros(3, 4)}
    torch.save(sentinel, real / "train.pt")

    new_settings, used = ensure_dataset_present(settings)
    assert used is False
    assert new_settings.task.processed_data_dir == str(real)
    blob = torch.load(real / "train.pt", map_location="cpu", weights_only=False)
    assert blob["x"].shape == (3, 1, 8, 8)
    assert not synthetic_dir_for(real).exists()


def test_existing_synthetic_is_reused_not_regenerated(settings):
    new_settings, _ = ensure_dataset_present(settings, n_train=10, n_val=5)
    synthetic = Path(new_settings.task.processed_data_dir)
    first = (synthetic / "train.pt").stat().st_mtime_ns

    new_settings_2, used = ensure_dataset_present(settings, n_train=10, n_val=5)
    assert used is True
    second = (synthetic / "train.pt").stat().st_mtime_ns
    assert first == second


def test_write_synthetic_shards_force_overwrite(settings):
    write_synthetic_shards(settings, n_train=10, n_val=5)
    synthetic = synthetic_dir_for(settings.task.processed_data_dir)
    assert (synthetic / "train.pt").exists()

    write_synthetic_shards(settings, n_train=10, n_val=5, overwrite=True)
    blob = torch.load(synthetic / "train.pt", map_location="cpu", weights_only=False)
    assert blob["x"].shape[0] == 10
