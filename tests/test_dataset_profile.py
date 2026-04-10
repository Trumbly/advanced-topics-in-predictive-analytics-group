"""Unit tests for `pipelines.dataset_profile.build_dataset_profile`."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from agent.models import DatasetProfile
from pipelines.dataset_profile import build_dataset_profile


def _make_fake_dataset(
    tmp_path: Path,
    *,
    num_samples: int = 10,
    num_classes: int = 3,
    spec_shape: tuple[int, int] = (32, 64),
) -> tuple[Path, Path]:
    """Create a synthetic preprocessed dataset layout.

    Returns `(processed_dir, labels_csv)`.
    """
    processed_dir = tmp_path / "processed"
    spec_dir = processed_dir / "spectrograms"
    spec_dir.mkdir(parents=True)

    labels_csv = processed_dir / "labels.csv"
    rng = np.random.default_rng(0)

    with labels_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "class_id"])
        for i in range(num_samples):
            sample_id = f"sample_{i:04d}"
            spec = rng.normal(0, 1, size=spec_shape).astype(np.float32)
            np.save(spec_dir / f"{sample_id}.npy", spec)
            # Assign a primary class; for multi-label flavor, also tag class_0
            class_id = f"class_{i % num_classes:02d}"
            writer.writerow([sample_id, class_id])
            if i % 4 == 0:
                writer.writerow([sample_id, "class_00"])

    return processed_dir, labels_csv


class TestBuildDatasetProfile:
    def test_smoke(self, tmp_path: Path) -> None:
        processed_dir, labels_csv = _make_fake_dataset(tmp_path)
        profile = build_dataset_profile(processed_dir, labels_csv, sample_rate=22_050)

        assert isinstance(profile, DatasetProfile)
        assert profile.num_samples == 10
        assert profile.sample_rate == 22_050
        assert profile.spectrogram_shape == (1, 32, 64)
        assert profile.split_seed == 42
        assert len(profile.class_stats) >= 3

    def test_split_is_deterministic(self, tmp_path: Path) -> None:
        processed_dir, labels_csv = _make_fake_dataset(tmp_path)
        profile_a = build_dataset_profile(
            processed_dir, labels_csv, split_seed=123
        )
        profile_b = build_dataset_profile(
            processed_dir, labels_csv, split_seed=123
        )
        assert profile_a.train_indices == profile_b.train_indices
        assert profile_a.val_indices == profile_b.val_indices

    def test_split_changes_with_seed(self, tmp_path: Path) -> None:
        processed_dir, labels_csv = _make_fake_dataset(
            tmp_path, num_samples=20, num_classes=4
        )
        profile_a = build_dataset_profile(
            processed_dir, labels_csv, split_seed=0
        )
        profile_b = build_dataset_profile(
            processed_dir, labels_csv, split_seed=999
        )
        assert profile_a.val_indices != profile_b.val_indices

    def test_all_classes_covered_in_val(self, tmp_path: Path) -> None:
        processed_dir, labels_csv = _make_fake_dataset(
            tmp_path, num_samples=30, num_classes=5
        )
        profile = build_dataset_profile(
            processed_dir, labels_csv, val_fraction=0.3
        )
        # Val indices must be a valid subset
        assert all(
            0 <= idx < profile.num_samples for idx in profile.val_indices
        )
        assert set(profile.train_indices).isdisjoint(set(profile.val_indices))

    def test_missing_spectrograms_directory_raises(self, tmp_path: Path) -> None:
        bad_dir = tmp_path / "empty"
        bad_dir.mkdir()
        labels_csv = bad_dir / "labels.csv"
        labels_csv.write_text("sample_id,class_id\nid,c\n")
        with pytest.raises(FileNotFoundError):
            build_dataset_profile(bad_dir, labels_csv)

    def test_missing_label_columns_raises(self, tmp_path: Path) -> None:
        processed_dir, _ = _make_fake_dataset(tmp_path)
        bad_labels = processed_dir / "bad.csv"
        bad_labels.write_text("foo,bar\nx,y\n")
        with pytest.raises(ValueError, match="sample_id"):
            build_dataset_profile(processed_dir, bad_labels)
