"""Load precomputed mel-spectrograms into PyTorch datasets.

This is the single entry point the LLM-generated training code must use:

    from pipelines.data_loader import load_precomputed_dataset
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        profile_path="data/processed/dataset_profile.json",
        spectrograms_dir="data/processed/spectrograms",
        labels_csv="data/processed/labels.csv",
        batch_size=32,
        augmentation={"time_shift": True, "noise_injection": True},
    )

By routing all data access through this function we guarantee that every
experiment sees the same train/val split (from the DatasetProfile) and the
same preprocessing config.

This module is intentionally lazy-imported: torch is only imported inside
`load_precomputed_dataset` so that `agent.models` and `pipelines.dataset_profile`
can be imported in a lightweight CI environment without torch.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from agent.models import DatasetProfile
from pipelines.audio_pipeline import AudioPipeline, AugmentationConfig


# ---------------------------------------------------------------------------
# Default path resolution
# ---------------------------------------------------------------------------
#
# LLM-generated training scripts run inside a sandbox subprocess whose cwd
# is `sandbox/<study_id>/<exp_id>/`. Relative paths like
# `data/processed/dataset_profile.json` therefore do NOT resolve against
# the repo root — they fail.
#
# The orchestrator's executor sets three environment variables to absolute
# paths before spawning the subprocess:
#
#   BIRDCLEF_DATASET_PROFILE   -> absolute path to dataset_profile.json
#   BIRDCLEF_SPECTROGRAMS_DIR  -> absolute path to spectrograms/
#   BIRDCLEF_LABELS_CSV        -> absolute path to labels.csv
#
# `load_precomputed_dataset` reads these env vars when its path arguments
# are not supplied, so LLM-generated code can simply call:
#
#     train_loader, val_loader, num_classes = load_precomputed_dataset(
#         batch_size=32,
#         augmentation={"time_shift": True, "noise_injection": True},
#     )
#
# and forget about paths entirely.


def _default_profile_path() -> Path:
    return Path(
        os.environ.get("BIRDCLEF_DATASET_PROFILE")
        or "data/processed/dataset_profile.json"
    )


def _default_spectrograms_dir() -> Path:
    return Path(
        os.environ.get("BIRDCLEF_SPECTROGRAMS_DIR")
        or "data/processed/spectrograms"
    )


def _default_labels_csv() -> Path:
    return Path(
        os.environ.get("BIRDCLEF_LABELS_CSV")
        or "data/processed/labels.csv"
    )


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def _load_multilabel(
    labels_csv: Path, class_to_idx: dict[str, int]
) -> dict[str, np.ndarray]:
    """Return `sample_id -> multi-hot label vector`."""
    num_classes = len(class_to_idx)
    out: dict[str, np.ndarray] = {}
    with Path(labels_csv).open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = row["sample_id"]
            cid = row["class_id"]
            vec = out.setdefault(sid, np.zeros(num_classes, dtype=np.float32))
            if cid in class_to_idx:
                vec[class_to_idx[cid]] = 1.0
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PrecomputedSpectrogramDataset:
    """A torch-style dataset of `(spectrogram, multi_hot_label)` pairs.

    Not a subclass of `torch.utils.data.Dataset` — we lazy-bind inside
    `load_precomputed_dataset` to keep torch optional at import time.
    """

    def __init__(
        self,
        sample_ids: list[str],
        spectrograms_dir: Path,
        labels: dict[str, np.ndarray],
        *,
        augmentation: AugmentationConfig | None = None,
        audio_pipeline: AudioPipeline | None = None,
    ) -> None:
        self.sample_ids = sample_ids
        self.spectrograms_dir = Path(spectrograms_dir)
        self.labels = labels
        self.augmentation = augmentation
        self.audio_pipeline = audio_pipeline or AudioPipeline()
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sid = self.sample_ids[index]
        spec = np.load(self.spectrograms_dir / f"{sid}.npy")
        # Ensure channel dim: (n_mels, time) -> (1, n_mels, time)
        if spec.ndim == 2:
            spec = spec[np.newaxis, :, :]

        if self.augmentation is not None:
            # Apply per-channel
            augmented = np.stack(
                [
                    self.audio_pipeline.augment(spec[c], self.augmentation, self._rng)
                    for c in range(spec.shape[0])
                ]
            )
            spec = augmented.astype(np.float32)

        label = self.labels.get(sid)
        if label is None:
            # Sample has no labels — return zero vector (legitimate negative)
            # The caller's label dict should always contain all sample_ids,
            # so this is defensive.
            label = np.zeros(1, dtype=np.float32)
        return spec, label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_precomputed_dataset(
    profile_path: Path | str | None = None,
    spectrograms_dir: Path | str | None = None,
    labels_csv: Path | str | None = None,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    augmentation: dict[str, Any] | None = None,
    shuffle_train: bool = True,
) -> tuple[Any, Any, int]:
    """Load train and validation DataLoaders from a preprocessed dataset.

    This is the function the LLM-generated training code is expected to call.
    Returns `(train_loader, val_loader, num_classes)`.

    Path arguments are optional. If omitted (or None), they fall back to
    these env vars set by the orchestrator's executor:

        profile_path      -> BIRDCLEF_DATASET_PROFILE
        spectrograms_dir  -> BIRDCLEF_SPECTROGRAMS_DIR
        labels_csv        -> BIRDCLEF_LABELS_CSV

    This means LLM-generated code can simply call:

        load_precomputed_dataset(batch_size=32, augmentation={...})

    without worrying about path resolution across the sandbox cwd.

    The torch import happens inside this function, so `pipelines` can be
    imported in environments without torch (CI, unit tests for models).
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    profile_path = Path(profile_path) if profile_path else _default_profile_path()
    spectrograms_dir = (
        Path(spectrograms_dir) if spectrograms_dir else _default_spectrograms_dir()
    )
    labels_csv = Path(labels_csv) if labels_csv else _default_labels_csv()

    profile = DatasetProfile.from_json_file(profile_path)

    class_to_idx = {c.class_id: i for i, c in enumerate(profile.class_stats)}
    labels = _load_multilabel(labels_csv, class_to_idx)
    sample_ids = sorted(labels.keys())

    aug_config = (
        AugmentationConfig.from_dict(augmentation) if augmentation else None
    )
    audio_pipeline = AudioPipeline()

    # Build two lightweight datasets (train uses augmentation, val does not)
    train_ids = [sample_ids[i] for i in profile.train_indices if i < len(sample_ids)]
    val_ids = [sample_ids[i] for i in profile.val_indices if i < len(sample_ids)]

    train_ds = PrecomputedSpectrogramDataset(
        train_ids,
        spectrograms_dir,
        labels,
        augmentation=aug_config,
        audio_pipeline=audio_pipeline,
    )
    val_ds = PrecomputedSpectrogramDataset(
        val_ids,
        spectrograms_dir,
        labels,
        augmentation=None,
        audio_pipeline=audio_pipeline,
    )

    class _TorchAdapter(Dataset):  # type: ignore[misc]
        def __init__(self, inner: PrecomputedSpectrogramDataset) -> None:
            self.inner = inner

        def __len__(self) -> int:
            return len(self.inner)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            spec, label = self.inner[index]
            return (
                torch.from_numpy(np.ascontiguousarray(spec)),
                torch.from_numpy(np.ascontiguousarray(label)),
            )

    train_loader = DataLoader(
        _TorchAdapter(train_ds),
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        _TorchAdapter(val_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, profile.num_classes


__all__ = [
    "PrecomputedSpectrogramDataset",
    "load_precomputed_dataset",
]
