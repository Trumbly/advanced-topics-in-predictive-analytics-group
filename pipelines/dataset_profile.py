"""Build a DatasetProfile from a preprocessed BirdCLEF dataset.

The DatasetProfile is a frozen description of the data (class counts, shape,
split indices) that is computed once and then referenced by every Study.
Injecting it into the LLM prompt gives the agent full knowledge of the data
it is working with.

Layout expected on disk (produced by `scripts/build_profile.py`):

    data/processed/
        spectrograms/
            <audio_id>_<window_idx>.npy    # float32, shape (n_mels, time)
        labels.csv                          # columns: sample_id, class_id
        dataset_profile.json                # produced by this module

"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from agent.models import ClassStats, DatasetProfile


def _load_labels(labels_csv: Path) -> list[tuple[str, str]]:
    """Load `(sample_id, class_id)` rows from the labels CSV.

    For multi-label datasets, a single sample may appear multiple times
    (one row per positive class). This function returns all rows as-is;
    the caller is responsible for aggregation.
    """
    labels_csv = Path(labels_csv)
    rows: list[tuple[str, str]] = []
    with labels_csv.open() as fh:
        reader = csv.DictReader(fh)
        required = {"sample_id", "class_id"}
        fieldnames = set(reader.fieldnames or [])
        if not required.issubset(fieldnames):
            raise ValueError(
                f"{labels_csv} must contain columns {required}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            rows.append((row["sample_id"], row["class_id"]))
    return rows


def _infer_spectrogram_shape(processed_dir: Path) -> tuple[int, int, int]:
    """Probe the first `.npy` file to determine the spectrogram shape.

    Returns `(channels, n_mels, time_frames)`. We assume all spectrograms
    in the dataset share the same shape because they were produced by the
    same AudioPipeline config.
    """
    specs_dir = processed_dir / "spectrograms"
    if not specs_dir.exists():
        raise FileNotFoundError(f"No spectrograms directory found at {specs_dir}")
    try:
        first_npy = next(specs_dir.rglob("*.npy"))
    except StopIteration as exc:
        raise FileNotFoundError(f"No .npy files found under {specs_dir}") from exc

    spec = np.load(first_npy)
    if spec.ndim == 2:
        return (1, spec.shape[0], spec.shape[1])
    if spec.ndim == 3:
        return tuple(spec.shape)  # type: ignore[return-value]
    raise ValueError(
        f"Unexpected spectrogram shape {spec.shape} in {first_npy}; "
        "expected 2D (n_mels, time) or 3D (channels, n_mels, time)"
    )


def _stratified_split(
    sample_ids: list[str],
    sample_to_classes: dict[str, set[str]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Deterministic split that ensures every class is represented in val.

    For multi-label data we can't do a true stratified split, so we use a
    simple heuristic: shuffle with a fixed seed, then assign each sample to
    val until every class is covered, then split the remainder by ratio.
    """
    rng = np.random.default_rng(seed)
    indices = list(range(len(sample_ids)))
    rng.shuffle(indices)

    all_classes: set[str] = set()
    for classes in sample_to_classes.values():
        all_classes |= classes

    covered: set[str] = set()
    val_indices: list[int] = []
    train_indices: list[int] = []

    target_val_size = max(1, int(len(indices) * val_fraction))

    for idx in indices:
        sid = sample_ids[idx]
        sample_classes = sample_to_classes.get(sid, set())
        needs_coverage = bool(sample_classes - covered)
        if needs_coverage or len(val_indices) < target_val_size:
            val_indices.append(idx)
            covered |= sample_classes
        else:
            train_indices.append(idx)

    return sorted(train_indices), sorted(val_indices)


def _compute_class_stats(
    sample_to_classes: dict[str, set[str]],
    *,
    avg_duration_seconds: float = 5.0,
) -> list[ClassStats]:
    """Count samples per class.

    We don't have per-file durations at profile-build time (they would
    require loading every audio file), so we record a nominal average
    equal to the window length. The agent does not depend on this value
    for architecture decisions.
    """
    counts: dict[str, int] = defaultdict(int)
    for classes in sample_to_classes.values():
        for c in classes:
            counts[c] += 1
    return [
        ClassStats(
            class_id=class_id,
            sample_count=count,
            avg_duration_seconds=avg_duration_seconds,
        )
        for class_id, count in sorted(counts.items())
    ]


def build_dataset_profile(
    processed_dir: Path,
    labels_csv: Path,
    *,
    sample_rate: int = 32_000,
    split_strategy: str = "stratified_kfold",
    split_seed: int = 42,
    val_fraction: float = 0.2,
    avg_duration_seconds: float = 5.0,
) -> DatasetProfile:
    """Build a DatasetProfile for a preprocessed BirdCLEF dataset.

    Args:
        processed_dir: Directory containing `spectrograms/*.npy`
        labels_csv: CSV file with columns `sample_id, class_id`
        sample_rate: Audio sample rate used during preprocessing
        split_strategy: Label stored on the profile (documentation only)
        split_seed: Seed for deterministic train/val split
        val_fraction: Fraction of samples to assign to validation
        avg_duration_seconds: Nominal per-sample duration to record

    Returns:
        A fully populated `DatasetProfile` instance.
    """
    processed_dir = Path(processed_dir)

    rows = _load_labels(labels_csv)
    sample_to_classes: dict[str, set[str]] = defaultdict(set)
    for sid, cid in rows:
        sample_to_classes[sid].add(cid)
    sample_ids = sorted(sample_to_classes.keys())

    class_stats = _compute_class_stats(
        sample_to_classes, avg_duration_seconds=avg_duration_seconds
    )
    counts = [c.sample_count for c in class_stats]
    if not counts:
        raise ValueError("No classes found in labels file")

    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    spectrogram_shape = _infer_spectrogram_shape(processed_dir)

    train_indices, val_indices = _stratified_split(
        sample_ids,
        sample_to_classes,
        val_fraction=val_fraction,
        seed=split_seed,
    )

    return DatasetProfile(
        num_classes=len(class_stats),
        num_samples=len(sample_ids),
        spectrogram_shape=spectrogram_shape,
        sample_rate=sample_rate,
        class_stats=class_stats,
        imbalance_ratio=imbalance_ratio,
        min_class_samples=min_count,
        max_class_samples=max_count,
        split_strategy=split_strategy,
        split_seed=split_seed,
        train_indices=train_indices,
        val_indices=val_indices,
    )


__all__ = ["build_dataset_profile"]
