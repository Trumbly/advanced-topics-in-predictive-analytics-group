"""Build real BirdCLEF dataset for the skeleton.

Two output formats:

- **Lazy index (default)** — writes ``train_index.json`` + ``val_index.json``
  with all matched (sample_id, class_idx) entries. The skeleton's
  ``LazyMelDataset`` reads each ``.npy`` on demand at training time, so the
  full 233k-sample corpus is usable without holding 37 GB of tensors in RAM.

- **Eager shards** — when ``samples_per_class`` is set, stratified-subsamples
  each class, stacks into ``train.pt`` + ``val.pt`` tensors. Useful for
  smoke tests where I/O parallelism is overkill.

Inputs (already on disk, written by ``scripts/build_profile.py``):
  - ``data/processed/labels.csv`` -- ``sample_id,class_id`` per row
  - ``data/processed/spectrograms/<sample_id>.npy`` -- one mel per sample
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from lab.config import Settings

_LOG = logging.getLogger("lab.preprocess")


def build_real_shards(
    settings: Settings,
    *,
    samples_per_class: int | None = None,
    val_fraction: float = 0.2,
    spectrogram_subdir: str = "spectrograms",
    labels_filename: str = "labels.csv",
    seed: int = 0,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Materialise the dataset for the skeleton.

    - ``samples_per_class=None`` (default) writes a **lazy index** covering
      every sample from ``labels.csv`` -- fast, ~MB of JSON, training reads
      .npy files on-demand.
    - ``samples_per_class=N`` stratified-subsamples each class to N entries
      and writes eager train.pt + val.pt tensors.

    Returns ``(train_path, val_path)`` (.json paths in lazy mode, .pt in
    eager). Raises FileNotFoundError when labels.csv or the spectrograms
    directory is missing.
    """
    processed_dir = Path(settings.task.processed_data_dir)
    spectrograms_dir = processed_dir.parent / spectrogram_subdir
    labels_path = processed_dir.parent / labels_filename

    if not spectrograms_dir.exists():
        raise FileNotFoundError(
            f"missing spectrogram dir: {spectrograms_dir} — "
            "run scripts/build_profile.py first"
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"missing labels CSV: {labels_path}")

    samples_by_class = _read_labels(labels_path)
    class_ids = sorted(samples_by_class)
    class_to_idx = {cid: i for i, cid in enumerate(class_ids)}
    num_classes = len(class_ids)

    if num_classes != settings.task.expected_num_classes:
        _LOG.warning(
            "expected_num_classes=%d but labels.csv has %d distinct classes; "
            "using the data's count",
            settings.task.expected_num_classes,
            num_classes,
        )

    if samples_per_class is None:
        return _build_lazy_index(
            processed_dir,
            spectrograms_dir,
            samples_by_class,
            class_to_idx,
            num_classes,
            val_fraction=val_fraction,
            seed=seed,
            overwrite=overwrite,
        )

    return _build_eager_shards(
        processed_dir,
        spectrograms_dir,
        samples_by_class,
        class_ids,
        class_to_idx,
        num_classes,
        samples_per_class=samples_per_class,
        val_fraction=val_fraction,
        seed=seed,
        overwrite=overwrite,
        input_shape=tuple(settings.task.input_tensor_shape),
    )


def _read_labels(labels_path: Path) -> dict[str, list[str]]:
    samples_by_class: dict[str, list[str]] = {}
    with labels_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        sid_col = header.index("sample_id")
        cls_col = header.index("class_id")
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) <= max(sid_col, cls_col):
                continue
            samples_by_class.setdefault(parts[cls_col], []).append(parts[sid_col])
    return samples_by_class


def _build_lazy_index(
    processed_dir: Path,
    spectrograms_dir: Path,
    samples_by_class: dict[str, list[str]],
    class_to_idx: dict[str, int],
    num_classes: int,
    *,
    val_fraction: float,
    seed: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    train_path = processed_dir / "train_index.json"
    val_path = processed_dir / "val_index.json"
    if train_path.exists() and val_path.exists() and not overwrite:
        _LOG.info("lazy index already present at %s; skipping", processed_dir)
        return train_path, val_path

    rng = random.Random(seed)
    train_entries: list[dict] = []
    val_entries: list[dict] = []
    for cid, sids in samples_by_class.items():
        sids = list(sids)
        rng.shuffle(sids)
        n_val = max(1, int(round(len(sids) * val_fraction))) if len(sids) > 1 else 0
        for i, sid in enumerate(sids):
            entry = {"sid": sid, "class_idx": class_to_idx[cid]}
            (val_entries if i < n_val else train_entries).append(entry)

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path.write_text(
        json.dumps(
            {
                "spectrograms_dir": str(spectrograms_dir.resolve()),
                "num_classes": num_classes,
                "samples": train_entries,
            }
        )
    )
    val_path.write_text(
        json.dumps(
            {
                "spectrograms_dir": str(spectrograms_dir.resolve()),
                "num_classes": num_classes,
                "samples": val_entries,
            }
        )
    )
    _LOG.info(
        "wrote lazy index: %d train + %d val samples (no .pt stacking)",
        len(train_entries),
        len(val_entries),
    )
    return train_path, val_path


def _build_eager_shards(
    processed_dir: Path,
    spectrograms_dir: Path,
    samples_by_class: dict[str, list[str]],
    class_ids: list[str],
    class_to_idx: dict[str, int],
    num_classes: int,
    *,
    samples_per_class: int,
    val_fraction: float,
    seed: int,
    overwrite: bool,
    input_shape: tuple[int, ...],
) -> tuple[Path, Path]:
    import numpy as np
    import torch

    train_path = processed_dir / "train.pt"
    val_path = processed_dir / "val.pt"
    if train_path.exists() and val_path.exists() and not overwrite:
        _LOG.info("eager shards already present at %s; skipping", processed_dir)
        return train_path, val_path

    rng = np.random.default_rng(seed)

    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    val_x: list[np.ndarray] = []
    val_y: list[int] = []

    shape = input_shape

    for cid in class_ids:
        sids = samples_by_class[cid]
        rng.shuffle(sids)
        sids = sids[:samples_per_class]
        if not sids:
            continue
        n_val = max(1, int(round(len(sids) * val_fraction))) if len(sids) > 1 else 0
        for i, sid in enumerate(sids):
            spec_path = spectrograms_dir / f"{sid}.npy"
            if not spec_path.exists():
                continue
            arr = np.load(spec_path).astype("float32")
            if arr.shape != shape:
                # Some npy files might be (n_mels, n_frames); fold in channel dim.
                if arr.ndim == 2 and arr.shape == shape[1:]:
                    arr = arr[None, :, :]
                else:
                    _LOG.warning("dropping %s (shape %s != %s)", sid, arr.shape, shape)
                    continue
            label_idx = class_to_idx[cid]
            if i < n_val:
                val_x.append(arr)
                val_y.append(label_idx)
            else:
                train_x.append(arr)
                train_y.append(label_idx)

    if not train_x:
        raise RuntimeError(
            "no usable training samples after preprocessing — check that "
            f"{spectrograms_dir} contains .npy files matching labels.csv ids"
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    _save_shard(train_path, train_x, train_y, num_classes)
    _save_shard(val_path, val_x or train_x[: max(1, len(train_x) // 5)],
                val_y or train_y[: max(1, len(train_x) // 5)],
                num_classes)
    _LOG.info(
        "wrote %d train + %d val samples to %s",
        len(train_x),
        len(val_x),
        processed_dir,
    )
    return train_path, val_path


def _save_shard(
    path: "Path", xs, ys, num_classes: int
) -> None:
    import numpy as np
    import torch

    x = torch.from_numpy(np.stack(xs))  # (N, 1, 128, 313)
    y = torch.zeros(x.shape[0], num_classes, dtype=torch.float32)
    for row, cls_idx in enumerate(ys):
        y[row, cls_idx] = 1.0
    torch.save({"x": x, "y": y}, path)
