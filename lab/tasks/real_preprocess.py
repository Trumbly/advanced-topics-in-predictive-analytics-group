"""Build real train.pt + val.pt shards from BirdCLEF mel-spectrograms.

Reads ``data/processed/labels.csv`` (sample_id, class_id) and the matching
``data/processed/spectrograms/<sample_id>.npy`` files, applies a stratified
subsample (cap ``samples_per_class`` per class to keep RAM bounded), splits
80/20 per class, and writes a single ``train.pt`` / ``val.pt`` pair into
``settings.task.processed_data_dir`` (the same path the skeleton already
loads from).

234-class config vs 206-class data: the helper trusts the data and emits
a warning when ``expected_num_classes`` mismatches; the orchestrator's
strict guard in ``BirdclefAdapter.profile`` then needs the config aligned
with reality (we ship a refreshed track_b.yaml in this PR).
"""

from __future__ import annotations

import logging
from pathlib import Path

from lab.config import Settings

_LOG = logging.getLogger("lab.preprocess")


def build_real_shards(
    settings: Settings,
    *,
    samples_per_class: int = 50,
    val_fraction: float = 0.2,
    spectrogram_subdir: str = "spectrograms",
    labels_filename: str = "labels.csv",
    seed: int = 0,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Materialise train.pt + val.pt from the real Pantanal mel-spectrograms.

    Returns the (train_path, val_path). Raises FileNotFoundError when the
    upstream artefacts are missing — callers must bootstrap them out of
    band (see scripts/build_profile.py) before calling this.
    """
    import numpy as np
    import torch

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

    train_path = processed_dir / "train.pt"
    val_path = processed_dir / "val.pt"
    if train_path.exists() and val_path.exists() and not overwrite:
        _LOG.info("real shards already present at %s; skipping", processed_dir)
        return train_path, val_path

    rng = np.random.default_rng(seed)
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

    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    val_x: list[np.ndarray] = []
    val_y: list[int] = []

    shape = tuple(settings.task.input_tensor_shape)

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
