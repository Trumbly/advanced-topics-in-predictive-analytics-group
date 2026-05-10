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

Group split rationale: each source recording is sliced into multiple
5-second windows whose sids look like ``<recording>_w<index>``. A
sample-level random split lets adjacent windows from the same recording
end up on opposite sides of the train/val divide, so the model learns to
recognise the recording itself instead of the species. We split at the
recording level so all windows of one recording stay on the same side.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from lab.config import Settings

_LOG = logging.getLogger("lab.preprocess")

# Window suffix appended by the spectrogram pipeline (e.g. ``_w003``). Stripping
# it yields the source recording id, which is the unit we split on.
_WINDOW_SUFFIX_RE = re.compile(r"_w\d+$")


def _recording_id(sid: str) -> str:
    """Return the recording id underlying a window-level sample id.

    ``iNat818781_w003 -> iNat818781``. Falls back to the sid itself when the
    window suffix is absent (legacy single-window sids).
    """
    return _WINDOW_SUFFIX_RE.sub("", sid)


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
    sample_labels = _read_sample_labels(labels_path)

    canonical_path = labels_path.parent.parent / "raw" / "sample_submission.csv"
    if canonical_path.exists():
        from lab.tasks.soundscape_preprocess import load_canonical_class_order

        class_ids = load_canonical_class_order(canonical_path)
        class_to_idx = {cid: i for i, cid in enumerate(class_ids)}
        num_classes = len(class_ids)
        _LOG.info(
            "using canonical %d-class ordering from %s",
            num_classes,
            canonical_path,
        )
    else:
        class_ids = sorted(samples_by_class)
        class_to_idx = {cid: i for i, cid in enumerate(class_ids)}
        num_classes = len(class_ids)

    expected = settings.task.expected_num_classes
    if num_classes != expected:
        _LOG.warning(
            "expected_num_classes=%d but the class ordering yields %d; "
            "using %d",
            expected,
            num_classes,
            num_classes,
        )

    n_with_signal = sum(1 for cid in class_ids if samples_by_class.get(cid))
    if n_with_signal < num_classes:
        _LOG.warning(
            "only %d/%d canonical classes have at least one labelled "
            "sample -- the rest will train on zeros",
            n_with_signal,
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
            sample_labels=sample_labels,
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
    """Return ``{class_id: [sample_id, ...]}``.

    Accepts both the legacy single-label header (``sample_id,class_id``) and
    the unified multi-label header (``sample_id,class_ids``) where the value
    is a ``;``-separated list of class IDs. A sample with multi-label appears
    in the bucket of EVERY class it carries -- stratified sampling downstream
    therefore over-counts multi-labelled samples, which is the desired
    behaviour because it makes class-imbalance accounting honest about how
    much signal each class actually has.
    """
    samples_by_class: dict[str, list[str]] = {}
    with labels_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        sid_col = header.index("sample_id")
        if "class_ids" in header:
            cls_col = header.index("class_ids")
            multi = True
        elif "class_id" in header:
            cls_col = header.index("class_id")
            multi = False
        else:
            raise ValueError(
                f"labels CSV {labels_path} must have a `class_id` or "
                f"`class_ids` column; header={header}"
            )
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(sid_col, cls_col):
                continue
            sid = parts[sid_col]
            raw = parts[cls_col]
            class_ids = (
                [c for c in raw.split(";") if c] if multi else [raw]
            )
            for cid in class_ids:
                samples_by_class.setdefault(cid, []).append(sid)
    return samples_by_class


def _read_sample_labels(labels_path: Path) -> dict[str, list[str]]:
    """Return ``{sample_id: [class_id, ...]}`` -- the per-sample view.

    Multi-label samples have len(list) > 1; legacy single-label files always
    have a one-element list. Used by the lazy-index builder to write
    ``class_indices`` per entry.
    """
    samples: dict[str, list[str]] = {}
    with labels_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        sid_col = header.index("sample_id")
        if "class_ids" in header:
            cls_col = header.index("class_ids")
            multi = True
        elif "class_id" in header:
            cls_col = header.index("class_id")
            multi = False
        else:
            raise ValueError(
                f"labels CSV {labels_path} must have a `class_id` or "
                f"`class_ids` column; header={header}"
            )
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(sid_col, cls_col):
                continue
            sid = parts[sid_col]
            raw = parts[cls_col]
            class_ids = (
                [c for c in raw.split(";") if c] if multi else [raw]
            )
            if not class_ids:
                continue
            existing = samples.setdefault(sid, [])
            for cid in class_ids:
                if cid not in existing:
                    existing.append(cid)
    return samples


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
    sample_labels: dict[str, list[str]] | None = None,
) -> tuple[Path, Path]:
    """Lazy index: emits one entry per *sample* with the full ``class_indices``
    list when ``sample_labels`` is provided; otherwise falls back to
    single-class entries derived from ``samples_by_class`` for legacy callers.
    """
    train_path = processed_dir / "train_index.json"
    val_path = processed_dir / "val_index.json"
    if train_path.exists() and val_path.exists() and not overwrite:
        _LOG.info("lazy index already present at %s; skipping", processed_dir)
        return train_path, val_path

    rng = random.Random(seed)
    train_entries: list[dict] = []
    val_entries: list[dict] = []

    if sample_labels is not None:
        # Multi-label path. Group sids by recording (windows from the same
        # recording stay together), shuffle the recordings, and assign each
        # group to one side of the split. We bucket recordings by their
        # *primary* class (the class id whose canonical samples list contains
        # the recording) so the val fraction is roughly stratified. A sid
        # that appears under multiple class buckets is bucketed once.
        recordings_by_class: dict[str, list[str]] = {}
        recording_to_primary: dict[str, str] = {}
        recording_sids: dict[str, list[str]] = {}
        for cid in sorted(samples_by_class):
            for sid in samples_by_class[cid]:
                rec = _recording_id(sid)
                recording_sids.setdefault(rec, []).append(sid)
                if rec not in recording_to_primary:
                    recording_to_primary[rec] = cid
                    recordings_by_class.setdefault(cid, []).append(rec)
        for cid, recs in recordings_by_class.items():
            recs = list(recs)
            rng.shuffle(recs)
            n_val = (
                max(1, int(round(len(recs) * val_fraction))) if len(recs) > 1 else 0
            )
            for i, rec in enumerate(recs):
                target = val_entries if i < n_val else train_entries
                # Materialise every window of the recording into the same
                # split. Deduplicate sids — the same window can be listed
                # under multiple primary classes.
                seen: set[str] = set()
                for sid in recording_sids[rec]:
                    if sid in seen:
                        continue
                    seen.add(sid)
                    indices = [
                        class_to_idx[c]
                        for c in sample_labels.get(sid, [])
                        if c in class_to_idx
                    ]
                    if not indices:
                        continue
                    target.append({"sid": sid, "class_indices": indices})
    else:
        # Single-label legacy path. Same group-split logic but each
        # recording carries only one class.
        for cid, sids in samples_by_class.items():
            recordings: dict[str, list[str]] = {}
            for sid in sids:
                recordings.setdefault(_recording_id(sid), []).append(sid)
            recs = list(recordings)
            rng.shuffle(recs)
            n_val = (
                max(1, int(round(len(recs) * val_fraction))) if len(recs) > 1 else 0
            )
            for i, rec in enumerate(recs):
                target = val_entries if i < n_val else train_entries
                for sid in recordings[rec]:
                    target.append(
                        {
                            "sid": sid,
                            "class_idx": class_to_idx[cid],
                            "class_indices": [class_to_idx[cid]],
                        }
                    )

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


def regroup_lazy_indexes(
    settings: Settings,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Re-split existing ``train_index.json`` + ``val_index.json`` so that
    every recording's windows live entirely on one side of the split.

    The 5-second window pipeline emits sids of the form
    ``<recording>_w<idx>``. The original index builder sometimes split at
    the sid level, which let adjacent windows from one recording leak
    into both train and val and inflated ROC-AUC. This helper reads the
    union of both indexes, regroups by recording, and writes the
    corrected pair back to disk *without* recomputing any spectrograms.

    Returns the (train_path, val_path) of the rewritten files.
    """
    processed_dir = Path(settings.task.processed_data_dir)
    train_path = processed_dir / "train_index.json"
    val_path = processed_dir / "val_index.json"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"missing lazy indexes at {processed_dir}; run `lab preprocess` "
            "first to build the cache before regrouping"
        )

    train_payload = json.loads(train_path.read_text(encoding="utf-8"))
    val_payload = json.loads(val_path.read_text(encoding="utf-8"))
    spectrograms_dir = train_payload.get("spectrograms_dir") or val_payload.get(
        "spectrograms_dir"
    )
    num_classes = int(
        train_payload.get("num_classes") or val_payload.get("num_classes") or 0
    )

    # Union all entries — dedupe on sid so a sample that ended up on both
    # sides only appears once in the regrouped output.
    by_sid: dict[str, dict] = {}
    for entry in train_payload.get("samples", []):
        by_sid[entry["sid"]] = entry
    for entry in val_payload.get("samples", []):
        by_sid.setdefault(entry["sid"], entry)

    # Group sids by recording.
    recording_to_sids: dict[str, list[str]] = {}
    for sid in by_sid:
        recording_to_sids.setdefault(_recording_id(sid), []).append(sid)

    rng = random.Random(seed)
    recordings = sorted(recording_to_sids)
    rng.shuffle(recordings)
    n_val = max(1, int(round(len(recordings) * val_fraction))) if len(recordings) > 1 else 0

    new_train: list[dict] = []
    new_val: list[dict] = []
    for i, rec in enumerate(recordings):
        target = new_val if i < n_val else new_train
        for sid in recording_to_sids[rec]:
            target.append(by_sid[sid])

    train_path.write_text(
        json.dumps(
            {
                "spectrograms_dir": spectrograms_dir,
                "num_classes": num_classes,
                "samples": new_train,
            }
        )
    )
    val_path.write_text(
        json.dumps(
            {
                "spectrograms_dir": spectrograms_dir,
                "num_classes": num_classes,
                "samples": new_val,
            }
        )
    )
    _LOG.info(
        "regrouped lazy indexes: %d train + %d val (%d recordings -> %d train / %d val)",
        len(new_train),
        len(new_val),
        len(recordings),
        len(recordings) - n_val,
        n_val,
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
        # Group sids by recording so windows from the same source recording
        # always land in the same split (see module docstring).
        recordings: dict[str, list[str]] = {}
        for sid in sids:
            recordings.setdefault(_recording_id(sid), []).append(sid)
        rec_list = list(recordings)
        rng.shuffle(rec_list)
        # Cap by recording count rather than sample count when downsampling
        # so the per-class cap behaves intuitively against the group split.
        if samples_per_class:
            rec_list = rec_list[:samples_per_class]
        if not rec_list:
            continue
        n_val_rec = (
            max(1, int(round(len(rec_list) * val_fraction)))
            if len(rec_list) > 1
            else 0
        )
        for i, rec in enumerate(rec_list):
            for sid in recordings[rec]:
                spec_path = spectrograms_dir / f"{sid}.npy"
                if not spec_path.exists():
                    continue
                arr = np.load(spec_path).astype("float32")
                if arr.shape != shape:
                    # Some npy files might be (n_mels, n_frames); fold in channel dim.
                    if arr.ndim == 2 and arr.shape == shape[1:]:
                        arr = arr[None, :, :]
                    else:
                        _LOG.warning(
                            "dropping %s (shape %s != %s)", sid, arr.shape, shape
                        )
                        continue
                label_idx = class_to_idx[cid]
                if i < n_val_rec:
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
