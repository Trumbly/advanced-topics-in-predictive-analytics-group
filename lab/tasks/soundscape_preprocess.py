"""Soundscape ingest for BirdCLEF 2026.

The competition ships labels in two distinct files:

- ``train.csv``                       single-label per file (~35k rows, 206 species)
- ``train_soundscapes_labels.csv``    multi-label per 5-second window (~1.5k rows,
                                       all 234 target species incl. the 28 that
                                       have no per-file ``primary_label`` entry)

Our original preprocessing only consumed ``train.csv``, so 28 of the 234 target
species (25 Insect sonotypes + 3 Amphibia) were silently absent from training.
This module fills that gap by emitting a *unified, multi-label* `labels.csv`
keyed off the canonical 234-class ordering from `sample_submission.csv`.

Output schema (header row + N data rows)::

    sample_id,class_ids
    iNat1114648_w000,1161364
    BC2026_Train_0039_S22_20211231_201500_w001,22961;23158;24321;517063;65380

``class_ids`` is a ``;``-separated list of class IDs (one entry = single-label,
many = multi-label). Class IDs match the column order of
``sample_submission.csv``.

This module is import-safe: it never opens an audio file. The actual mel
generation for soundscape 5-s windows still has to be done by
``scripts/build_profile.py`` (or an extension thereof) -- here we only emit the
label index.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import OrderedDict
from pathlib import Path

_LOG = logging.getLogger("lab.preprocess.soundscape")


_TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")


def load_canonical_class_order(sample_submission_path: Path) -> list[str]:
    """Return the 234 class IDs in submission-column order.

    The first column of ``sample_submission.csv`` is ``row_id``; the rest is
    the canonical class-id list. This ordering is also what the submitted
    notebook must produce, so we use it as the single source of truth across
    preprocessing and training.
    """
    with sample_submission_path.open("r", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    if not header or header[0] != "row_id":
        raise ValueError(
            f"unexpected sample_submission.csv header: {header[:5]}..."
        )
    return list(header[1:])


def ingest_train_csv(train_csv_path: Path) -> "OrderedDict[str, list[str]]":
    """Single-label rows from ``train.csv`` keyed by sample_id.

    Sample IDs come from the basename of ``filename`` (extension stripped),
    matching the convention used by ``scripts/build_profile.py`` to name the
    per-clip ``.npy`` mels.
    """
    out: "OrderedDict[str, list[str]]" = OrderedDict()
    with train_csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            primary = (row.get("primary_label") or "").strip()
            if not primary:
                continue
            filename = row.get("filename") or ""
            sid = _sid_from_filename(filename)
            if not sid:
                continue
            out.setdefault(sid, []).append(primary)
    return out


def ingest_soundscape_labels(
    soundscape_labels_path: Path,
    *,
    window_seconds: int = 5,
) -> "OrderedDict[str, list[str]]":
    """Multi-label rows from ``train_soundscapes_labels.csv`` keyed by window.

    Each row is ``filename, start (HH:MM:SS), end, primary_label``. We compute
    the window index from ``start // window_seconds`` and emit one entry per
    window with the semicolon-split label set.
    """
    out: "OrderedDict[str, list[str]]" = OrderedDict()
    with soundscape_labels_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            filename = row.get("filename") or ""
            start = (row.get("start") or "").strip()
            raw_labels = (row.get("primary_label") or "").strip()
            if not filename or not raw_labels:
                continue
            start_sec = _parse_hms(start)
            if start_sec is None:
                continue
            window_idx = start_sec // window_seconds
            base = _sid_from_filename(filename)
            if not base:
                continue
            sid = f"{base}_w{window_idx:03d}"
            labels = [lab.strip() for lab in raw_labels.split(";") if lab.strip()]
            if not labels:
                continue
            # union with any existing labels (some 5-s windows can appear twice
            # if the upstream label is updated between competition releases)
            existing = out.setdefault(sid, [])
            for lab in labels:
                if lab not in existing:
                    existing.append(lab)
    return out


def merge_label_sources(
    *sources: "OrderedDict[str, list[str]]",
    canonical_classes: list[str],
) -> "OrderedDict[str, list[str]]":
    """Merge per-sample label dicts and drop labels not in the canonical set.

    Canonical filtering keeps the unified `labels.csv` aligned with
    `sample_submission.csv` -- training never sees an extra species the model
    cannot be evaluated against.
    """
    canonical = set(canonical_classes)
    merged: "OrderedDict[str, list[str]]" = OrderedDict()
    n_dropped = 0
    for src in sources:
        for sid, labels in src.items():
            kept = [lab for lab in labels if lab in canonical]
            n_dropped += len(labels) - len(kept)
            if not kept:
                continue
            existing = merged.setdefault(sid, [])
            for lab in kept:
                if lab not in existing:
                    existing.append(lab)
    if n_dropped:
        _LOG.info(
            "dropped %d label occurrences not in sample_submission columns",
            n_dropped,
        )
    return merged


def write_multilabel_labels_csv(
    target: Path,
    samples: "OrderedDict[str, list[str]]",
) -> None:
    """Emit the unified multi-label labels file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        fh.write("sample_id,class_ids\n")
        for sid, labels in samples.items():
            fh.write(f"{sid},{';'.join(labels)}\n")


def build_unified_labels(
    *,
    raw_dir: Path,
    out_path: Path,
    train_csv_name: str = "train.csv",
    soundscape_labels_name: str = "train_soundscapes_labels.csv",
    sample_submission_name: str = "sample_submission.csv",
) -> "OrderedDict[str, list[str]]":
    """Top-level: read all three sources, write unified multi-label labels.csv.

    Returns the in-memory mapping so callers can inspect class counts without
    re-reading the file.
    """
    canonical = load_canonical_class_order(raw_dir / sample_submission_name)

    train_path = raw_dir / train_csv_name
    sscape_path = raw_dir / soundscape_labels_name

    train_part = ingest_train_csv(train_path) if train_path.exists() else OrderedDict()
    sscape_part = (
        ingest_soundscape_labels(sscape_path) if sscape_path.exists() else OrderedDict()
    )
    if not train_part and not sscape_part:
        raise FileNotFoundError(
            f"neither {train_path} nor {sscape_path} exists; nothing to ingest"
        )

    merged = merge_label_sources(
        train_part, sscape_part, canonical_classes=canonical
    )
    write_multilabel_labels_csv(out_path, merged)
    _LOG.info(
        "unified labels.csv: %d samples (%d single-label, %d multi-label) "
        "across %d distinct class_ids out of %d canonical target classes",
        len(merged),
        sum(1 for v in merged.values() if len(v) == 1),
        sum(1 for v in merged.values() if len(v) > 1),
        len({c for v in merged.values() for c in v}),
        len(canonical),
    )
    return merged


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sid_from_filename(filename: str) -> str:
    """``"1161364/iNat1114648.ogg"`` -> ``"iNat1114648"``.

    Uses ``Path(filename).stem`` so suffixes like ``.ogg``/``.wav`` get
    stripped while any sub-directory in the filename is dropped.
    """
    if not filename:
        return ""
    return Path(filename).stem


def _parse_hms(text: str) -> int | None:
    """``"00:01:35"`` -> 95 seconds. Returns None on parse failure."""
    m = _TIMESTAMP_RE.match(text)
    if not m:
        return None
    hours, minutes, seconds = (int(g) for g in m.groups())
    return hours * 3600 + minutes * 60 + seconds
