"""Track B — BirdCLEF+ 2026 (audio multi-label classification).

Loads precomputed mel-spectrogram ``.npy`` files produced by the
preprocessing script. Keeps the spectrogram-based IO format compatible
with the preprocessing we used in ``max_development`` so the existing
data cache can be reused.

All configuration (sample rate, n_mels, window seconds, ...) comes from
``config/tasks/track_b.yaml`` — nothing is hardcoded in this module.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from lab.core.models import DatasetProfile
from lab.tasks.base import TaskAdapter


class BirdclefAdapter(TaskAdapter):
    name = "track_b"
    kind = "audio_multilabel"
    primary_metric = "f1_macro"

    def spawn_triggering_calls(self) -> tuple[str, ...]:
        return ("load_audio_dataset",)

    def build_profile(self) -> DatasetProfile:
        raw = self.task_cfg.get("data", {}).get("raw", {})
        meta_csv = self.settings.abspath(raw.get("metadata_csv", "")) if raw.get("metadata_csv") else None
        features = self.task_cfg.get("data", {}).get("features", {})
        n_samples = None
        classes: set[str] = set()
        if meta_csv and meta_csv.exists():
            with meta_csv.open() as fh:
                reader = csv.DictReader(fh)
                n_samples = 0
                class_col = self.task_cfg["data"]["labels"]["class_column"]
                for row in reader:
                    n_samples += 1
                    v = row.get(class_col)
                    if v:
                        classes.add(v)
        n_mels = int(features.get("n_mels", 128))
        seconds = float(features.get("window_seconds", 5.0))
        sr = int(features.get("sample_rate", 32000))
        hop = int(features.get("hop_length", 512))
        time_bins = int(seconds * sr / hop) + 1
        return DatasetProfile(
            task_name=self.name,
            kind=self.kind,
            num_classes=len(classes) or None,
            num_train_samples=n_samples,
            extras={
                "spectrogram_shape": [1, n_mels, time_bins],
                "sample_rate": sr,
                "window_seconds": seconds,
                "n_mels": n_mels,
                "n_classes": len(classes),
            },
        )

    def build_submission(self, experiment_code: str, experiment_id: str, out_dir: Path) -> Path:
        """Emit a Kaggle CPU-only notebook that inference-only reruns
        the model on test_soundscapes and writes ``submission.csv``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        nb_path = out_dir / "submission.ipynb"
        cells = [
            {
                "cell_type": "markdown",
                "source": [
                    f"# Track B submission — {experiment_id}\n\n",
                    "CPU-only. Loads test spectrograms, predicts, writes `submission.csv`.\n",
                ],
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": [
                    "import os\n",
                    "os.environ['AGENT_DEVICE'] = 'cpu'\n",
                    "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n",
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "source": experiment_code.splitlines(keepends=True),
                "metadata": {},
                "execution_count": None,
                "outputs": [],
            },
        ]
        nb = {
            "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "language": "python"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(nb, indent=2))
        return nb_path


# ---------------------------------------------------------------------------
# Data loading (called by LLM-generated code)
# ---------------------------------------------------------------------------


def load_audio_dataset(
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    val_fraction: float = 0.1,
    seed: int = 1337,
):
    """Return ``(train_loader, val_loader, num_classes)``.

    Reads from precomputed ``.npy`` spectrograms + ``labels.csv`` prepared
    by ``scripts/preprocess.py``. Paths come from ``AGENT_*`` env vars
    (set by the orchestrator) — no hardcoded directories.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, random_split

    prefix = os.environ.get("AGENT_ENV_PREFIX", "AGENT")
    processed = Path(os.environ.get(f"{prefix}_PROCESSED_DIR", "data/processed/track_b"))
    spectrograms_dir = processed / "spectrograms"
    labels_csv = processed / "labels.csv"

    if not labels_csv.exists():
        raise FileNotFoundError(
            f"No labels.csv at {labels_csv}. Run `python scripts/build_profile.py --task track_b`"
            " to preprocess the audio first."
        )

    class_list: list[str] = []
    rows: list[tuple[str, list[str]]] = []
    with labels_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels = [s for s in row.get("labels", "").split("|") if s]
            rows.append((row["filename"], labels))
            for c in labels:
                if c not in class_list:
                    class_list.append(c)
    class_list.sort()
    class_to_idx = {c: i for i, c in enumerate(class_list)}

    class _SpecDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            fname, labels = self.rows[i]
            arr = np.load(spectrograms_dir / fname)
            x = torch.from_numpy(arr).float()
            if x.ndim == 2:
                x = x.unsqueeze(0)
            y = torch.zeros(len(class_list), dtype=torch.float)
            for c in labels:
                y[class_to_idx[c]] = 1.0
            return x, y

    ds = _SpecDataset(rows)
    val_len = max(1, int(len(ds) * val_fraction))
    train_len = len(ds) - val_len
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=bool(num_workers),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=bool(num_workers),
    )
    return train_loader, val_loader, len(class_list)


__all__ = ["BirdclefAdapter", "load_audio_dataset"]
