"""Track A — Kaggle "Natural Language Processing with Disaster Tweets".

Binary text classification. ~10k train tweets, ~3k test.

This module implements both the ``TaskAdapter`` *and* the user-facing
``load_text_dataset`` function that the LLM-generated training code calls
directly. Keeping both in one file makes the Track-A path easy to read.

The tokenizer is deliberately tiny (whitespace + lowercase + UNK) so Track
A trains end-to-end on CPU in under a minute. Swapping in a HuggingFace
tokenizer is a one-adapter-change away.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.core.models import DatasetProfile
from lab.tasks.base import TaskAdapter


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DisasterTweetsAdapter(TaskAdapter):
    name = "track_a"
    kind = "text_classification_binary"
    primary_metric = "f1_binary"

    def spawn_triggering_calls(self) -> tuple[str, ...]:
        return ("load_text_dataset",)

    def build_profile(self) -> DatasetProfile:
        raw = self.task_cfg.get("data", {}).get("raw", {})
        train_csv = self.settings.abspath(raw["train_csv"])
        test_csv = self.settings.abspath(raw["test_csv"]) if raw.get("test_csv") else None
        n_train = n_test = 0
        n_pos = 0
        lengths = []
        if train_csv.exists():
            with train_csv.open() as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    n_train += 1
                    if row.get("target") == "1":
                        n_pos += 1
                    lengths.append(len(row.get("text", "").split()))
        if test_csv and test_csv.exists():
            with test_csv.open() as fh:
                reader = csv.DictReader(fh)
                n_test = sum(1 for _ in reader)
        return DatasetProfile(
            task_name=self.name,
            kind=self.kind,
            num_classes=2,
            num_train_samples=n_train or None,
            num_test_samples=n_test or None,
            extras={
                "positive_fraction": (n_pos / n_train) if n_train else None,
                "avg_tokens_per_sample": (sum(lengths) / len(lengths)) if lengths else None,
                "max_length": self.task_cfg.get("data", {}).get("features", {}).get("max_length", 64),
            },
        )

    def build_submission(self, experiment_code: str, experiment_id: str, out_dir: Path) -> Path:
        """Emit a standalone Python script that trains on the full training
        set and writes ``submission.csv`` in the Kaggle-expected format."""
        out_dir.mkdir(parents=True, exist_ok=True)
        submission_script = out_dir / "submit.py"
        header = (
            f"# Auto-generated submission for track_a, experiment {experiment_id}\n"
            "# Trains on full training set, predicts on test.csv, writes submission.csv\n"
        )
        submission_script.write_text(header + experiment_code)
        return submission_script


# ---------------------------------------------------------------------------
# Data loading (called by LLM-generated code)
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _Vocab:
    token_to_id: dict[str, int]
    size: int

    def encode(self, tokens: list[str], max_length: int) -> list[int]:
        ids = [self.token_to_id.get(t, 1) for t in tokens[:max_length]]
        if len(ids) < max_length:
            ids += [0] * (max_length - len(ids))
        return ids


def _build_vocab(train_path: Path, vocab_size: int) -> _Vocab:
    counter: Counter[str] = Counter()
    with train_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            counter.update(_tokenize(row.get("text", "")))
    # 0 = PAD, 1 = UNK
    most_common = [t for t, _ in counter.most_common(vocab_size - 2)]
    tok_to_id = {"<pad>": 0, "<unk>": 1}
    for i, t in enumerate(most_common, start=2):
        tok_to_id[t] = i
    return _Vocab(token_to_id=tok_to_id, size=len(tok_to_id))


def _read_rows(path: Path) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = row.get("text", "")
            target = int(row.get("target", "0")) if row.get("target") else 0
            out.append((text, target))
    return out


def load_text_dataset(
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    val_fraction: float = 0.1,
    seed: int = 1337,
):
    """Return ``(train_loader, val_loader, vocab_size)``.

    Paths are read from the AGENT_* environment variables populated by the
    orchestrator. This function is the entry point the LLM-generated
    training script imports.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset, random_split

    prefix = os.environ.get("AGENT_ENV_PREFIX", "AGENT")
    train_csv = Path(os.environ.get(f"{prefix}_TRAIN_CSV", "data/raw/track_a/train.csv"))
    processed_dir = Path(
        os.environ.get(f"{prefix}_PROCESSED_DIR", "data/processed/track_a")
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    profile_path = processed_dir / "dataset_profile.json"
    max_length = 64
    vocab_size_cap = 20_000
    if profile_path.exists():
        try:
            data = json.loads(profile_path.read_text())
            max_length = int(data.get("extras", {}).get("max_length", 64))
        except Exception:
            pass

    # Build / load vocab
    vocab_cache = processed_dir / "vocab.json"
    if vocab_cache.exists():
        tok_to_id = json.loads(vocab_cache.read_text())
        vocab = _Vocab(token_to_id=tok_to_id, size=len(tok_to_id))
    else:
        vocab = _build_vocab(train_csv, vocab_size_cap)
        vocab_cache.write_text(json.dumps(vocab.token_to_id))

    rows = _read_rows(train_csv)
    ids = torch.tensor([vocab.encode(_tokenize(t), max_length) for t, _ in rows], dtype=torch.long)
    targets = torch.tensor([y for _, y in rows], dtype=torch.long)

    class _TextDataset(Dataset):
        def __len__(self):
            return len(targets)

        def __getitem__(self, i):
            return ids[i], targets[i]

    ds = _TextDataset()
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
    return train_loader, val_loader, vocab.size


__all__ = ["DisasterTweetsAdapter", "load_text_dataset"]
