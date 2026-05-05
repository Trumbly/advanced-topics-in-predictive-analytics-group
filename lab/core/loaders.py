"""Bulk study loaders. Used by the UI list view and reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from lab.core.models import Study, StudyNotFoundError


def list_studies(root: Path) -> list[str]:
    """Return study IDs found under ``root`` (directories with ``study.json``)."""
    root_p = Path(root)
    if not root_p.exists():
        return []
    ids: list[str] = []
    for child in sorted(root_p.iterdir()):
        if child.is_dir() and (child / "study.json").exists():
            ids.append(child.name)
    return ids


def load_many(root: Path, study_ids: Iterable[str]) -> list[Study]:
    """Load each study by id; missing IDs are skipped silently for resilience."""
    out: list[Study] = []
    for sid in study_ids:
        try:
            out.append(Study.load(root, sid))
        except StudyNotFoundError:
            continue
    return out


def load_all(root: Path) -> list[Study]:
    return load_many(root, list_studies(root))
