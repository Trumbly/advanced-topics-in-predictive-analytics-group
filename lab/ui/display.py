from __future__ import annotations

from typing import Iterable, Mapping


def study_suffix(study_id: str) -> str:
    """Pull the trailing 4-char suffix off a study id."""
    if "_" in study_id:
        tail = study_id.rsplit("_", 1)[-1]
        if tail:
            return tail
    return study_id[-4:]


def study_label(study_id: str, ordinal: int) -> str:
    """Human-readable study label, e.g. 'Study 3 (n479)'."""
    return f"Study {ordinal} ({study_suffix(study_id)})"


def exp_label(exp) -> str:
    """Human-readable experiment label, e.g. 'Exp 1 (0000)'.

    `exp` is any object with `.id` (string like 'exp_0000') and
    `.index` (zero-based int).
    """
    short = exp.id.split("_", 1)[-1] if "_" in exp.id else exp.id
    return f"Exp {exp.index + 1} ({short})"


def build_study_ordinals(studies: Iterable) -> Mapping[str, int]:
    """Map study id -> 1-based chronological ordinal by created_at."""
    ordered = sorted(studies, key=lambda s: s.created_at)
    return {s.id: i + 1 for i, s in enumerate(ordered)}
