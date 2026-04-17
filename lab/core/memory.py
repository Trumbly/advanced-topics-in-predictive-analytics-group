"""Experiment memory — the compact knowledge base injected into every prompt.

The memory stores the top-K successful experiments (by primary metric) and
the most recent failures, and renders them as a tiny markdown blob suitable
for LLM context. Predecessor seeding supports study continuation: a new
study inherits the predecessor's memory so the LLM can see what was
already tried.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lab.core.models import Experiment, Study


@dataclass
class MemoryEntry:
    experiment_id: str
    architecture_name: str
    architecture_family: str | None
    status: str
    primary_metric: str
    primary_score: float | None
    error_type: str | None
    error_message: str | None
    duration_seconds: float | None
    takeaway: str = ""
    checkpoint_path: str | None = None

    @classmethod
    def from_experiment(cls, exp: Experiment) -> "MemoryEntry":
        err = exp.error
        return cls(
            experiment_id=exp.id,
            architecture_name=exp.architecture_name or "<unknown>",
            architecture_family=exp.architecture_family,
            status=exp.status.value,
            primary_metric=exp.primary_metric,
            primary_score=exp.primary_score,
            error_type=err.error_type if err else None,
            error_message=err.message if err else None,
            duration_seconds=exp.duration_seconds,
            checkpoint_path=exp.checkpoint_path,
        )


@dataclass
class Memory:
    score_metric: str = ""
    entries: list[MemoryEntry] = field(default_factory=list)

    # -- mutation -----------------------------------------------------------

    def add(self, exp: Experiment) -> None:
        self.entries.append(MemoryEntry.from_experiment(exp))

    def seed_from_predecessor(self, predecessor: Study) -> None:
        """Copy the predecessor study's experiments into this memory."""
        for exp in predecessor.experiments:
            self.entries.append(MemoryEntry.from_experiment(exp))

    # -- queries ------------------------------------------------------------

    def successes(self) -> list[MemoryEntry]:
        return [e for e in self.entries if e.status == "completed" and e.primary_score is not None]

    def failures(self) -> list[MemoryEntry]:
        return [e for e in self.entries if e.status == "failed"]

    def top_k(self, k: int) -> list[MemoryEntry]:
        wins = sorted(
            self.successes(),
            key=lambda e: e.primary_score or float("-inf"),
            reverse=True,
        )
        return wins[:k]

    def recent_failures(self, k: int) -> list[MemoryEntry]:
        return self.failures()[-k:]

    # -- rendering ----------------------------------------------------------

    def to_markdown(self, *, top_k: int = 5) -> str:
        lines: list[str] = []
        wins = self.top_k(top_k)
        if wins:
            lines.append(f"### Top {len(wins)} by {self.score_metric or 'primary metric'}")
            for e in wins:
                score = f"{e.primary_score:.4f}" if e.primary_score is not None else "—"
                fam = f" [{e.architecture_family}]" if e.architecture_family else ""
                lines.append(f"- `{e.architecture_name}`{fam} — {score} ({e.experiment_id})")
        fails = self.recent_failures(max(1, top_k // 2))
        if fails:
            lines.append("\n### Recent failures")
            for e in fails:
                why = e.error_type or "—"
                msg = (e.error_message or "").split("\n", 1)[0][:80]
                lines.append(f"- `{e.architecture_name}` — {why}: {msg}")
        if not lines:
            lines.append("_no experiments yet_")
        return "\n".join(lines)

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_metric": self.score_metric,
            "entries": [asdict(e) for e in self.entries],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        (path.parent / "memory.md").write_text(self.to_markdown())

    @classmethod
    def load(cls, path: Path) -> "Memory":
        data = json.loads(path.read_text())
        return cls(
            score_metric=data.get("score_metric", ""),
            entries=[MemoryEntry(**e) for e in data.get("entries", [])],
        )


__all__ = ["Memory", "MemoryEntry"]
