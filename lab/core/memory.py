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
    # Compact per-epoch curve summary so the LLM can judge whether more
    # epochs would help. Example: "loss: 0.50→0.30→0.25 | f1_macro: 0.10→0.30→0.35 | trend: improving"
    curve_summary: str = ""
    epochs_trained: int | None = None

    @classmethod
    def from_experiment(cls, exp: Experiment) -> "MemoryEntry":
        err = exp.error
        curve, epochs = _summarize_curve(exp.history, exp.primary_metric)
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
            curve_summary=curve,
            epochs_trained=epochs,
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
                ep = f", {e.epochs_trained}ep" if e.epochs_trained else ""
                ckpt = " ✓ckpt" if e.checkpoint_path else ""
                lines.append(
                    f"- `{e.architecture_name}`{fam} — {score} ({e.experiment_id}{ep}{ckpt})"
                )
                if e.curve_summary:
                    lines.append(f"  curve: {e.curve_summary}")
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


def _summarize_curve(
    history: list[dict[str, Any]] | None,
    primary_metric: str,
) -> tuple[str, int | None]:
    """Return a compact one-line curve summary + epoch count.

    Designed to be token-cheap in prompts while giving the LLM enough
    signal to decide "continue training" vs "switch architecture".
    """
    if not history:
        return "", None
    epochs = len(history)

    def _fmt(v: object) -> str:
        if isinstance(v, (int, float)):
            return f"{float(v):.3f}"
        return "?"

    # Build per-metric arrow strings: "0.50→0.30→0.25"
    parts: list[str] = []
    for key in ("loss", primary_metric):
        if not key:
            continue
        vals = [h.get(key) for h in history if isinstance(h.get(key), (int, float))]
        if not vals:
            continue
        if len(vals) <= 5:
            arrow = "→".join(_fmt(v) for v in vals)
        else:
            arrow = "→".join(_fmt(v) for v in [vals[0], vals[len(vals) // 2], vals[-1]])
        parts.append(f"{key}: {arrow}")

    # Trend: is the metric still improving at the end?
    metric_vals = [
        float(h[primary_metric])
        for h in history
        if isinstance(h.get(primary_metric), (int, float))
    ]
    loss_vals = [
        float(h["loss"])
        for h in history
        if isinstance(h.get("loss"), (int, float))
    ]
    trend = "flat"
    if len(metric_vals) >= 2:
        last_half = metric_vals[len(metric_vals) // 2 :]
        if last_half[-1] > last_half[0]:
            trend = "improving"
        elif last_half[-1] < last_half[0] * 0.98:
            trend = "degrading"
    elif len(loss_vals) >= 2:
        if loss_vals[-1] < loss_vals[-2]:
            trend = "improving"

    parts.append(f"trend: {trend}")
    return " | ".join(parts), epochs


__all__ = ["Memory", "MemoryEntry"]
