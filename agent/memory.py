"""ExperimentMemory — structured history of all experiments in a Study.

Responsibilities
----------------
- Keep every Experiment the orchestrator has run
- Persist to disk as JSON (queryable, machine-readable)
- Render a compact Markdown summary for LLM prompt injection (token-efficient)
- Expose top-k retrieval by score + recent failures for error-recovery prompts
- Signal "empty" to the ContextHandler so cold-start experiments use the
  fallback prompt

Storage layout
--------------
Inside a Study directory:
    experiments/studies/<study_id>/
        memory.json   — structured list of Experiment JSON blobs
        memory.md     — human-readable markdown (for debugging + the report)

Only `memory.json` is the source of truth. `memory.md` is regenerated on
every `save()` call. Neither is committed to git — both are listed in
`.gitignore`.

Implements the `MemoryLike` Protocol from `agent.context_handler`, so it
drops directly into the ContextHandler without changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from agent.models import Experiment, ExperimentStatus


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class ExperimentMemory:
    """In-memory + on-disk store of experiments belonging to one Study."""

    study_dir: Path
    score_metric: str = "f1_macro"
    _experiments: list[Experiment] = field(default_factory=list)

    # -- lifecycle ----------------------------------------------------------

    def __post_init__(self) -> None:
        self.study_dir = Path(self.study_dir)
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.load()

    @property
    def json_path(self) -> Path:
        return self.study_dir / "memory.json"

    @property
    def markdown_path(self) -> Path:
        return self.study_dir / "memory.md"

    def load(self) -> None:
        """Load from disk if the file exists, otherwise start empty."""
        if not self.json_path.exists():
            self._experiments = []
            return
        raw = json.loads(self.json_path.read_text())
        self._experiments = [Experiment.model_validate(item) for item in raw]

    def save(self) -> None:
        """Persist both memory.json and memory.md."""
        self.json_path.write_text(
            json.dumps(
                [exp.model_dump(mode="json") for exp in self._experiments],
                indent=2,
                default=str,
            )
        )
        self.markdown_path.write_text(self.to_markdown(top_k=len(self._experiments)))

    # -- mutation -----------------------------------------------------------

    def append(self, experiment: Experiment) -> None:
        """Add a new experiment. Replaces any existing entry with the same ID."""
        # Drop any stale copy so reruns overwrite cleanly
        self._experiments = [
            e for e in self._experiments if e.experiment_id != experiment.experiment_id
        ]
        self._experiments.append(experiment)
        self.save()

    def extend(self, experiments: Iterable[Experiment]) -> None:
        for exp in experiments:
            self.append(exp)

    # -- queries ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._experiments)

    def is_empty(self) -> bool:
        return len(self._experiments) == 0

    def all(self) -> list[Experiment]:
        return list(self._experiments)

    def last(self) -> Experiment | None:
        return self._experiments[-1] if self._experiments else None

    def get(self, experiment_id: str) -> Experiment | None:
        for exp in self._experiments:
            if exp.experiment_id == experiment_id:
                return exp
        return None

    def successes(self) -> list[Experiment]:
        """Return only experiments that completed with results."""
        return [
            e
            for e in self._experiments
            if e.status == ExperimentStatus.COMPLETED.value and e.results is not None
        ]

    def failures(self) -> list[Experiment]:
        """Return experiments that failed, timed out, or are missing results."""
        bad = (
            ExperimentStatus.FAILED.value,
            ExperimentStatus.TIMEOUT.value,
        )
        return [e for e in self._experiments if e.status in bad]

    def top_k(self, k: int, *, by: str | None = None) -> list[Experiment]:
        """Return the top K successful experiments sorted by the given metric.

        Ties are broken by more recent experiments first (stable sort).
        """
        metric = by or self.score_metric
        successes = self.successes()

        def score_of(exp: Experiment) -> float:
            if exp.results is None:
                return float("-inf")
            return float(exp.results.metrics.get(metric, float("-inf")))

        ranked = sorted(successes, key=score_of, reverse=True)
        return ranked[:k]

    def recent_failures(self, k: int) -> list[Experiment]:
        """Return the most recent K failures (newest first)."""
        return list(reversed(self.failures()))[:k]

    def best(self) -> Experiment | None:
        """Return the single best experiment, or None."""
        top = self.top_k(1)
        return top[0] if top else None

    def best_score(self) -> float | None:
        best = self.best()
        if best is None or best.results is None:
            return None
        return best.results.metrics.get(self.score_metric)

    # -- serialization ------------------------------------------------------

    def to_markdown(self, top_k: int = 5) -> str:
        """Compact markdown summary for LLM prompt injection.

        Shows the top-K best experiments and the most recent failures, so
        the LLM gets both "what works" and "what to avoid". Token-efficient:
        every experiment is at most ~6 lines.
        """
        if not self._experiments:
            return ""

        top = self.top_k(top_k)
        recent_fails = self.recent_failures(max(1, top_k // 2))

        lines: list[str] = []
        if top:
            lines.append("### Top-K Successful Experiments")
            for exp in top:
                lines.extend(_render_experiment(exp, include_metrics=True))
                lines.append("")
        if recent_fails:
            lines.append("### Recent Failures")
            for exp in recent_fails:
                lines.extend(_render_experiment(exp, include_metrics=False))
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_experiment(exp: Experiment, *, include_metrics: bool) -> list[str]:
    """One-experiment markdown block. ~5 lines per experiment."""
    lines = [f"- **{exp.experiment_id}** [{exp.status}]"]
    if exp.config:
        arch_parts = [f"arch: {exp.config.architecture}"]
        if exp.config.pretrained_model:
            arch_parts.append(f"pretrained: {exp.config.pretrained_model}")
        lines.append("  - " + " | ".join(arch_parts))
        if exp.config.hyperparams:
            hp = ", ".join(f"{k}={v}" for k, v in exp.config.hyperparams.items())
            lines.append(f"  - hyperparams: {hp}")
    if include_metrics and exp.results:
        metrics = ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(exp.results.metrics.items())
        )
        lines.append(f"  - metrics: {metrics}")
        lines.append(f"  - duration: {exp.results.duration_seconds:.1f}s")
    # Failures carry info in the experiment-level status; detailed errors live
    # at the Task level and are pulled by the error_recovery prompt separately.
    return lines


__all__ = ["ExperimentMemory"]
