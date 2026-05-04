"""Compact experiment memory rendered as ≤3 kB markdown (ADR-006 + ADR-007).

Stores the top-K successful experiments (by ``primary_score``) plus the N most
recent failures. Always serialisable to disk via ``save()`` / ``load()``.
``seed_from_predecessor`` carries memory across studies for the resume path;
``seed_from_agent_memory`` walks every saved study and seeds the global
top-K when the agent-memory toggle is on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from lab.core.models import Experiment, Study, StudyNotFoundError

_MAX_MARKDOWN_BYTES = 3072


class Memory:
    def __init__(self, top_k: int, recent_failures: int, path: Path):
        if top_k < 0 or recent_failures < 0:
            raise ValueError("top_k and recent_failures must be non-negative")
        self.top_k = top_k
        self.recent_failures = recent_failures
        self.path = Path(path)
        self._wins: list[Experiment] = []
        self._fails: list[Experiment] = []

    # ------------------------------------------------------------------
    # mutate
    # ------------------------------------------------------------------

    def add(self, exp: Experiment) -> None:
        if _is_success(exp):
            self._insert_win(exp)
        else:
            self._insert_failure(exp)

    def _insert_win(self, exp: Experiment) -> None:
        # replace existing entry if same id, else append
        self._wins = [e for e in self._wins if e.id != exp.id] + [exp]
        self._wins.sort(key=lambda e: (-(e.primary_score or 0.0), e.index))
        self._wins = self._wins[: self.top_k]

    def _insert_failure(self, exp: Experiment) -> None:
        self._fails = [e for e in self._fails if e.id != exp.id] + [exp]
        self._fails = self._fails[-self.recent_failures :]

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"## Top-{self.top_k} experiments")
        if not self._wins:
            lines.append("(none yet)")
        else:
            for i, exp in enumerate(self._wins, start=1):
                lines.append(self._win_line(i, exp))

        lines.append("")
        lines.append(f"## Recent failures (last {self.recent_failures})")
        if not self._fails:
            lines.append("(none yet)")
        else:
            for exp in self._fails:
                lines.append(self._fail_line(exp))

        md = "\n".join(lines).strip() + "\n"
        if len(md.encode("utf-8")) > _MAX_MARKDOWN_BYTES:
            md = self._trim_to_budget(md)
        return md

    @staticmethod
    def _win_line(rank: int, exp: Experiment) -> str:
        arch = (
            exp.proposal.architecture_name
            if exp.proposal is not None
            else "unknown"
        )
        family = exp.proposal.family if exp.proposal is not None else "unknown"
        score_repr = (
            f"{exp.primary_metric}={exp.primary_score:.4f}"
            if exp.primary_score is not None
            else f"{exp.primary_metric}=n/a"
        )
        curve = summarize_curve(exp.history, exp.primary_metric)
        return f"{rank}. {exp.id} · {arch} ({family}) · {score_repr} · {curve}"

    @staticmethod
    def _fail_line(exp: Experiment) -> str:
        arch = (
            exp.proposal.architecture_name
            if exp.proposal is not None
            else "unknown"
        )
        err = _first_error(exp)
        if err is None:
            return f"- {exp.id} · {arch} · {exp.status}"
        return f"- {exp.id} · {arch} · {err.error_type}: {err.message[:200]}"

    @staticmethod
    def _trim_to_budget(md: str) -> str:
        # Drop trailing bullets until under budget.
        kept_lines: list[str] = []
        running = 0
        for line in md.splitlines():
            chunk = (line + "\n").encode("utf-8")
            if running + len(chunk) > _MAX_MARKDOWN_BYTES:
                break
            kept_lines.append(line)
            running += len(chunk)
        return "\n".join(kept_lines).rstrip() + "\n"

    # ------------------------------------------------------------------
    # seeding
    # ------------------------------------------------------------------

    def seed_from_predecessor(self, predecessor: Study) -> None:
        for exp in predecessor.experiments:
            self.add(exp)

    def seed_from_agent_memory(self, studies_root: Path, task: str) -> None:
        from lab.core.loaders import list_studies, load_many

        ids = list_studies(studies_root)
        for st in load_many(studies_root, ids):
            if st.task_name != task:
                continue
            for exp in st.experiments:
                if _is_success(exp):
                    self._insert_win(exp)

    # ------------------------------------------------------------------
    # disk
    # ------------------------------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "top_k": self.top_k,
            "recent_failures": self.recent_failures,
            "wins": [json.loads(e.model_dump_json()) for e in self._wins],
            "fails": [json.loads(e.model_dump_json()) for e in self._fails],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, top_k: int, recent_failures: int) -> "Memory":
        m = cls(top_k=top_k, recent_failures=recent_failures, path=Path(path))
        if not m.path.exists():
            return m
        payload = json.loads(m.path.read_text(encoding="utf-8"))
        m._wins = [Experiment.model_validate(e) for e in payload.get("wins", [])]
        m._fails = [Experiment.model_validate(e) for e in payload.get("fails", [])]
        # apply current caps in case the saved file used different ones
        m._wins.sort(key=lambda e: (-(e.primary_score or 0.0), e.index))
        m._wins = m._wins[: m.top_k]
        m._fails = m._fails[-m.recent_failures :]
        return m

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def has_checkpoint_for(self, experiment_id: str) -> bool:
        for exp in self._wins:
            if exp.id == experiment_id and exp.checkpoint_path:
                return True
        return False

    @property
    def wins(self) -> list[Experiment]:
        return list(self._wins)

    @property
    def failures(self) -> list[Experiment]:
        return list(self._fails)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_success(exp: Experiment) -> bool:
    return (
        exp.primary_score is not None
        and exp.status in {"COMPLETED", "JUDGED"}
    )


def _first_error(exp: Experiment):
    for task in exp.tasks:
        if task.error is not None:
            return task.error
    return None


def summarize_curve(history: list[dict[str, Any]], metric: str) -> str:
    """Render a one-line curve summary: ``loss A→B→C | metric X→Y→Z | trend``."""
    if not history:
        return "no history"

    losses = [h.get("loss") for h in history if isinstance(h.get("loss"), (int, float))]
    scores = [h.get(metric) for h in history if isinstance(h.get(metric), (int, float))]

    parts: list[str] = []
    if losses:
        parts.append("loss " + _arrow_repr(losses))
    if scores:
        parts.append(f"{metric} " + _arrow_repr(scores))
    if not parts:
        return "no usable curve"

    parts.append(f"trend {_classify_trend(scores or losses, higher_is_better=bool(scores))}")
    return " | ".join(parts)


def _arrow_repr(values: list[float]) -> str:
    if len(values) <= 3:
        return "→".join(f"{v:.2f}" for v in values)
    # show first, mid, last
    return "→".join(f"{v:.2f}" for v in (values[0], values[len(values) // 2], values[-1]))


def _classify_trend(values: list[float], *, higher_is_better: bool) -> str:
    if len(values) < 2:
        return "flat"
    delta = values[-1] - values[0]
    threshold = max(0.005, 0.02 * abs(values[0] or 1.0))
    if abs(delta) < threshold:
        return "flat"
    if (delta > 0) == higher_is_better:
        return "improving"
    return "regressing"
