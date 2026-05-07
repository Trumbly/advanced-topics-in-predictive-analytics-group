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

_DEFAULT_MAX_MARKDOWN_BYTES = 8192


class Memory:
    def __init__(
        self,
        top_k: int,
        recent_failures: int,
        path: Path,
        *,
        max_markdown_bytes: int = _DEFAULT_MAX_MARKDOWN_BYTES,
    ):
        if top_k < 0 or recent_failures < 0:
            raise ValueError("top_k and recent_failures must be non-negative")
        self.top_k = top_k
        self.recent_failures = recent_failures
        self.path = Path(path)
        self.max_markdown_bytes = max_markdown_bytes
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
                lines.extend(self._win_block(i, exp))

        lines.append("")
        lines.append(f"## Recent failures (last {self.recent_failures})")
        if not self._fails:
            lines.append("(none yet)")
        else:
            for exp in self._fails:
                lines.extend(self._fail_block(exp))

        md = "\n".join(lines).strip() + "\n"
        if len(md.encode("utf-8")) > self.max_markdown_bytes:
            md = self._trim_to_budget(md)
        return md

    def _win_block(self, rank: int, exp: Experiment) -> list[str]:
        """Multi-line per-win entry: header + hyperparams + trajectory +
        verdict rationale + checkpoint pointer. Lets the LLM judge whether
        a continue-training proposal makes sense without re-reading
        ``study.json`` from disk."""
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
        out = [f"### {rank}. {exp.id} · {arch} ({family}) · {score_repr}"]

        if exp.proposal is not None:
            p = exp.proposal
            wd = (
                f", wd={p.weight_decay:.0e}"
                if p.weight_decay is not None
                else ""
            )
            init = (
                f", init_from={p.init_from_experiment_id}"
                if p.init_from_experiment_id
                else ""
            )
            cont = (
                f", continued_from={p.continue_from_experiment_id}"
                if p.continue_from_experiment_id
                else ""
            )
            out.append(
                f"- hp: lr={p.lr:.0e}, sched={p.lr_schedule}, "
                f"epochs={p.epochs}{wd}{init}{cont}"
            )

        out.append("- trajectory: " + summarize_curve(exp.history, exp.primary_metric))

        # Last-3-epoch detail so the LLM can judge "still improving" vs
        # "plateauing" beyond the one-line trend.
        recent = _recent_epochs(exp.history, exp.primary_metric, n=3)
        if recent:
            out.append("- recent epochs: " + recent)

        if exp.verdict is not None:
            out.append(
                f"- verdict: **{exp.verdict.verdict}** "
                f"(conf {exp.verdict.score:.2f}) — {exp.verdict.rationale[:240]}"
            )

        if exp.checkpoint_path:
            out.append(f"- checkpoint available → can `continue_from`: {exp.id}")

        out.append("")  # blank line between entries
        return out

    def _fail_block(self, exp: Experiment) -> list[str]:
        arch = (
            exp.proposal.architecture_name
            if exp.proposal is not None
            else "unknown"
        )
        err = _first_error(exp)
        out = [f"- **{exp.id}** · {arch} · {exp.status}"]
        if err is not None:
            out.append(f"  - {err.error_type}: {err.message[:240]}")
        # Surface the recover attempts so the LLM does not propose the
        # exact same architecture that just failed N times in a row.
        n_recover = sum(1 for t in exp.tasks if t.name == "recover")
        if n_recover:
            out.append(f"  - {n_recover} recovery attempt(s) before giving up")
        return out

    def _trim_to_budget(self, md: str) -> str:
        # Drop trailing bullets until under budget.
        kept_lines: list[str] = []
        running = 0
        for line in md.splitlines():
            chunk = (line + "\n").encode("utf-8")
            if running + len(chunk) > self.max_markdown_bytes:
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


def _recent_epochs(history: list[dict[str, Any]], metric: str, *, n: int = 3) -> str:
    """Render the last ``n`` epoch rows as ``e7: loss=0.34, metric=0.71``.

    Lets the LLM see fresh signal beyond the single-line trend arrow —
    useful when judging whether a few more epochs would push the model
    over a plateau vs. it having genuinely stalled.
    """
    if not history:
        return ""
    tail = history[-n:]
    parts: list[str] = []
    for h in tail:
        ep = h.get("epoch")
        loss = h.get("loss")
        score = h.get(metric)
        chunk = f"e{ep}" if ep is not None else "e?"
        if isinstance(loss, (int, float)):
            chunk += f": loss={loss:.3f}"
        if isinstance(score, (int, float)):
            sep = ", " if isinstance(loss, (int, float)) else ": "
            chunk += f"{sep}{metric}={score:.3f}"
        parts.append(chunk)
    return " | ".join(parts)


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
