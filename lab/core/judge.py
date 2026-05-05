"""Per-experiment + per-study judge role (ADR-009).

Renders the dedicated judge prompts and parses strict JSON Verdicts. On parse
failure the model is re-asked once with the validation error appended (same
pattern as ADR-018).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from lab.core.llm import LLMClient
from lab.core.memory import Memory, summarize_curve
from lab.core.models import Experiment, Study, Verdict
from lab.prompts.engine import PromptEngine


class JudgeError(Exception):
    """Raised when the judge cannot produce a valid Verdict after one retry."""


class Judge:
    def __init__(self, client: LLMClient, engine: PromptEngine):
        self.client = client
        self.engine = engine

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    def judge_experiment(self, exp: Experiment, memory: Memory) -> Verdict:
        slots = self._experiment_slots(exp, memory)
        return self._chat_to_verdict("judge_experiment", slots)

    def judge_study(self, study: Study, *, budget_used: int, budget_total: int) -> Verdict:
        slots = self._study_slots(study, budget_used=budget_used, budget_total=budget_total)
        return self._chat_to_verdict("judge_study", slots)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _chat_to_verdict(self, task: str, slots: dict[str, str]) -> Verdict:
        sys_prompt, user_prompt = self.engine.render(task, slots)  # type: ignore[arg-type]
        first = self.client.chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        try:
            return _parse_verdict(first)
        except (json.JSONDecodeError, ValidationError) as exc:
            retry_user = (
                user_prompt
                + f"\n\nYour previous response could not be parsed: {exc}.\n"
                "Reply with ONLY the JSON verdict object, no prose, no fences."
            )
            second = self.client.chat(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": retry_user},
                ]
            )
            try:
                return _parse_verdict(second)
            except (json.JSONDecodeError, ValidationError) as exc2:
                raise JudgeError(f"judge produced invalid JSON twice: {exc2}") from exc2

    # ------------------------------------------------------------------
    # slot builders
    # ------------------------------------------------------------------

    @staticmethod
    def _experiment_slots(exp: Experiment, memory: Memory) -> dict[str, str]:
        arch = exp.proposal.architecture_name if exp.proposal else "unknown"
        family = exp.proposal.family if exp.proposal else "unknown"
        score = "n/a" if exp.primary_score is None else f"{exp.primary_score:.4f}"
        curve = summarize_curve(exp.history, exp.primary_metric)
        err = "(none)"
        for task in exp.tasks:
            if task.error is not None:
                err = f"{task.error.error_type}: {task.error.message[:300]}"
                break
        return {
            "architecture_name": arch,
            "architecture_family": family,
            "primary_metric": exp.primary_metric,
            "primary_score": score,
            "curve_summary": curve,
            "error": err,
            "experiment_memory": memory.to_markdown(),
            "personality": "exploratory",  # study-level slot supplied by caller in practice
        }

    @staticmethod
    def _study_slots(
        study: Study, *, budget_used: int, budget_total: int
    ) -> dict[str, str]:
        return {
            "top_experiments_table": _top_table(study),
            "failure_breakdown": _failure_breakdown(study),
            "budget_used": str(budget_used),
            "budget_total": str(budget_total),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_verdict(text: str) -> Verdict:
    cleaned = _strip_code_fences(text).strip()
    payload = json.loads(cleaned)
    return Verdict.model_validate(payload)


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text
    chunks = text.split("```")
    if len(chunks) >= 3:
        body = chunks[1]
        if body.startswith("json\n"):
            body = body[len("json\n") :]
        return body
    return text


def _top_table(study: Study, top_n: int = 5) -> str:
    successes = [
        e for e in study.experiments if e.primary_score is not None
    ]
    successes.sort(key=lambda e: e.primary_score or 0.0, reverse=True)
    rows: list[str] = ["| # | id | arch | family | score |", "|---|---|---|---|---|"]
    for i, e in enumerate(successes[:top_n], start=1):
        arch = e.proposal.architecture_name if e.proposal else "?"
        family = e.proposal.family if e.proposal else "?"
        rows.append(
            f"| {i} | {e.id} | {arch} | {family} | {e.primary_score:.4f} |"
        )
    if len(rows) == 2:
        rows.append("| - | (none) | - | - | - |")
    return "\n".join(rows)


def _failure_breakdown(study: Study) -> str:
    counts: dict[str, int] = {}
    for e in study.experiments:
        if e.primary_score is not None:
            continue
        for task in e.tasks:
            if task.error is not None:
                counts[task.error.error_type] = counts.get(task.error.error_type, 0) + 1
                break
    if not counts:
        return "(none)"
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
