"""Aggregate LLM-call generation metrics across studies and experiments.

Each Task created by the orchestrator carries an optional
``output["llm_stats"]`` entry written by ``lab.core.experiment._stats_dict``;
this module walks those entries and computes per-experiment / per-study /
per-model summaries that the dashboard renders.

All numeric outputs are rounded to two decimals.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from lab.core.models import Experiment, Study


@dataclass(frozen=True)
class LLMCallSummary:
    """Aggregate over a set of LLM calls."""

    n_calls: int
    total_completion_tokens: int
    total_prompt_tokens: int
    total_seconds: float
    tps_avg: float | None
    tps_min: float | None
    tps_max: float | None
    ttft_avg: float | None
    ttft_min: float | None
    ttft_max: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "n_calls": self.n_calls,
            "total_completion_tokens": self.total_completion_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_seconds": round(self.total_seconds, 2),
            "tps_avg": _round(self.tps_avg),
            "tps_min": _round(self.tps_min),
            "tps_max": _round(self.tps_max),
            "ttft_avg": _round(self.ttft_avg),
            "ttft_min": _round(self.ttft_min),
            "ttft_max": _round(self.ttft_max),
        }


@dataclass(frozen=True)
class ModelMetrics:
    """Per-model breakdown for the dashboard."""

    model: str
    summary: LLMCallSummary


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def iter_call_stats(experiment: Experiment) -> Iterable[dict[str, object]]:
    """Yield each ``llm_stats`` dict attached to a task in ``experiment``."""
    for task in experiment.tasks:
        stats = task.output.get("llm_stats") if task.output else None
        if isinstance(stats, dict):
            yield stats


def iter_study_call_stats(study: Study) -> Iterable[dict[str, object]]:
    """Yield ``llm_stats`` dicts across every experiment in ``study``."""
    for exp in study.experiments:
        yield from iter_call_stats(exp)


# ---------------------------------------------------------------------------
# summarisation
# ---------------------------------------------------------------------------


def summarise(stats_list: Iterable[dict[str, object]]) -> LLMCallSummary:
    """Reduce a stream of llm_stats dicts into a single summary."""
    n = 0
    total_completion = 0
    total_prompt = 0
    total_seconds = 0.0
    tps_values: list[float] = []
    ttft_values: list[float] = []
    for s in stats_list:
        n += 1
        total_completion += int(s.get("completion_tokens") or 0)
        total_prompt += int(s.get("prompt_tokens") or 0)
        total_seconds += float(s.get("total_seconds") or 0.0)
        tps = s.get("tps")
        if isinstance(tps, (int, float)):
            tps_values.append(float(tps))
        ttft = s.get("ttft_seconds")
        if isinstance(ttft, (int, float)):
            ttft_values.append(float(ttft))
    return LLMCallSummary(
        n_calls=n,
        total_completion_tokens=total_completion,
        total_prompt_tokens=total_prompt,
        total_seconds=total_seconds,
        tps_avg=_avg(tps_values),
        tps_min=min(tps_values) if tps_values else None,
        tps_max=max(tps_values) if tps_values else None,
        ttft_avg=_avg(ttft_values),
        ttft_min=min(ttft_values) if ttft_values else None,
        ttft_max=max(ttft_values) if ttft_values else None,
    )


def experiment_summary(experiment: Experiment) -> LLMCallSummary:
    return summarise(iter_call_stats(experiment))


def study_summary(study: Study) -> LLMCallSummary:
    return summarise(iter_study_call_stats(study))


def aggregate_by_model(studies: Iterable[Study]) -> list[ModelMetrics]:
    """Per-model summary across all calls in ``studies``.

    Groups by the ``model`` field stored on each call (which carries the
    actual provider tag the LLM client used at the moment of the call) and
    sorts the result by call count, descending.
    """
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for study in studies:
        for stats in iter_study_call_stats(study):
            model = str(stats.get("model") or "unknown")
            by_model[model].append(stats)
    out = [
        ModelMetrics(model=model, summary=summarise(call_stats))
        for model, call_stats in by_model.items()
    ]
    out.sort(key=lambda m: m.summary.n_calls, reverse=True)
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None, ndigits: int = 2) -> float | None:
    return None if value is None else round(value, ndigits)
