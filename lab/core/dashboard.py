"""Dashboard KPI aggregation across every saved study.

Drives the UI `/dashboard` route and the `/api/dashboard` JSON endpoint.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from lab.core.codegen_rates import (
    ModelCodegenStats,
    aggregate_model_codegen,
)
from lab.core.llm_metrics import aggregate_by_model
from lab.core.loaders import list_studies, load_many
from lab.core.models import Experiment, Study
from lab.prompts.scoring import (
    PromptScoreStats,
    aggregate_prompt_scores_by_prompt_task,
)

_FAILED_STUDY_STATUSES = {"FAILED", "ABORTED"}


class BestModel(BaseModel):
    experiment_id: str
    study_id: str
    architecture_name: str
    family: str
    primary_metric: str
    primary_score: float


class PromptKPI(BaseModel):
    """Per prompt-task: best version (with min_runs filter) + per-version stats."""

    best_version: str | None
    versions: list[PromptScoreStats]


class LLMModelMetrics(BaseModel):
    """One row in the dashboard's per-model generation-metrics table."""

    model: str
    n_calls: int
    total_completion_tokens: int
    total_seconds: float
    tps_avg: float | None
    tps_min: float | None
    tps_max: float | None
    ttft_avg: float | None
    ttft_min: float | None
    ttft_max: float | None


class DashboardKPIs(BaseModel):
    total_studies: int
    total_experiments: int
    experiment_error_rate: float  # fraction of experiments that failed/never scored
    study_error_rate: float       # fraction of studies that ended FAILED/ABORTED
    best_roc_auc: float | None
    best_f1: float | None
    best_model: BestModel | None
    best_prompts: dict[str, PromptKPI]
    model_codegen: dict[str, ModelCodegenStats]
    llm_metrics: list[LLMModelMetrics]


def compute_kpis(experiments_dir: Path, *, min_runs: int = 3) -> DashboardKPIs:
    studies = load_many(experiments_dir, list_studies(experiments_dir))
    total_studies = len(studies)

    all_exps: list[tuple[Study, Experiment]] = [
        (s, e) for s in studies for e in s.experiments
    ]
    total_experiments = len(all_exps)

    failed_exps = sum(1 for _, e in all_exps if not _is_scored(e))
    experiment_error_rate = (
        failed_exps / total_experiments if total_experiments else 0.0
    )

    failed_studies = sum(
        1
        for s in studies
        if s.status in _FAILED_STUDY_STATUSES
        or (s.experiments and not any(_is_scored(e) for e in s.experiments))
    )
    study_error_rate = failed_studies / total_studies if total_studies else 0.0

    aucs = [
        e.metrics["roc_auc_macro"]
        for _, e in all_exps
        if isinstance(e.metrics.get("roc_auc_macro"), (int, float))
    ]
    f1s = [
        e.metrics["f1_macro"]
        for _, e in all_exps
        if isinstance(e.metrics.get("f1_macro"), (int, float))
    ]

    best_model = _best_model(all_exps)
    best_prompts = _best_prompts(experiments_dir, min_runs=min_runs)

    return DashboardKPIs(
        total_studies=total_studies,
        total_experiments=total_experiments,
        experiment_error_rate=experiment_error_rate,
        study_error_rate=study_error_rate,
        best_roc_auc=max(aucs) if aucs else None,
        best_f1=max(f1s) if f1s else None,
        best_model=best_model,
        best_prompts=best_prompts,
        model_codegen=aggregate_model_codegen(experiments_dir),
        llm_metrics=[
            LLMModelMetrics(
                model=m.model,
                n_calls=m.summary.n_calls,
                total_completion_tokens=m.summary.total_completion_tokens,
                total_seconds=round(m.summary.total_seconds, 2),
                tps_avg=_round(m.summary.tps_avg),
                tps_min=_round(m.summary.tps_min),
                tps_max=_round(m.summary.tps_max),
                ttft_avg=_round(m.summary.ttft_avg),
                ttft_min=_round(m.summary.ttft_min),
                ttft_max=_round(m.summary.ttft_max),
            )
            for m in aggregate_by_model(studies)
        ],
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_scored(exp: Experiment) -> bool:
    """A "successful" experiment for KPI purposes.

    An experiment counts as scored only when:
      - it was not marked FAILED/ABORTED at runtime,
      - training produced a numeric primary_score, AND
      - the judge did not return a "discard" verdict (which means the
        experiment ran but the score is not trustworthy -- e.g. the channel
        mismatch crash, or a NaN-loss flat-line). Without this guard,
        "DISCARDED" experiments inflate the success rate and `best_*` KPIs.
    """
    if exp.primary_score is None or exp.status in _FAILED_STUDY_STATUSES:
        return False
    if exp.verdict is not None and exp.verdict.verdict in {"discard", "abort_study"}:
        return False
    return True


def _round(value: float | None, ndigits: int = 2) -> float | None:
    return None if value is None else round(value, ndigits)


def _best_model(pairs: list[tuple[Study, Experiment]]) -> BestModel | None:
    scored = [(s, e) for s, e in pairs if _is_scored(e)]
    if not scored:
        return None
    s, e = max(scored, key=lambda se: se[1].primary_score or 0.0)
    return BestModel(
        experiment_id=e.id,
        study_id=s.id,
        architecture_name=e.proposal.architecture_name if e.proposal else "?",
        family=e.proposal.family if e.proposal else "?",
        primary_metric=e.primary_metric,
        primary_score=e.primary_score or 0.0,
    )


def _best_prompts(
    experiments_dir: Path, *, min_runs: int
) -> dict[str, PromptKPI]:
    table = aggregate_prompt_scores_by_prompt_task(experiments_dir)
    out: dict[str, PromptKPI] = {}
    for prompt_task, by_version in table.items():
        rows = sorted(
            by_version.values(),
            key=lambda s: (s.mean, s.count),
            reverse=True,
        )
        eligible = [r for r in rows if r.count >= min_runs]
        best_version = eligible[0].version if eligible else None
        out[prompt_task] = PromptKPI(best_version=best_version, versions=rows)
    return out
