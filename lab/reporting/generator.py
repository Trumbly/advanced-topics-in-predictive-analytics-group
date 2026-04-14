"""Render a study report.

Template-first: a Jinja2 markdown template drives the full structure. The
only piece the LLM writes is the single executive-summary paragraph, and
that piece is optional — if the LLM is unreachable a deterministic
fallback paragraph is used so the report always renders.
"""
from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from lab.config import Settings
from lab.core.llm import LLMClient, LLMError, format_messages
from lab.core.models import Study
from lab.prompts.engine import PromptEngine
from lab.prompts.registry import PromptRegistry
from lab.reporting import figures


logger = logging.getLogger("lab.report")


@dataclass
class FamilyRow:
    family: str
    count: int
    best: float
    mean: float


@dataclass
class FailureRow:
    type: str
    count: int
    example: str


def generate_report(
    study: Study,
    settings: Settings,
    *,
    write_exec_summary: bool = True,
) -> Path:
    experiments_dir = settings.abspath(settings.paths.experiments)
    out_dir = experiments_dir / study.id / "report"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig_paths = figures.render_all(study, fig_dir)
    except Exception:  # noqa: BLE001
        logger.exception("Figure rendering failed; continuing without figures.")
        fig_paths = []

    n_success = sum(1 for e in study.experiments if e.primary_score is not None)
    n_failed = sum(1 for e in study.experiments if e.error is not None)

    best_experiment = None
    if study.best_experiment_id:
        best_experiment = next(
            (e for e in study.experiments if e.id == study.best_experiment_id), None,
        )

    # Family stats
    by_family: dict[str, list[float]] = defaultdict(list)
    for e in study.experiments:
        if e.primary_score is None:
            continue
        by_family[e.architecture_family or "unknown"].append(e.primary_score)
    family_rows = [
        FamilyRow(
            family=k,
            count=len(v),
            best=max(v),
            mean=statistics.fmean(v),
        )
        for k, v in sorted(by_family.items(), key=lambda kv: -max(kv[1]))
    ]

    # Failure analysis
    by_err: dict[str, list[str]] = defaultdict(list)
    for e in study.experiments:
        if e.error:
            by_err[e.error.error_type].append(e.error.message)
    failure_rows = [
        FailureRow(
            type=k,
            count=len(msgs),
            example=(msgs[0] or "")[:100].replace("\n", " "),
        )
        for k, msgs in sorted(by_err.items(), key=lambda kv: -len(kv[1]))
    ]

    exec_summary = ""
    if write_exec_summary:
        exec_summary = _render_exec_summary(study, settings, family_rows, n_success, n_failed)

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=False, lstrip_blocks=False,
    )
    template = env.get_template("study_report.md.j2")
    md = template.render(
        study=study,
        exec_summary=exec_summary,
        best_experiment=best_experiment,
        n_success=n_success, n_failed=n_failed,
        family_rows=family_rows, failure_rows=failure_rows,
        has_learning_curve=(fig_dir / "best_learning_curve.png").exists(),
        has_family_box=(fig_dir / "per_family_box.png").exists(),
    )
    out_path = out_dir / "report.md"
    out_path.write_text(md)
    return out_path


def _render_exec_summary(
    study: Study,
    settings: Settings,
    family_rows: list[FamilyRow],
    n_success: int,
    n_failed: int,
) -> str:
    """One paragraph via LLM; fall back to a deterministic string on error."""
    best_score = study.best_score if study.best_score is not None else 0.0
    best = next((e for e in study.experiments if e.id == study.best_experiment_id), None)
    fam_summary = "\n".join(
        f"- {r.family}: best={r.best:.4f} mean={r.mean:.4f} ({r.count} runs)"
        for r in family_rows
    ) or "- none —"

    try:
        registry = PromptRegistry(settings.abspath(settings.paths.prompts_dir))
        engine = PromptEngine(registry)
        system, user = engine.render(
            "executive_summary",
            {
                "task_description": study.task_name,
                "n_experiments": len(study.experiments),
                "n_success": n_success,
                "n_failed": n_failed,
                "primary_metric": study.primary_metric,
                "best_score": best_score,
                "best_architecture": (best.architecture_name if best else "n/a"),
                "family_summary": fam_summary,
            },
        )
        llm = LLMClient(
            base_url=settings.llm.base_url,
            model=settings.llm.default_model,
            api_key=settings.llm.api_key,
            provider=settings.llm.provider,
            temperature=0.3,
            max_tokens=400,
            timeout_seconds=settings.llm.timeout_seconds,
            retry_attempts=1,
        )
        return llm.chat(format_messages(system, user)).strip()
    except (LLMError, Exception) as exc:  # noqa: BLE001
        logger.info("Using deterministic exec summary fallback: %s", exc)
        return (
            f"Across {len(study.experiments)} experiments ({n_success} successful, "
            f"{n_failed} failed), the best configuration achieved "
            f"{study.primary_metric} = {best_score:.4f}"
            + (f" with architecture `{best.architecture_name}`" if best else "")
            + ". See the per-family breakdown below for which architecture families "
              "dominated and which failed modes were most common."
        )


__all__ = ["generate_report"]
