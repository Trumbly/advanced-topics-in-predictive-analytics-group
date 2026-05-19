"""Report generator: figures + Jinja2 markdown + HTML."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape

from lab.config import Settings
from lab.core.loaders import list_studies, load_many
from lab.core.models import Study
from lab.reporting.figures import render_figures
from lab.ui.display import build_study_ordinals, exp_label, study_label

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_report(study: Study, settings: Settings) -> Path:
    """Render the full study report (markdown + HTML) and return the markdown path."""
    out_dir = Path(settings.paths.experiments_dir) / study.id
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = render_figures(study, out_dir)
    figures_relative = {k: v.name for k, v in figures.items()}

    experiments_root = Path(settings.paths.experiments_dir)
    all_studies = load_many(experiments_root, list_studies(experiments_root))
    ordinals = build_study_ordinals(all_studies)
    ord_n = ordinals.get(study.id, len(ordinals) + 1)

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape([]),
        trim_blocks=False,
        lstrip_blocks=False,
    )
    template = env.get_template("report.md.j2")

    md = template.render(
        study=study,
        study_label=study_label(study.id, ord_n),
        exp_label_for=exp_label,
        figures=figures_relative,
        top_experiments=_top_experiments(study, n=5),
        failure_breakdown=_failure_breakdown(study),
        executive_summary=_summary(study),
    )

    md_path = out_dir / "report.md"
    md_path.write_text(md, encoding="utf-8")

    html_path = out_dir / "report.html"
    try:
        from markdown_it import MarkdownIt  # type: ignore[import-not-found]

        html_path.write_text(MarkdownIt().render(md), encoding="utf-8")
    except ImportError:  # pragma: no cover - optional dep
        html_path.write_text(f"<pre>{md}</pre>", encoding="utf-8")

    return md_path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _top_experiments(study: Study, *, n: int = 5) -> list:
    successes = [e for e in study.experiments if e.primary_score is not None]
    successes.sort(key=lambda e: e.primary_score or 0.0, reverse=True)
    return successes[:n]


def _failure_breakdown(study: Study) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for exp in study.experiments:
        if exp.primary_score is not None:
            continue
        for task in exp.tasks:
            if task.error is not None:
                counts[task.error.error_type] += 1
                break
    return dict(counts)


def _summary(study: Study) -> str:
    if not study.experiments:
        return "No experiments were recorded for this study."

    completed = [e for e in study.experiments if e.primary_score is not None]
    if not completed:
        return f"Study completed {len(study.experiments)} experiments; none produced a primary score."

    best = max(completed, key=lambda e: e.primary_score or 0.0)
    family = best.proposal.family if best.proposal else "unknown"
    return (
        f"Study completed {len(study.experiments)} experiments "
        f"({len(completed)} scored). Best so far: {best.id} "
        f"({family}) at {best.primary_metric}={best.primary_score:.4f}."
    )
