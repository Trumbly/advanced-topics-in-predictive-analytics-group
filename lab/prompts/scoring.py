"""Cross-study aggregation of primary_score per (task, prompt version).

Drives the prompt A/B dashboard and the ``--use-best-prompts`` CLI switch.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from lab.core.loaders import list_studies, load_many


class PromptScoreStats(BaseModel):
    version: str
    mean: float
    stdev: float
    best: float
    count: int


def aggregate_prompt_scores(
    experiments_dir: Path,
) -> dict[str, dict[str, PromptScoreStats]]:
    """Walk every study under ``experiments_dir`` and aggregate scores."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    studies = load_many(Path(experiments_dir), list_studies(experiments_dir))
    for study in studies:
        version = _resolve_version(study)
        if version is None:
            continue
        for exp in study.experiments:
            if exp.primary_score is None:
                continue
            buckets[study.task_name][version].append(float(exp.primary_score))

    out: dict[str, dict[str, PromptScoreStats]] = {}
    for task, by_version in buckets.items():
        out[task] = {}
        for version, scores in by_version.items():
            out[task][version] = PromptScoreStats(
                version=version,
                mean=statistics.fmean(scores),
                stdev=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                best=max(scores),
                count=len(scores),
            )
    return out


def get_best_version(
    task: str,
    experiments_dir: Path,
    *,
    min_runs: int = 3,
) -> str | None:
    """Return the highest-mean version for ``task`` with at least ``min_runs``."""
    table = aggregate_prompt_scores(experiments_dir).get(task, {})
    eligible = [s for s in table.values() if s.count >= min_runs]
    if not eligible:
        return None
    eligible.sort(key=lambda s: (s.mean, s.count), reverse=True)
    return eligible[0].version


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_version(study) -> str | None:
    """Pull the propose_architecture version off `study.prompt_template_paths`."""
    paths = study.prompt_template_paths or {}
    propose = paths.get("propose_architecture")
    if propose is None:
        return None
    name = Path(str(propose)).stem  # 'v1' from 'v1.yaml'
    return name if name.startswith("v") else None
