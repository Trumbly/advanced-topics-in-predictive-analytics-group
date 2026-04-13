"""Score aggregation for prompt A/B testing.

Scans all studies, reads which prompt version each study used per task,
and attributes experiment scores to those versions. Returns aggregate
stats (mean, count) per task × version combination.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PromptScoreStats:
    """Aggregate scores for one prompt version used in one task."""

    task_name: str
    version: str
    mean_score: float
    experiment_count: int
    study_count: int
    scores: list[float] = field(default_factory=list)


def aggregate_prompt_scores(
    studies_root: Path,
    score_metric: str = "roc_auc_macro",
) -> dict[str, dict[str, PromptScoreStats]]:
    """Return {task_name: {version: PromptScoreStats}} across all studies.

    For each study, reads `study.json` → `prompt_template_paths` to determine
    which prompt version was used per task. For each completed experiment in
    that study, its score is attributed to those versions.

    Studies WITHOUT `prompt_template_paths` (pre-versioning) attribute all
    scores to "v1" (the migrated original).
    """
    if not studies_root.exists():
        return {}

    # {task_name: {version: [scores]}}
    raw: dict[str, dict[str, list[float]]] = {}
    study_counts: dict[str, dict[str, set[str]]] = {}  # task → version → study_ids

    for study_dir in studies_root.iterdir():
        if not study_dir.is_dir():
            continue
        study_json = study_dir / "study.json"
        if not study_json.exists():
            continue

        try:
            study_data = json.loads(study_json.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        study_id = study_data.get("study_id", study_dir.name)
        prompt_paths: dict[str, str] = study_data.get("prompt_template_paths") or {}

        # Map task_name → version from the paths
        # e.g. {"generate_code": "config/prompts/generate_code/v2.yaml"} → version "v2"
        task_versions: dict[str, str] = {}
        for task_name, path_str in prompt_paths.items():
            p = Path(path_str)
            # Extract version from filename: "v2.yaml" → "v2"
            if p.stem.startswith("v"):
                task_versions[task_name] = p.stem

        # Collect scores from completed experiments
        exp_dir = study_dir / "experiments"
        if not exp_dir.exists():
            continue

        exp_scores: list[float] = []
        for eid in study_data.get("experiment_ids", []):
            exp_json = exp_dir / eid / "experiment.json"
            if not exp_json.exists():
                continue
            try:
                exp = json.loads(exp_json.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if exp.get("status") != "completed":
                continue
            results = exp.get("results")
            if not results:
                continue
            score = (results.get("metrics") or {}).get(score_metric)
            if score is not None:
                exp_scores.append(float(score))

        if not exp_scores:
            continue

        # Attribute scores to task versions.
        # For LLM tasks not in task_versions, default to "v1".
        llm_tasks = {"propose_architecture", "generate_code", "analyze_results", "error_recovery", "report"}
        for task_name in llm_tasks:
            version = task_versions.get(task_name, "v1")
            raw.setdefault(task_name, {}).setdefault(version, []).extend(exp_scores)
            study_counts.setdefault(task_name, {}).setdefault(version, set()).add(study_id)

    # Build stats
    result: dict[str, dict[str, PromptScoreStats]] = {}
    for task_name, versions in raw.items():
        result[task_name] = {}
        for version, scores in versions.items():
            result[task_name][version] = PromptScoreStats(
                task_name=task_name,
                version=version,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                experiment_count=len(scores),
                study_count=len(study_counts.get(task_name, {}).get(version, set())),
                scores=scores,
            )

    return result


__all__ = ["PromptScoreStats", "aggregate_prompt_scores"]
