"""Aggregate experiment scores by (task, prompt_version) for A/B comparison.

Scans ``experiments/studies/*/study.json`` and attributes each experiment's
``primary_score`` to every prompt version that was active during the study.
Output is a nested dict and can be rendered as a bar chart.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PromptScoreStats:
    task: str
    version: str
    scores: list[float] = field(default_factory=list)
    studies: set[str] = field(default_factory=set)

    @property
    def count(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores) if self.scores else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def best(self) -> float:
        return max(self.scores) if self.scores else 0.0


def aggregate_prompt_scores(experiments_dir: Path) -> dict[str, dict[str, PromptScoreStats]]:
    """Return ``{task_name: {version: PromptScoreStats}}``."""
    out: dict[str, dict[str, PromptScoreStats]] = defaultdict(dict)
    if not experiments_dir.exists():
        return out

    for study_dir in sorted(experiments_dir.iterdir()):
        study_json = study_dir / "study.json"
        if not study_json.exists():
            continue
        try:
            data = json.loads(study_json.read_text())
        except json.JSONDecodeError:
            continue

        paths = data.get("prompt_template_paths", {})
        versions = {task: _version_from_path(p) for task, p in paths.items()}

        for exp in data.get("experiments", []):
            score = exp.get("primary_score")
            if not isinstance(score, (int, float)):
                continue
            for task, ver in versions.items():
                if not ver:
                    continue
                stats = out[task].setdefault(ver, PromptScoreStats(task=task, version=ver))
                stats.scores.append(float(score))
                stats.studies.add(data.get("id", ""))
    return dict(out)


def _version_from_path(path: str) -> str | None:
    # Expect ".../propose_architecture/v2.yaml" — grab the stem.
    try:
        return Path(path).stem
    except Exception:  # noqa: BLE001
        return None


__all__ = ["PromptScoreStats", "aggregate_prompt_scores"]
