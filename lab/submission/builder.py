"""Submission dispatch.

Called by the CLI's ``lab submit`` subcommand. Looks up the study's best
experiment and calls the task adapter's ``build_submission`` hook.
"""
from __future__ import annotations

from pathlib import Path

from lab.config import Settings, load_settings
from lab.core.models import Study
from lab.tasks.registry import get_task_adapter


def build_submission_for_study(
    study_id: str,
    *,
    settings: Settings | None = None,
    out_dir: Path | None = None,
) -> Path:
    settings = settings or load_settings()
    experiments_dir = settings.abspath(settings.paths.experiments)
    study_dir = experiments_dir / study_id
    if not (study_dir / "study.json").exists():
        raise FileNotFoundError(f"No study found at {study_dir}")
    study = Study.load(study_dir)

    if not study.best_experiment_id:
        raise ValueError(f"Study {study_id} has no best experiment yet")
    best = next(e for e in study.experiments if e.id == study.best_experiment_id)
    if not best.code:
        raise ValueError(f"Best experiment {best.id} has no stored code")

    adapter = get_task_adapter(task_name=study.task_name)
    out = out_dir or (study_dir / "submission")
    return adapter.build_submission(best.code, best.id, out)


__all__ = ["build_submission_for_study"]
