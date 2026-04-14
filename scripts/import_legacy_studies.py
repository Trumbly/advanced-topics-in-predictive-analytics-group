"""Import legacy studies from the `max_development` branch into the v2 layout.

The v1 studies live at ``experiments/studies/<study_id>/`` on the
``max_development`` branch with:

    study.json                                   — summary
    experiments/<exp_id>/experiment.json         — per-experiment payload
    experiments/<exp_id>/tasks/*.json            — per-task payload

The v2 UI reads a single ``study.json`` per study, with the experiments
inlined. This script extracts the v1 studies from git (any ref) and
rewrites them in the v2 schema so they show up in the new dashboard.

Usage (run from repo root, on the max_development_2 branch)::

    python scripts/import_legacy_studies.py --ref max_development
    python scripts/import_legacy_studies.py --ref max_development --dry-run

By default the imported studies are tagged ``legacy`` and ``track_b``
(everything on ``max_development`` was BirdCLEF) and left unpublished.

The conversion is non-destructive: it never touches the source branch.
Existing v2 studies with the same ID are skipped unless ``--force`` is
given.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.config import load_settings  # noqa: E402
from lab.core.models import (  # noqa: E402
    Experiment,
    ExperimentStatus,
    Study,
    StudyStatus,
    Task,
    TaskError,
    TaskStatus,
)


# Old BirdCLEF default. We map the old primary metric to the new schema so
# that the v2 UI ranks the imported studies correctly.
DEFAULT_LEGACY_TASK = "track_b"
DEFAULT_LEGACY_METRIC = "roc_auc_macro"
DEFAULT_LEGACY_TAGS = ["legacy", "max_development"]


FAMILY_RE = re.compile(r"^\s*\[([a-zA-Z0-9_]+)\]")


# ---------------------------------------------------------------------------
# Git extraction
# ---------------------------------------------------------------------------


def _extract_ref_to(tmp: Path, ref: str, subpath: str = "experiments/studies") -> Path:
    """Use ``git archive`` to dump ``<ref>:<subpath>`` into ``tmp``."""
    cmd = ["git", "archive", "--format=tar", ref, subpath]
    result = subprocess.run(cmd, check=True, capture_output=True)
    archive = tmp / "legacy.tar"
    archive.write_bytes(result.stdout)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp)  # noqa: S202 — trusted source (git ref on local repo)
    src = tmp / subpath
    if not src.exists():
        raise FileNotFoundError(f"No {subpath} in ref {ref}")
    return src


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _map_status(raw: str) -> StudyStatus:
    v = (raw or "").lower()
    if v == "completed":
        return StudyStatus.COMPLETED
    if v == "running":
        return StudyStatus.RUNNING
    if v == "aborted":
        return StudyStatus.ABORTED
    if v == "failed":
        return StudyStatus.FAILED
    return StudyStatus.COMPLETED


def _map_exp_status(raw: str) -> ExperimentStatus:
    v = (raw or "").lower()
    if v == "completed":
        return ExperimentStatus.COMPLETED
    if v == "running":
        return ExperimentStatus.RUNNING
    if v == "failed":
        return ExperimentStatus.FAILED
    if v == "aborted":
        return ExperimentStatus.ABORTED
    return ExperimentStatus.PENDING


def _family_from_arch(arch: str | None) -> str | None:
    if not arch:
        return None
    m = FAMILY_RE.match(arch)
    return m.group(1) if m else None


def _history_from_curves(curves: dict[str, list[float]], primary_metric: str) -> list[dict[str, Any]]:
    """Transpose ``{metric: [v1, v2, ...]}`` into ``[{metric: v1, ...}, ...]``."""
    if not curves:
        return []
    lengths = {len(v) for v in curves.values() if isinstance(v, list)}
    n = max(lengths) if lengths else 0
    history: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {"epoch": i + 1}
        for k, v in curves.items():
            if isinstance(v, list) and i < len(v):
                row[k] = v[i]
        history.append(row)
    return history


def _first_task_error(tasks_dir: Path) -> TaskError | None:
    """Return the first error from a task JSON in chronological order."""
    if not tasks_dir.exists():
        return None
    for p in sorted(tasks_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        err = data.get("error")
        if err:
            return TaskError(
                error_type=err.get("error_type", "UnknownError"),
                message=err.get("message", ""),
                traceback=err.get("traceback"),
            )
    return None


def _load_exp_tasks(exp_dir: Path) -> list[Task]:
    tasks_dir = exp_dir / "tasks"
    out: list[Task] = []
    if not tasks_dir.exists():
        return out
    for p in sorted(tasks_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        err_raw = data.get("error") or None
        err = None
        if err_raw:
            err = TaskError(
                error_type=err_raw.get("error_type", "UnknownError"),
                message=err_raw.get("message", ""),
                traceback=err_raw.get("traceback"),
            )
        status_raw = (data.get("status") or "").lower()
        if status_raw == "completed":
            status = TaskStatus.COMPLETED
        elif status_raw == "failed":
            status = TaskStatus.FAILED
        elif status_raw == "running":
            status = TaskStatus.RUNNING
        elif status_raw == "skipped":
            status = TaskStatus.SKIPPED
        else:
            status = TaskStatus.PENDING
        out.append(Task(
            name=data.get("task_name") or data.get("task_type") or "unknown",
            status=status,
            output=data.get("output") or {},
            code_used=data.get("code_used"),
            error=err,
        ))
    return out


def convert_experiment(exp_dir: Path, study_id: str, primary_metric: str) -> Experiment:
    data = json.loads((exp_dir / "experiment.json").read_text())
    config = data.get("config", {}) or {}
    arch = config.get("architecture")
    results = data.get("results") or {}
    metrics_raw = results.get("metrics") or {}
    metrics = {k: float(v) for k, v in metrics_raw.items() if isinstance(v, (int, float))}
    history = _history_from_curves(results.get("training_curves") or {}, primary_metric)
    primary_score = metrics.get(primary_metric)

    status = _map_exp_status(data.get("status", ""))
    err = _first_task_error(exp_dir / "tasks") if status == ExperimentStatus.FAILED else None

    # Find the code the experiment ended up running (last generate_code task).
    code = None
    for p in sorted((exp_dir / "tasks").glob("*generate_code*.json")) if (exp_dir / "tasks").exists() else []:
        try:
            d = json.loads(p.read_text())
            if d.get("code_used"):
                code = d["code_used"]
        except (OSError, json.JSONDecodeError):
            continue

    return Experiment(
        id=data.get("experiment_id", exp_dir.name),
        study_id=study_id,
        index=int(re.sub(r"\D", "", exp_dir.name) or 0),
        status=status,
        architecture_name=arch,
        architecture_family=_family_from_arch(arch),
        architecture_proposal=json.dumps(config, indent=2) if config else None,
        code=code,
        primary_metric=primary_metric,
        primary_score=primary_score,
        metrics=metrics,
        history=history,
        error=err,
        tasks=_load_exp_tasks(exp_dir),
        started_at=_parse_dt(data.get("started_at") or data.get("created_at")),
        completed_at=_parse_dt(data.get("completed_at")),
        duration_seconds=(results.get("duration_seconds") if isinstance(results.get("duration_seconds"), (int, float)) else None),
        sandbox_path=None,
    )


def _parse_dt(raw: str | None):
    if not raw:
        return None
    from datetime import datetime
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def convert_study(study_dir: Path, *, task_name: str, primary_metric: str, tags: list[str]) -> Study:
    data = json.loads((study_dir / "study.json").read_text())
    study_id = data.get("study_id", study_dir.name)

    experiments: list[Experiment] = []
    exp_root = study_dir / "experiments"
    if exp_root.exists():
        for exp_dir in sorted(exp_root.iterdir()):
            if not (exp_dir / "experiment.json").exists():
                continue
            try:
                experiments.append(convert_experiment(exp_dir, study_id, primary_metric))
            except Exception as exc:  # noqa: BLE001
                print(f"    ! skipping {exp_dir.name}: {exc}", file=sys.stderr)

    study = Study(
        id=study_id,
        name=data.get("name") or study_id,
        task_name=task_name,
        status=_map_status(data.get("status", "")),
        primary_metric=primary_metric,
        best_experiment_id=data.get("best_experiment_id"),
        best_score=data.get("best_score"),
        experiments=experiments,
        prompt_template_paths=data.get("prompt_template_paths") or {},
        predecessor_id=None,
        publish=False,
        tags=list(tags),
        notes=(data.get("hypothesis") or ""),
    )
    if isinstance(_parse_dt(data.get("created_at")), object):
        dt = _parse_dt(data.get("created_at"))
        if dt:
            study.created_at = dt
            study.started_at = dt
    study.completed_at = _parse_dt(data.get("updated_at") or data.get("completed_at"))
    return study


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", default="max_development",
                   help="git ref to import from (default: max_development)")
    p.add_argument("--task", default=DEFAULT_LEGACY_TASK,
                   help="task name to attach to imported studies (default: track_b)")
    p.add_argument("--primary-metric", default=DEFAULT_LEGACY_METRIC,
                   help="primary metric to use (default: roc_auc_macro)")
    p.add_argument("--tags", default=",".join(DEFAULT_LEGACY_TAGS),
                   help="comma-separated tags to attach (default: legacy,max_development)")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing v2 studies with the same ID")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be imported without writing anything")
    args = p.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    settings = load_settings()
    out_dir = settings.abspath(settings.paths.experiments)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print(f"extracting {args.ref}:experiments/studies ...", file=sys.stderr)
        src = _extract_ref_to(tmp, args.ref)

        imported = 0
        skipped = 0
        failed = 0
        for study_dir in sorted(src.iterdir()):
            if not (study_dir / "study.json").exists():
                continue
            study_id = study_dir.name
            target = out_dir / study_id
            if target.exists() and not args.force:
                print(f"  = {study_id}  (already imported; skip)")
                skipped += 1
                continue
            try:
                study = convert_study(
                    study_dir,
                    task_name=args.task,
                    primary_metric=args.primary_metric,
                    tags=tags,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {study_id}  ({exc})", file=sys.stderr)
                failed += 1
                continue

            n_exp = len(study.experiments)
            best = f"{study.best_score:.4f}" if study.best_score is not None else "—"
            marker = "[dry]" if args.dry_run else "[imp]"
            print(f"  {marker} {study_id}  n={n_exp}  best={best}")
            if not args.dry_run:
                study.save(out_dir)
            imported += 1

    print(f"\nDone. imported={imported}  skipped={skipped}  failed={failed}", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
