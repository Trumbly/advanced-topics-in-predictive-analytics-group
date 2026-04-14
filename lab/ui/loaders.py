"""Read-only helpers: walk ``experiments/studies/*`` and load studies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.core.models import Study


# Metric-name aliases we aggregate into the overview columns. The same
# semantic family may have been recorded under different names across
# v1 and v2 agents (f1_macro vs f1_binary for instance). We surface the
# best value any experiment produced under any recognised alias.
_ROC_KEYS = ("roc_auc_macro", "roc_auc", "roc_auc_binary")
_F1_KEYS = ("f1_macro", "f1", "f1_binary")


def _best_across_keys(metrics_list: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    best: float | None = None
    for metrics in metrics_list:
        if not isinstance(metrics, dict):
            continue
        for k in keys:
            v = metrics.get(k)
            if isinstance(v, (int, float)):
                if best is None or v > best:
                    best = float(v)
    return best


@dataclass
class StudySummary:
    id: str
    name: str
    task_name: str
    status: str
    created_at: str | None
    primary_metric: str
    best_score: float | None
    best_roc_auc: float | None
    best_f1: float | None
    experiments_count: int
    publish: bool
    tags: list[str]
    executor_backend: str = "local"
    executor_infrastructure: dict[str, Any] | None = None
    predecessor_id: str | None = None


def _summarise(data: dict[str, Any], fallback_id: str) -> StudySummary:
    """Derive a StudySummary from a loaded study.json dict."""
    experiments = data.get("experiments", []) or []
    metrics_list = [e.get("metrics", {}) for e in experiments]
    return StudySummary(
        id=data.get("id", fallback_id),
        name=data.get("name", fallback_id),
        task_name=data.get("task_name", ""),
        status=data.get("status", "unknown"),
        created_at=data.get("created_at"),
        primary_metric=data.get("primary_metric", ""),
        best_score=data.get("best_score"),
        best_roc_auc=_best_across_keys(metrics_list, _ROC_KEYS),
        best_f1=_best_across_keys(metrics_list, _F1_KEYS),
        experiments_count=len(experiments),
        publish=bool(data.get("publish", False)),
        tags=list(data.get("tags", [])),
        executor_backend=data.get("executor_backend", "local"),
        executor_infrastructure=data.get("executor_infrastructure") or {},
        predecessor_id=data.get("predecessor_id"),
    )


def iter_studies(experiments_dir: Path, *, reconcile_with: Path | None = None) -> list[StudySummary]:
    """Walk the studies directory and return a summary per study.

    ``reconcile_with`` is the repo root: when set, studies recorded as
    ``running`` but with no matching live launch get relabeled to
    ``crashed`` in the returned summaries (not on disk — this is
    purely a display fix). That happens when the agent subprocess
    SIGKILL'd / the machine rebooted mid-study.
    """
    if not experiments_dir.exists():
        return []

    live_study_ids: set[str] = set()
    if reconcile_with is not None:
        from lab.ui import launches  # local import avoids cycle
        for l in launches.active_launches(reconcile_with):
            if l.study_id:
                live_study_ids.add(l.study_id)

    out: list[StudySummary] = []
    for d in sorted(experiments_dir.iterdir(), reverse=True):
        sj = d / "study.json"
        if not sj.exists():
            continue
        try:
            data = json.loads(sj.read_text())
        except json.JSONDecodeError:
            continue
        summary = _summarise(data, d.name)
        if (reconcile_with is not None
                and summary.status == "running"
                and summary.id not in live_study_ids):
            summary.status = "crashed"
        out.append(summary)
    return out


def load_study(experiments_dir: Path, study_id: str) -> Study | None:
    study_dir = experiments_dir / study_id
    if not (study_dir / "study.json").exists():
        return None
    return Study.load(study_dir)


def save_study(study: Study, experiments_dir: Path) -> None:
    study.save(experiments_dir)


def load_experiment_raw(experiments_dir: Path, study_id: str, exp_id: str) -> dict[str, Any] | None:
    study = load_study(experiments_dir, study_id)
    if not study:
        return None
    for e in study.experiments:
        if e.id == exp_id:
            return e.model_dump(mode="json")
    return None


def read_live_stdout(sandbox_root: Path, exp_id: str, *, tail: int = 400) -> str:
    path = sandbox_root / exp_id / "stdout.log"
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-tail:])


__all__ = [
    "StudySummary",
    "iter_studies",
    "load_study",
    "save_study",
    "load_experiment_raw",
    "read_live_stdout",
]
