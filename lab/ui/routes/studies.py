"""Study list + detail pages."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from lab.core.models import Study
from lab.prompts.registry import PromptRegistry
from lab.tasks.registry import get_task_adapter, list_available_tasks
from lab.ui import launches, loaders


router = APIRouter()


def _status_value(status_obj) -> str:
    """Normalize pydantic enum/string status values into a plain string."""
    value = getattr(status_obj, "value", status_obj)
    return str(value or "").lower()


def _resolve_stdout_path(experiment, sandbox_root: Path) -> Path:
    if experiment.sandbox_path:
        return Path(experiment.sandbox_path) / "stdout.log"
    return sandbox_root / experiment.id / "stdout.log"


def _metric_from_history(history: list, metric: str) -> float | None:
    best: float | None = None
    for row in history:
        if not isinstance(row, dict):
            continue
        value = row.get(metric)
        if not isinstance(value, (int, float)):
            continue
        fv = float(value)
        if best is None or fv > best:
            best = fv
    return best


def _metric_value_for_experiment(experiment, metric: str) -> float | None:
    if not metric:
        return None
    metrics = experiment.metrics if isinstance(experiment.metrics, dict) else {}
    value = metrics.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    return _metric_from_history(experiment.history or [], metric)


def _recompute_best(study: Study) -> None:
    candidates = [
        e for e in study.experiments
        if e.primary_score is not None
        and (not study.primary_metric or e.primary_metric == study.primary_metric)
    ]
    if not candidates:
        candidates = [e for e in study.experiments if e.primary_score is not None]
    if not candidates:
        study.best_experiment_id = None
        study.best_score = None
        return
    best = max(candidates, key=lambda e: e.primary_score)
    study.best_experiment_id = best.id
    study.best_score = float(best.primary_score)


def _study_metric_options(study: Study) -> list[str]:
    out: list[str] = []
    try:
        adapter = get_task_adapter(task_name=study.task_name)
        out.extend(adapter.available_primary_metrics())
    except Exception:  # noqa: BLE001
        pass
    for e in study.experiments:
        if e.primary_metric and e.primary_metric not in out:
            out.append(e.primary_metric)
        if isinstance(e.metrics, dict):
            for key in e.metrics:
                if isinstance(key, str) and key and key not in out:
                    out.append(key)
    return out


def _filter_studies(
    studies: list,
    *,
    q: str = "",
    task: str = "",
    status: str = "",
    tag: str = "",
    min_score: float | None = None,
    max_score: float | None = None,
    sort: str = "created_desc",
):
    """Apply the query-param filters + sort to a list of StudySummary."""
    q_norm = q.lower().strip()
    task = task.strip()
    status = status.strip()
    tag = tag.strip()

    out = []
    for s in studies:
        if q_norm and q_norm not in s.id.lower() and q_norm not in s.name.lower():
            continue
        if task and s.task_name != task:
            continue
        if status and s.status != status:
            continue
        if tag and tag not in s.tags:
            continue
        score = s.best_score if s.best_score is not None else None
        if min_score is not None:
            if score is None or score < min_score:
                continue
        if max_score is not None:
            if score is None or score > max_score:
                continue
        out.append(s)

    # Sorting. Keys are "<column>_<direction>" so the template can toggle
    # direction by appending/flipping the suffix on header clicks.
    NONE_LOW = float("-inf")
    NONE_HIGH = float("inf")
    sort_key_map = {
        "score": lambda s: s.best_score,
        "roc":   lambda s: s.best_roc_auc,
        "f1":    lambda s: s.best_f1,
        "experiments": lambda s: s.experiments_count,
        "name":  lambda s: s.name.lower(),
        "task":  lambda s: s.task_name,
        "status": lambda s: s.status,
        "created": lambda s: s.created_at or "",
    }
    col, _, direction = sort.rpartition("_")
    if not col or direction not in {"asc", "desc"}:
        col, direction = "created", "desc"
    getter = sort_key_map.get(col, sort_key_map["created"])
    reverse = direction == "desc"

    def _key(s):
        v = getter(s)
        if v is None:
            return NONE_LOW if reverse else NONE_HIGH
        return v

    out.sort(key=_key, reverse=reverse)
    return out


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    q: str = "",
    task: str = "",
    status: str = "",
    tag: str = "",
    min_score: str = "",
    max_score: str = "",
    sort: str = "created_desc",
):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    all_studies = loaders.iter_studies(experiments_dir, reconcile_with=settings.repo_root)

    def _maybe_float(raw: str):
        try:
            return float(raw) if raw != "" else None
        except ValueError:
            return None

    filtered = _filter_studies(
        all_studies,
        q=q, task=task, status=status, tag=tag,
        min_score=_maybe_float(min_score),
        max_score=_maybe_float(max_score),
        sort=sort,
    )

    # Facets for the filter dropdowns: only show values that exist on disk.
    facets = {
        "tasks": sorted({s.task_name for s in all_studies if s.task_name}),
        "statuses": sorted({s.status for s in all_studies if s.status}),
        "tags": sorted({t for s in all_studies for t in s.tags}),
    }

    return request.app.state.templates.TemplateResponse(
        request,
        "index.html",
        {
            "studies": filtered,
            "total": len(all_studies),
            "tasks": list_available_tasks(settings),
            "facets": facets,
            "filters": {
                "q": q, "task": task, "status": status, "tag": tag,
                "min_score": min_score, "max_score": max_score, "sort": sort,
            },
            "active_launches": launches.active_launches(settings.repo_root),
        },
    )


@router.get("/studies/{study_id}", response_class=HTMLResponse)
async def study_detail(study_id: str, request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    sandbox_root = settings.abspath(settings.paths.sandbox)
    study = loaders.load_study(experiments_dir, study_id)
    if not study:
        raise HTTPException(404, f"No study {study_id}")
    running_launch = launches.launch_for_study(study_id, settings.repo_root)
    # If the launch is alive, seed the log panel with the last few hundred
    # lines so the page is useful even before SSE kicks in.
    launch_log_tail = ""
    if running_launch:
        log_path = settings.repo_root / running_launch.log_path
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            launch_log_tail = "\n".join(text.splitlines()[-400:])

    running_experiment = next(
        (e for e in study.experiments if _status_value(e.status) == "running"),
        None,
    )
    running_stdout = ""
    running_stdout_path_display = ""
    running_stdout_exists = False
    if running_experiment is not None:
        stdout_path = _resolve_stdout_path(running_experiment, sandbox_root)
        running_stdout_path_display = str(stdout_path)
        running_stdout_exists = stdout_path.exists()
        if stdout_path.exists():
            text = stdout_path.read_text(errors="replace")
            running_stdout = "\n".join(text.splitlines()[-500:])
    return request.app.state.templates.TemplateResponse(
        request, "study.html",
        {
            "study": study,
            "metric_options": _study_metric_options(study),
            "running_launch": running_launch,
            "launch_log_tail": launch_log_tail,
            "running_experiment": running_experiment,
            "running_stdout": running_stdout,
            "running_stdout_path_display": running_stdout_path_display,
            "running_stdout_exists": running_stdout_exists,
        },
    )


@router.post("/studies/{study_id}/publish")
async def toggle_publish(study_id: str, request: Request, publish: str = Form("false")):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    study = loaders.load_study(experiments_dir, study_id)
    if not study:
        raise HTTPException(404)
    study.publish = publish.lower() in {"1", "true", "on", "yes"}
    loaders.save_study(study, experiments_dir)
    return RedirectResponse(f"/studies/{study_id}", status_code=303)


@router.post("/studies/{study_id}/tags")
async def update_tags(study_id: str, request: Request, tags: str = Form("")):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    study = loaders.load_study(experiments_dir, study_id)
    if not study:
        raise HTTPException(404)
    study.tags = [t.strip() for t in tags.split(",") if t.strip()]
    loaders.save_study(study, experiments_dir)
    return RedirectResponse(f"/studies/{study_id}", status_code=303)


@router.post("/studies/{study_id}/primary-metric")
async def update_study_primary_metric(
    study_id: str,
    request: Request,
    primary_metric: str = Form(""),
):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    study = loaders.load_study(experiments_dir, study_id)
    if not study:
        raise HTTPException(404)
    if _status_value(study.status) == "running":
        raise HTTPException(409, "Cannot change primary metric while study is running")
    metric = primary_metric.strip()
    if not metric:
        raise HTTPException(400, "primary_metric is required")
    allowed = _study_metric_options(study)
    if allowed and metric not in allowed:
        raise HTTPException(400, f"Unknown metric {metric!r}")
    study.primary_metric = metric
    _recompute_best(study)
    loaders.save_study(study, experiments_dir)
    return RedirectResponse(f"/studies/{study_id}", status_code=303)


@router.post("/studies/{study_id}/experiments/{exp_id}/primary-metric")
async def update_experiment_primary_metric(
    study_id: str,
    exp_id: str,
    request: Request,
    primary_metric: str = Form(""),
):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    study = loaders.load_study(experiments_dir, study_id)
    if not study:
        raise HTTPException(404)
    if _status_value(study.status) == "running":
        raise HTTPException(409, "Cannot change experiment metric while study is running")
    metric = primary_metric.strip()
    if not metric:
        raise HTTPException(400, "primary_metric is required")
    allowed = _study_metric_options(study)
    if allowed and metric not in allowed:
        raise HTTPException(400, f"Unknown metric {metric!r}")
    for exp in study.experiments:
        if exp.id != exp_id:
            continue
        exp.primary_metric = metric
        exp.primary_score = _metric_value_for_experiment(exp, metric)
        _recompute_best(study)
        loaders.save_study(study, experiments_dir)
        return RedirectResponse(f"/studies/{study_id}", status_code=303)
    raise HTTPException(404, f"No experiment {exp_id} in study {study_id}")


@router.post("/studies/{study_id}/delete")
async def delete_study(study_id: str, request: Request):
    """Permanently remove a study directory from disk.

    Only the `experiments/studies/<id>/` tree is removed — sandbox dirs and
    the prompt registry are untouched. The caller already confirmed in the
    UI via a JS ``confirm()`` dialog.
    """
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    study_dir = experiments_dir / study_id
    if not (study_dir / "study.json").exists():
        raise HTTPException(404, f"No study {study_id}")
    shutil.rmtree(study_dir, ignore_errors=False)
    return RedirectResponse("/", status_code=303)


@router.get("/new", response_class=HTMLResponse)
async def new_study_form(request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    registry = PromptRegistry(settings.abspath(settings.paths.prompts_dir))

    prompt_tasks = []
    for task in registry.list_tasks():
        versions = [v.version for v in registry.list_versions(task)]
        prompt_tasks.append({
            "name": task,
            "active": registry.active_version(task),
            "versions": versions,
        })

    return request.app.state.templates.TemplateResponse(
        request,
        "new_study.html",
        {
            "tasks": list_available_tasks(settings),
            "default_task": settings.default_task,
            "studies": loaders.iter_studies(experiments_dir),
            "prompt_tasks": prompt_tasks,
            "default_executor": settings.executor.backend,
            "kaggle_configured": bool(settings.executor.kaggle.username),
            "modal_configured": bool(settings.executor.modal.app_name),
        },
    )
