"""Study list + detail pages."""
from __future__ import annotations

import shutil

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from lab.prompts.registry import PromptRegistry
from lab.tasks.registry import list_available_tasks
from lab.ui import launches, loaders


router = APIRouter()


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
    all_studies = loaders.iter_studies(experiments_dir)

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
    return request.app.state.templates.TemplateResponse(
        request, "study.html",
        {
            "study": study,
            "running_launch": running_launch,
            "launch_log_tail": launch_log_tail,
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
        },
    )
