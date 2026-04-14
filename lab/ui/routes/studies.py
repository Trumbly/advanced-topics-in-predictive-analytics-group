"""Study list + detail pages."""
from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from lab.prompts.registry import PromptRegistry
from lab.tasks.registry import list_available_tasks
from lab.ui import loaders


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    studies = loaders.iter_studies(experiments_dir)
    return request.app.state.templates.TemplateResponse(
        request,
        "index.html",
        {"studies": studies, "tasks": list_available_tasks(settings)},
    )


@router.get("/studies/{study_id}", response_class=HTMLResponse)
async def study_detail(study_id: str, request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    study = loaders.load_study(experiments_dir, study_id)
    if not study:
        raise HTTPException(404, f"No study {study_id}")
    return request.app.state.templates.TemplateResponse(
        request, "study.html", {"study": study},
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
        },
    )
