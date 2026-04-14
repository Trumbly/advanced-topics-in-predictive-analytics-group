"""Study list + detail pages."""
from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from lab.tasks.registry import list_available_tasks
from lab.ui import loaders


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    studies = loaders.iter_studies(experiments_dir)
    return request.app.state.templates.TemplateResponse(
        "index.html",
        {"request": request, "studies": studies, "tasks": list_available_tasks(settings)},
    )


@router.get("/studies/{study_id}", response_class=HTMLResponse)
async def study_detail(study_id: str, request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    study = loaders.load_study(experiments_dir, study_id)
    if not study:
        raise HTTPException(404, f"No study {study_id}")
    return request.app.state.templates.TemplateResponse(
        "study.html", {"request": request, "study": study},
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
    return request.app.state.templates.TemplateResponse(
        "new_study.html",
        {
            "request": request,
            "tasks": list_available_tasks(settings),
            "default_task": settings.default_task,
            "studies": loaders.iter_studies(experiments_dir),
        },
    )
