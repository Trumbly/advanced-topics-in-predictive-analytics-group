"""Prompt A/B UI.

Three pages:
  * Dashboard — mean score per (task, version) across all studies.
  * Version viewer — read-only YAML.
  * Draft editor — create a new immutable ``vN.yaml``.
"""
from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from lab.prompts.registry import PromptRegistry
from lab.prompts.scoring import aggregate_prompt_scores


router = APIRouter(prefix="/prompts")


@router.get("", response_class=HTMLResponse)
async def dashboard(request: Request):
    settings = request.app.state.settings
    registry = PromptRegistry(settings.abspath(settings.paths.prompts_dir))
    experiments_dir = settings.abspath(settings.paths.experiments)

    tasks = registry.list_tasks()
    version_map = {t: registry.list_versions(t) for t in tasks}
    active_map = {t: registry.active_version(t) for t in tasks}
    scores = aggregate_prompt_scores(experiments_dir)

    return request.app.state.templates.TemplateResponse(
        request,
        "prompts.html",
        {
            "tasks": tasks,
            "versions": version_map,
            "active": active_map,
            "scores": scores,
        },
    )


@router.get("/{task}/{version}", response_class=HTMLResponse)
async def view_prompt(task: str, version: str, request: Request):
    settings = request.app.state.settings
    registry = PromptRegistry(settings.abspath(settings.paths.prompts_dir))
    try:
        content = registry.load_prompt(task, version=version)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404)
    import yaml
    raw = yaml.safe_dump(content, sort_keys=False, allow_unicode=True)
    return request.app.state.templates.TemplateResponse(
        request,
        "prompt_version.html",
        {"task": task, "version": version, "raw": raw},
    )


@router.post("/{task}/draft")
async def draft_new_version(
    task: str,
    request: Request,
    system: str = Form(""),
    template: str = Form(""),
    description: str = Form(""),
    make_active: str = Form("false"),
):
    settings = request.app.state.settings
    registry = PromptRegistry(settings.abspath(settings.paths.prompts_dir))
    new_version = registry.save_new_version(
        task,
        content={"system": system, "template": template},
        description=description,
        make_active=make_active.lower() in {"1", "true", "on", "yes"},
    )
    return RedirectResponse(f"/prompts/{task}/{new_version.version}", status_code=303)


@router.post("/{task}/activate/{version}")
async def activate_version(task: str, version: str, request: Request):
    settings = request.app.state.settings
    registry = PromptRegistry(settings.abspath(settings.paths.prompts_dir))
    try:
        registry.set_active(task, version)
    except KeyError:
        raise HTTPException(404)
    return RedirectResponse("/prompts", status_code=303)
