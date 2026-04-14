"""Experiment detail pages."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from lab.ui import loaders


router = APIRouter()


@router.get("/studies/{study_id}/experiments/{exp_id}", response_class=HTMLResponse)
async def experiment_detail(study_id: str, exp_id: str, request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    sandbox_root = settings.abspath(settings.paths.sandbox)
    study = loaders.load_study(experiments_dir, study_id)
    if not study:
        raise HTTPException(404)
    exp = next((e for e in study.experiments if e.id == exp_id), None)
    if exp is None:
        raise HTTPException(404)
    live_stdout = loaders.read_live_stdout(sandbox_root, exp_id, tail=500)
    return request.app.state.templates.TemplateResponse(
        "experiment.html",
        {"request": request, "study": study, "experiment": exp, "live_stdout": live_stdout},
    )
