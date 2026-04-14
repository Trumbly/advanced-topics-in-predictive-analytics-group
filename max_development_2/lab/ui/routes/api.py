"""JSON endpoints consumed by HTMX fragments and test tooling."""
from __future__ import annotations

from fastapi import APIRouter, Request

from lab.ui import loaders


router = APIRouter(prefix="/api")


@router.get("/studies")
async def list_studies(request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    studies = loaders.iter_studies(experiments_dir)
    return {"studies": [s.__dict__ for s in studies]}


@router.get("/studies/{study_id}")
async def study_json(study_id: str, request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    study = loaders.load_study(experiments_dir, study_id)
    if not study:
        return {"error": "not_found"}, 404
    return study.model_dump(mode="json")
