"""FastAPI application factory.

Usage::

    from lab.ui.app import create_app
    app = create_app()

Run with::

    uvicorn lab.ui.app:create_app --factory --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lab.config import Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="lab", version="2.0.0")

    static_dir = Path(__file__).parent / "static"
    templates_dir = Path(__file__).parent / "templates"

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.state.templates = Jinja2Templates(directory=templates_dir)
    app.state.settings = settings

    from lab.ui.routes import (
        studies, experiments, prompts, tasks, reports, api,
        live as live_route, config_editor,
    )
    app.include_router(studies.router)
    app.include_router(experiments.router)
    app.include_router(prompts.router)
    app.include_router(tasks.router)
    app.include_router(reports.router)
    app.include_router(api.router)
    app.include_router(live_route.router)
    app.include_router(config_editor.router)

    return app


app = create_app  # noqa: N816 — export factory for uvicorn --factory


__all__ = ["create_app"]
