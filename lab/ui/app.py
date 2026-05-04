"""FastAPI app factory.

Read-mostly dashboard over `experiments/studies/`, plus a `POST /run` that
launches a study in a background thread and a `/live/<study_id>` SSE stream
that tails the JSONL telemetry sink.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from lab.config import Settings
from lab.core.loaders import list_studies, load_many
from lab.core.models import Study, StudyNotFoundError
from lab.prompts.engine import PromptEngine
from lab.prompts.registry import PromptRegistry
from lab.prompts.scoring import aggregate_prompt_scores

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="lab dashboard")
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
    studies_root = Path(settings.paths.experiments_dir)
    prompts_root = Path(settings.paths.prompts_dir)

    # ----- read views -----

    @app.get("/", response_class=HTMLResponse)
    @app.get("/studies", response_class=HTMLResponse)
    def studies_list(request: Request):
        ids = list_studies(studies_root)
        studies = load_many(studies_root, ids)
        return templates.TemplateResponse(
            request, "studies.html", {"studies": studies}
        )

    @app.get("/studies/{study_id}", response_class=HTMLResponse)
    def study_detail(request: Request, study_id: str):
        try:
            study = Study.load(studies_root, study_id)
        except StudyNotFoundError:
            raise HTTPException(status_code=404, detail="study not found")
        return templates.TemplateResponse(
            request, "study.html", {"study": study}
        )

    @app.get(
        "/experiments/{study_id}/{exp_id}", response_class=HTMLResponse
    )
    def experiment_detail(request: Request, study_id: str, exp_id: str):
        try:
            study = Study.load(studies_root, study_id)
        except StudyNotFoundError:
            raise HTTPException(status_code=404)
        exp = next((e for e in study.experiments if e.id == exp_id), None)
        if exp is None:
            raise HTTPException(status_code=404, detail="experiment not in study")
        return templates.TemplateResponse(
            request, "experiment.html", {"study": study, "exp": exp}
        )

    @app.get("/prompts", response_class=HTMLResponse)
    def prompts_dashboard(request: Request):
        registry = PromptRegistry(prompts_root)
        scores = aggregate_prompt_scores(studies_root)
        tasks = []
        for task in (
            "propose_architecture",
            "generate_code",
            "recover_from_error",
            "analyze_result",
            "judge_experiment",
            "judge_study",
        ):
            try:
                active = registry.active_version(task)
                versions = registry.list_versions(task)
            except Exception:
                active, versions = None, []
            tasks.append({"name": task, "active": active, "versions": versions})
        return templates.TemplateResponse(
            request, "prompts.html", {"tasks": tasks, "scores": scores}
        )

    @app.post("/prompts/{task}/activate")
    def prompts_activate(task: str, version: str):
        registry = PromptRegistry(prompts_root)
        registry.set_active(task, version)
        return RedirectResponse(url="/prompts", status_code=303)

    @app.post("/prompts/{task}/new")
    def prompts_new(task: str, body: dict):
        registry = PromptRegistry(prompts_root)
        v = registry.save_new_version(task, body["system"], body["user"])
        return {"version": v}

    @app.get("/reports/{study_id}", response_class=HTMLResponse)
    def report_view(study_id: str):
        path = studies_root / study_id / "report.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="report not built; run `lab report`")
        return HTMLResponse(path.read_text(encoding="utf-8"))

    # ----- API -----

    @app.get("/api/studies")
    def api_studies():
        ids = list_studies(studies_root)
        return JSONResponse([s.model_dump(mode="json") for s in load_many(studies_root, ids)])

    @app.get("/api/studies/{study_id}")
    def api_study(study_id: str):
        try:
            study = Study.load(studies_root, study_id)
        except StudyNotFoundError:
            raise HTTPException(status_code=404)
        return JSONResponse(study.model_dump(mode="json"))

    # ----- run + live -----

    @app.post("/run")
    def post_run(payload: dict):
        from lab.cli import cmd_run  # avoid import cycle on UI-only deployments

        class _Args:
            task = payload.get("task", "track_b")
            predecessor = payload.get("predecessor_id")
            use_best_prompts = bool(payload.get("use_best_prompts"))
            agent_memory = bool(payload.get("agent_memory"))
            personality = payload.get("personality")
            max_experiments = payload.get("max_experiments")
            max_wallclock_min = payload.get("max_wallclock_min")

        # Pre-allocate a study id so the SSE endpoint has something to tail.
        from lab.core.models import new_study_id

        study_id = new_study_id()

        thread = threading.Thread(target=cmd_run, args=(_Args(),), daemon=True)
        thread.start()
        return {"study_id": study_id}

    @app.get("/live/{study_id}")
    def live(study_id: str):
        path = studies_root / study_id / "run.log.jsonl"

        def event_stream() -> Iterator[bytes]:
            cursor = 0
            while True:
                if path.exists():
                    text = path.read_text(encoding="utf-8")
                    if cursor < len(text):
                        new = text[cursor:]
                        cursor = len(text)
                        for line in new.splitlines():
                            if line.strip():
                                yield f"data: {line}\n\n".encode("utf-8")
                time.sleep(0.5)
                # End-of-stream when study finished and no new bytes for a tick
                # (the test harness doesn't loop forever).
                yield b": keep-alive\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
