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

from fastapi import FastAPI, Form, HTTPException, Request
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
    def studies_list(request: Request, launched: int = 0):
        ids = list_studies(studies_root)
        studies = load_many(studies_root, ids)
        # Most recent first so a freshly-launched run is at the top.
        studies.sort(key=lambda s: s.created_at, reverse=True)
        return templates.TemplateResponse(
            request,
            "studies.html",
            {"studies": studies, "launched": launched},
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

    @app.get("/benchmark", response_class=HTMLResponse)
    def benchmark_view(request: Request):
        from lab.core.benchmark import benchmark

        rows = benchmark(studies_root)
        return templates.TemplateResponse(
            request, "benchmark.html", {"rows": rows}
        )

    @app.get("/api/benchmark")
    def api_benchmark():
        from lab.core.benchmark import benchmark

        return JSONResponse(
            [r.model_dump(mode="json") for r in benchmark(studies_root)]
        )

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

    # ----- run form + live -----

    @app.get("/new", response_class=HTMLResponse)
    def new_study_form(request: Request):
        registry = PromptRegistry(prompts_root)
        prompt_versions = {
            t: registry.list_versions(t)
            for t in (
                "propose_architecture",
                "generate_code",
                "recover_from_error",
                "analyze_result",
                "judge_experiment",
                "judge_study",
            )
        }
        recent = load_many(studies_root, list_studies(studies_root))
        recent.sort(key=lambda s: s.created_at, reverse=True)
        return templates.TemplateResponse(
            request,
            "new_study.html",
            {
                "tasks": [settings.default_task],
                "personalities": ["exploratory", "conservative"],
                "prompt_versions": prompt_versions,
                "predecessors": [s.id for s in recent[:20]],
                "defaults": {
                    "task": settings.default_task,
                    "max_experiments": settings.compute_budget.max_experiments,
                    "max_wallclock_min": settings.compute_budget.max_wallclock_minutes,
                },
            },
        )

    @app.post("/run")
    def post_run(payload: dict):
        return _launch_study(
            task=payload.get("task", "track_b"),
            predecessor=payload.get("predecessor_id"),
            use_best_prompts=bool(payload.get("use_best_prompts")),
            agent_memory=bool(payload.get("agent_memory")),
            personality=payload.get("personality"),
            max_experiments=payload.get("max_experiments"),
            max_wallclock_min=payload.get("max_wallclock_min"),
        )

    @app.post("/run-form")
    def post_run_form(
        task: str = Form("track_b"),
        predecessor_id: str | None = Form(default=None),
        use_best_prompts: bool = Form(default=False),
        agent_memory: bool = Form(default=False),
        personality: str = Form("exploratory"),
        max_experiments: int | None = Form(default=None),
        max_wallclock_min: int | None = Form(default=None),
    ):
        _launch_study(
            task=task,
            predecessor=predecessor_id or None,
            use_best_prompts=use_best_prompts,
            agent_memory=agent_memory,
            personality=personality,
            max_experiments=max_experiments,
            max_wallclock_min=max_wallclock_min,
        )
        # cmd_run generates its own study id; redirect to the list and let the
        # user pick the just-started run (it appears once StudyRunner.run()
        # writes its first study.json save).
        return RedirectResponse(url="/studies?launched=1", status_code=303)

    def _launch_study(
        *,
        task: str,
        predecessor: str | None,
        use_best_prompts: bool,
        agent_memory: bool,
        personality: str | None,
        max_experiments: int | None,
        max_wallclock_min: int | None,
    ) -> dict:
        from lab.cli import cmd_run
        from lab.core.models import new_study_id

        class _Args:
            pass

        args = _Args()
        args.task = task
        args.predecessor = predecessor
        args.use_best_prompts = use_best_prompts
        args.agent_memory = agent_memory
        args.personality = personality
        args.max_experiments = max_experiments
        args.max_wallclock_min = max_wallclock_min

        study_id = new_study_id()
        threading.Thread(target=cmd_run, args=(args,), daemon=True).start()
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
