"""FastAPI app factory.

Read-mostly dashboard over `experiments/studies/`, plus a `POST /run` that
launches a study in a background thread and a `/live/<study_id>` SSE stream
that tails the JSONL telemetry sink.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
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
from lab.ui.display import build_study_ordinals, exp_label, study_label, study_suffix

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class _BuildState:
    started_at: float
    error: str | None = None
    thread: threading.Thread | None = None


_report_builds: dict[str, _BuildState] = {}
_report_builds_lock = threading.Lock()


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="lab dashboard")
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
    templates.env.globals["study_label"] = study_label
    templates.env.globals["exp_label"] = exp_label
    templates.env.globals["study_suffix"] = study_suffix
    studies_root = Path(settings.paths.experiments_dir)
    prompts_root = Path(settings.paths.prompts_dir)

    # ----- read views -----

    @app.get("/", response_class=HTMLResponse)
    @app.get("/studies", response_class=HTMLResponse)
    def studies_list(request: Request, launched: int = 0):
        ids = list_studies(studies_root)
        studies = load_many(studies_root, ids)
        ordinals = build_study_ordinals(studies)
        # Most recent first so a freshly-launched run is at the top.
        studies.sort(key=lambda s: s.created_at, reverse=True)
        return templates.TemplateResponse(
            request,
            "studies.html",
            {"studies": studies, "launched": launched, "ordinals": ordinals},
        )

    @app.get("/studies/{study_id}", response_class=HTMLResponse)
    def study_detail(request: Request, study_id: str):
        from lab.core.codegen_rates import study_rates
        from lab.core.llm_metrics import (
            experiment_summary,
            study_summary,
        )
        from lab.ui.learning_curves import (
            collect_series,
            render_study_loss_svg,
            render_svg,
            study_loss_series,
            study_to_chart_data,
        )
        from lab.ui.pipeline import (
            PHASES,
            current_step,
            derive_study_pipelines,
        )

        try:
            study = Study.load(studies_root, study_id)
        except StudyNotFoundError:
            raise HTTPException(status_code=404, detail="study not found")
        all_study_ids = list_studies(studies_root)
        ordinals = build_study_ordinals(load_many(studies_root, all_study_ids))
        rates = {r.experiment_id: r for r in study_rates(study)}
        llm_summary = study_summary(study).to_dict()
        exp_llm = {e.id: experiment_summary(e).to_dict() for e in study.experiments}
        # One mini-sparkline per experiment so the user can scan
        # convergence shape across the whole study at a glance.
        exp_curves = {
            e.id: render_svg(
                collect_series(e.history, e.primary_metric),
                width=360, height=120, pad_left=36, pad_top=10, pad_bottom=22,
            )
            for e in study.experiments
        }
        return templates.TemplateResponse(
            request,
            "study.html",
            {
                "study": study,
                "phases": PHASES,
                "pipelines": derive_study_pipelines(study),
                "current": current_step(study),
                "rates": rates,
                "llm_summary": llm_summary,
                "exp_llm": exp_llm,
                "exp_curves": exp_curves,
                "study_chart_data": study_to_chart_data(study.experiments),
                "ordinals": ordinals,
            },
        )

    @app.get(
        "/experiments/{study_id}/{exp_id}", response_class=HTMLResponse
    )
    def experiment_detail(request: Request, study_id: str, exp_id: str):
        from lab.core.llm_metrics import experiment_summary
        from lab.ui.learning_curves import (
            collect_series,
            history_to_chart_data,
            last_epoch_summary,
            render_svg,
        )

        try:
            study = Study.load(studies_root, study_id)
        except StudyNotFoundError:
            raise HTTPException(status_code=404)
        exp = next((e for e in study.experiments if e.id == exp_id), None)
        if exp is None:
            raise HTTPException(status_code=404, detail="experiment not in study")
        all_study_ids = list_studies(studies_root)
        ordinals = build_study_ordinals(load_many(studies_root, all_study_ids))
        series = collect_series(exp.history, exp.primary_metric)
        return templates.TemplateResponse(
            request,
            "experiment.html",
            {
                "study": study,
                "exp": exp,
                "llm_summary": experiment_summary(exp).to_dict(),
                "exp_chart_data": history_to_chart_data(exp.history, exp.primary_metric),
                "last_epoch": last_epoch_summary(exp.history, exp.primary_metric),
                "ordinals": ordinals,
            },
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

    @app.get("/prompts/{task}", response_class=HTMLResponse)
    def prompts_task_root(task: str):
        registry = PromptRegistry(prompts_root)
        try:
            active = registry.active_version(task)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown task: {task}")
        return RedirectResponse(url=f"/prompts/{task}/{active}", status_code=303)

    @app.get("/prompts/{task}/{version}", response_class=HTMLResponse)
    def prompts_edit_view(
        request: Request,
        task: str,
        version: str,
        saved: int = 0,
        error: str | None = None,
    ):
        registry = PromptRegistry(prompts_root)
        try:
            tmpl = registry.load(task, version)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            active = registry.active_version(task)
        except KeyError:
            active = None
        versions = registry.list_versions(task)
        return templates.TemplateResponse(
            request,
            "prompt_edit.html",
            {
                "task": task,
                "version": version,
                "active": active,
                "versions": versions,
                "system": tmpl.system,
                "user_text": tmpl.user,
                "saved": saved,
                "error": error,
                "is_new": False,
            },
        )

    @app.get("/prompts/{task}/new/blank", response_class=HTMLResponse)
    def prompts_edit_new(request: Request, task: str):
        """Blank editor seeded from the active version (so a fresh
        version still has the structural slots the engine expects)."""
        registry = PromptRegistry(prompts_root)
        try:
            active = registry.active_version(task)
            tmpl = registry.load(task, active)
            seed_sys, seed_user = tmpl.system, tmpl.user
        except (KeyError, FileNotFoundError):
            active, seed_sys, seed_user = None, "", ""
        return templates.TemplateResponse(
            request,
            "prompt_edit.html",
            {
                "task": task,
                "version": "(new)",
                "active": active,
                "versions": registry.list_versions(task),
                "system": seed_sys,
                "user_text": seed_user,
                "saved": 0,
                "error": None,
                "is_new": True,
            },
        )

    @app.post("/prompts/{task}/edit")
    def prompts_edit_save(
        task: str,
        system: str = Form(...),
        user: str = Form(...),
        activate: str = Form(default=""),
    ):
        registry = PromptRegistry(prompts_root)
        try:
            new_version = registry.save_new_version(task, system, user)
        except (FileExistsError, OSError) as exc:
            return RedirectResponse(
                url=f"/prompts/{task}/new/blank?error={exc}", status_code=303
            )
        if activate:
            registry.set_active(task, new_version)
        return RedirectResponse(
            url=f"/prompts/{task}/{new_version}?saved=1", status_code=303
        )

    @app.get("/reports/{study_id}", response_class=HTMLResponse)
    def report_view(request: Request, study_id: str):
        from lab.reporting.generator import generate_report

        report_path = studies_root / study_id / "report.html"
        if report_path.exists():
            return HTMLResponse(report_path.read_text(encoding="utf-8"))

        # Confirm the study itself exists — 404 if the user typed a bad id.
        try:
            study = Study.load(studies_root, study_id)
        except StudyNotFoundError:
            raise HTTPException(status_code=404, detail="study not found")

        with _report_builds_lock:
            state = _report_builds.get(study_id)
            if state is None or (state.thread and not state.thread.is_alive() and state.error is None):
                # Either no build yet, or a previous build finished without
                # writing the file (race: report wasn't where we expected,
                # so just retry).
                state = _BuildState(started_at=time.time())
                _report_builds[study_id] = state

                def _run_build(sid: str = study_id, st: _BuildState = state) -> None:
                    try:
                        generate_report(Study.load(studies_root, sid), settings)
                        with _report_builds_lock:
                            # Successful build: drop the entry so future
                            # invalidations (e.g. user re-runs the study and
                            # wants a fresh report) re-trigger the build path.
                            _report_builds.pop(sid, None)
                    except Exception as exc:  # pragma: no cover - defensive
                        with _report_builds_lock:
                            st.error = f"{type(exc).__name__}: {exc}"

                t = threading.Thread(target=_run_build, daemon=True)
                state.thread = t
                t.start()

            elapsed_s = int(time.time() - state.started_at)
            last_error = state.error

        return templates.TemplateResponse(
            request,
            "report_building.html",
            {
                "study": study,
                "elapsed_s": elapsed_s,
                "last_error": last_error,
            },
            status_code=202 if last_error is None else 500,
        )

    # ---- submission ----

    @app.post("/studies/{study_id}/submission", response_class=HTMLResponse)
    def build_submission(request: Request, study_id: str):
        """Build both submission.ipynb and submission.csv for a study.

        - ipynb: rendered from the notebook template; runs on Kaggle to
          produce the same wide-format CSV directly there.
        - csv: produced locally by running the best experiment's model
          over data/raw/test_soundscapes/*.ogg with matching mel params.
        """
        from lab.submission.builder import (
            SubmissionValidationError,
            build_local_csv_for_study,
            build_submission_for_study,
            copy_weights_for_study,
        )

        try:
            study = Study.load(studies_root, study_id)
        except StudyNotFoundError:
            raise HTTPException(status_code=404, detail="study not found")

        all_study_ids = list_studies(studies_root)
        ordinals = build_study_ordinals(load_many(studies_root, all_study_ids))
        errors: list[str] = []
        notebook_path: Path | None = None
        csv_path: Path | None = None
        weights_path: Path | None = None
        try:
            notebook_path = build_submission_for_study(study, settings)
        except SubmissionValidationError as exc:
            errors.append(f"notebook: {exc}")
        try:
            weights_path = copy_weights_for_study(study, settings)
        except SubmissionValidationError as exc:
            errors.append(f"weights: {exc}")
        try:
            csv_path = build_local_csv_for_study(study, settings)
        except SubmissionValidationError as exc:
            errors.append(f"csv: {exc}")

        return templates.TemplateResponse(
            request,
            "submission.html",
            {
                "study": study,
                "notebook_path": notebook_path,
                "csv_path": csv_path,
                "weights_path": weights_path,
                "errors": errors,
                "ordinals": ordinals,
            },
        )

    @app.get("/studies/{study_id}/submission.ipynb")
    def download_submission_notebook(study_id: str):
        path = studies_root / study_id / "submission.ipynb"
        if not path.exists():
            raise HTTPException(status_code=404, detail="notebook not built yet")
        from fastapi.responses import FileResponse

        return FileResponse(
            path, media_type="application/x-ipynb+json", filename="submission.ipynb"
        )

    @app.get("/studies/{study_id}/submission.csv")
    def download_submission_csv(study_id: str):
        path = studies_root / study_id / "submission.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="csv not built yet")
        from fastapi.responses import FileResponse

        return FileResponse(path, media_type="text/csv", filename="submission.csv")

    @app.post(
        "/studies/{study_id}/experiments/{exp_id}/submission",
        response_class=HTMLResponse,
    )
    def build_experiment_submission(request: Request, study_id: str, exp_id: str):
        """Build the Kaggle artifacts (ipynb + csv + weights) for ONE
        successful experiment instead of the study's best one. Outputs
        land under ``experiments/studies/<study_id>/experiments/<exp_id>/``
        so they don't clobber the study-level best submission."""
        from lab.submission.builder import (
            SubmissionValidationError,
            build_local_csv_for_study,
            build_submission_for_study,
            copy_weights_for_study,
        )

        try:
            study = Study.load(studies_root, study_id)
        except StudyNotFoundError:
            raise HTTPException(status_code=404, detail="study not found")

        all_study_ids = list_studies(studies_root)
        ordinals = build_study_ordinals(load_many(studies_root, all_study_ids))
        errors: list[str] = []
        notebook_path: Path | None = None
        csv_path: Path | None = None
        weights_path: Path | None = None
        try:
            notebook_path = build_submission_for_study(
                study, settings, experiment_id=exp_id
            )
        except SubmissionValidationError as exc:
            errors.append(f"notebook: {exc}")
        try:
            weights_path = copy_weights_for_study(
                study, settings, experiment_id=exp_id
            )
        except SubmissionValidationError as exc:
            errors.append(f"weights: {exc}")
        try:
            csv_path = build_local_csv_for_study(
                study, settings, experiment_id=exp_id
            )
        except SubmissionValidationError as exc:
            errors.append(f"csv: {exc}")

        return templates.TemplateResponse(
            request,
            "submission.html",
            {
                "study": study,
                "experiment_id": exp_id,
                "notebook_path": notebook_path,
                "csv_path": csv_path,
                "weights_path": weights_path,
                "errors": errors,
                "ordinals": ordinals,
            },
        )

    @app.get("/studies/{study_id}/experiments/{exp_id}/submission.ipynb")
    def download_experiment_submission_notebook(study_id: str, exp_id: str):
        path = studies_root / study_id / "experiments" / exp_id / "submission.ipynb"
        if not path.exists():
            raise HTTPException(status_code=404, detail="notebook not built yet")
        from fastapi.responses import FileResponse

        return FileResponse(
            path,
            media_type="application/x-ipynb+json",
            filename=f"submission_{exp_id}.ipynb",
        )

    @app.get("/studies/{study_id}/experiments/{exp_id}/submission.csv")
    def download_experiment_submission_csv(study_id: str, exp_id: str):
        path = studies_root / study_id / "experiments" / exp_id / "submission.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="csv not built yet")
        from fastapi.responses import FileResponse

        return FileResponse(
            path, media_type="text/csv", filename=f"submission_{exp_id}.csv"
        )

    @app.get("/studies/{study_id}/experiments/{exp_id}/weights.pt")
    def download_experiment_weights(study_id: str, exp_id: str):
        path = studies_root / study_id / "experiments" / exp_id / "weights.pt"
        if not path.exists():
            raise HTTPException(status_code=404, detail="weights not built yet")
        from fastapi.responses import FileResponse

        return FileResponse(
            path, media_type="application/octet-stream", filename=f"weights_{exp_id}.pt"
        )

    @app.post("/studies/{study_id}/stop")
    def stop_study(study_id: str):
        """Flip the abort flag on the in-flight runner.

        The runner's loop checks the flag between experiments, so the
        currently-running experiment finishes before the study exits.
        Real-time interrupt mid-experiment is not supported (subprocess
        watchdog catches frozen runs separately).

        When no runner is registered (UI restarted, CLI-launched run, or
        the runner already finished but study.json wasn't refreshed yet),
        we treat that as "already stopped" -- mark study.status=ABORTED
        on disk so the UI's running/finished split picks the resume
        button next time. Avoids spurious 404 errors from a stale UI.
        """
        from datetime import datetime, timezone

        from lab.core.lifecycle import get_running_runner

        runner = get_running_runner(study_id)
        if runner is None:
            try:
                study = Study.load(studies_root, study_id)
            except StudyNotFoundError:
                raise HTTPException(status_code=404, detail="study not found")
            if study.status == "RUNNING":
                study.status = "ABORTED"
                study.finished_at = datetime.now(timezone.utc)
                study.save(studies_root)
            return RedirectResponse(
                url=f"/studies/{study_id}", status_code=303
            )
        runner.abort()
        return RedirectResponse(url=f"/studies/{study_id}", status_code=303)

    @app.post("/studies/{study_id}/resume")
    def resume_study(study_id: str):
        """Launch a NEW study seeded from ``study_id`` as predecessor.

        Memory carries over via Memory.seed_from_predecessor (top-K wins
        + recent failures), and the LLM can pick `continue_from` on any
        of the source's experiments via the existing proposal field.
        That's resume in the agent-loop sense — same task, full memory,
        free to pick up where the stopped run left off.
        """
        try:
            source = Study.load(studies_root, study_id)
        except StudyNotFoundError:
            raise HTTPException(status_code=404, detail="study not found")
        from lab.cli import cmd_run

        class _Args:
            pass

        args = _Args()
        args.task = source.task_name
        args.predecessor = source.id
        args.use_best_prompts = False
        args.agent_memory = True
        args.personality = source.personality
        args.max_experiments = None
        args.max_wallclock_min = None
        args.llm_model = None
        args.data_subset = source.data_subset_percent
        threading.Thread(target=cmd_run, args=(args,), daemon=True).start()
        return RedirectResponse(url="/studies?launched=1", status_code=303)

    @app.get("/studies/{study_id}/weights.pt")
    def download_weights(study_id: str):
        path = studies_root / study_id / "weights.pt"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "weights not exported yet — click "
                    "'Build Kaggle submission' on the study page first"
                ),
            )
        from fastapi.responses import FileResponse

        return FileResponse(
            path, media_type="application/octet-stream", filename="weights.pt"
        )

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

    @app.get("/settings", response_class=HTMLResponse)
    def settings_view(request: Request, saved: int = 0, error: str | None = None):
        from lab.ui.settings_io import (
            read_global_yaml,
            repo_root_for,
        )

        root = repo_root_for(settings)
        current = read_global_yaml(root)
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "config": current,
                "saved": saved,
                "error": error,
                "config_path": "config/config.yaml",
            },
        )

    @app.post("/settings")
    async def settings_save(request: Request):
        from pydantic import ValidationError
        from lab.ui.settings_io import (
            parse_form_into_yaml,
            read_global_yaml,
            repo_root_for,
            validate_yaml,
            write_global_yaml,
        )

        form_data = await request.form()
        form = {k: str(v) for k, v in form_data.items()}
        root = repo_root_for(settings)
        current = read_global_yaml(root)
        proposed = parse_form_into_yaml(current, form)

        try:
            validate_yaml(root, proposed)
        except ValidationError as exc:
            msg = str(exc).replace("\n", " | ")[:600]
            return RedirectResponse(
                url=f"/settings?error={msg}", status_code=303
            )

        write_global_yaml(root, proposed)
        return RedirectResponse(url="/settings?saved=1", status_code=303)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_view(request: Request):
        from lab.core.dashboard import compute_kpis

        kpis = compute_kpis(studies_root)
        all_study_ids = list_studies(studies_root)
        ordinals = build_study_ordinals(load_many(studies_root, all_study_ids))
        best_study_label: str | None = None
        best_exp_label_str: str | None = None
        if kpis.best_model:
            sid = kpis.best_model.study_id
            ord_n = ordinals.get(sid, len(ordinals) + 1)
            best_study_label = study_label(sid, ord_n)
            eid = kpis.best_model.experiment_id
            # Try to look up the real experiment to get the correct index.
            # Fall back to a heuristic (last numeric segment) if unavailable.
            try:
                best_study_obj = Study.load(studies_root, sid)
                best_exp_obj = next(
                    (e for e in best_study_obj.experiments if e.id == eid), None
                )
                if best_exp_obj is not None:
                    best_exp_label_str = exp_label(best_exp_obj)
            except Exception:
                best_exp_obj = None
            if best_exp_label_str is None:
                # Heuristic: parse trailing digits from experiment id
                try:
                    idx = int(eid.rsplit("_", 1)[-1])
                except (ValueError, IndexError):
                    idx = 0

                class _FakeExp:
                    pass

                _FakeExp.id = eid
                _FakeExp.index = idx
                best_exp_label_str = exp_label(_FakeExp())
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "kpis": kpis,
                "ordinals": ordinals,
                "best_study_label": best_study_label,
                "best_exp_label_str": best_exp_label_str,
            },
        )

    @app.get("/api/dashboard")
    def api_dashboard():
        from lab.core.dashboard import compute_kpis

        return JSONResponse(compute_kpis(studies_root).model_dump(mode="json"))

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

    @app.get("/api/studies/{study_id}/log")
    def api_study_log(study_id: str, tail: int = 500):
        path = studies_root / study_id / "run.log.jsonl"
        if not path.exists():
            return JSONResponse([])
        lines = [
            l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        return JSONResponse(lines[-tail:])

    @app.get("/api/experiments/{experiment_id}/stdout")
    def api_experiment_stdout(
        experiment_id: str,
        tail_bytes: int = 4096,
        study_id: str | None = None,
    ):
        """Live tail of the subprocess stdout for a running experiment.

        Sandbox is per-study (``sandbox/<study_id>/<exp_id>/``) so the same
        ``exp_0000`` from two different studies doesn't collide. Falls back
        to the legacy flat layout when ``study_id`` is omitted (tests).
        """
        base = Path(settings.paths.sandbox)
        candidates = []
        if study_id:
            candidates.append(base / study_id / experiment_id / "stdout.log")
        candidates.append(base / experiment_id / "stdout.log")
        for path in candidates:
            if path.exists():
                size = path.stat().st_size
                with path.open("rb") as fh:
                    if size > tail_bytes:
                        fh.seek(size - tail_bytes)
                    chunk = fh.read()
                return JSONResponse(
                    {
                        "text": chunk.decode("utf-8", errors="replace"),
                        "size": size,
                        "path": str(path),
                    }
                )
        return JSONResponse({"text": "", "size": 0, "path": None})

    # ----- run form + live -----

    @app.get("/new", response_class=HTMLResponse)
    def new_study_form(request: Request):
        from lab.core.llm_catalog import candidate_models

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
                "llm_models": candidate_models(
                    settings.llm.model, base_url=settings.llm.base_url
                ),
                "data_subset_percentages": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "defaults": {
                    "task": settings.default_task,
                    "max_experiments": settings.compute_budget.max_experiments,
                    "max_wallclock_min": settings.compute_budget.max_wallclock_minutes,
                    "llm_model": settings.llm.model,
                    "data_subset": settings.compute_budget.data_subset_percent,
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
            llm_model=payload.get("llm_model"),
            data_subset=payload.get("data_subset"),
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
        llm_model: str | None = Form(default=None),
        data_subset: int = Form(default=100),
    ):
        _launch_study(
            task=task,
            predecessor=predecessor_id or None,
            use_best_prompts=use_best_prompts,
            agent_memory=agent_memory,
            personality=personality,
            max_experiments=max_experiments,
            max_wallclock_min=max_wallclock_min,
            llm_model=llm_model or None,
            data_subset=data_subset,
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
        llm_model: str | None = None,
        data_subset: int | None = None,
    ) -> dict:
        # Pre-flight: refuse to launch if neither lazy index nor eager
        # shards are present.
        processed = Path(settings.task.processed_data_dir)
        if not (
            (processed / "train.pt").exists()
            or (processed / "train_index.json").exists()
        ):
            raise HTTPException(
                status_code=412,
                detail=(
                    f"no processed dataset at {processed}. "
                    "Run `lab preprocess` (lazy, all samples) or "
                    "`lab preprocess --samples-per-class N` before launching."
                ),
            )
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
        args.llm_model = llm_model
        args.data_subset = data_subset

        study_id = new_study_id()
        threading.Thread(target=cmd_run, args=(args,), daemon=True).start()
        return {"study_id": study_id}

    @app.get("/live/{study_id}")
    def live(study_id: str, replay: int = 0):
        """SSE stream of telemetry events.

        Defaults to streaming only NEW events (cursor starts at file EOF) so
        clients that already loaded historical events via
        ``/api/studies/<id>/log`` do not see duplicates. Pass ``?replay=1`` to
        stream the full file from the start.
        """
        path = studies_root / study_id / "run.log.jsonl"

        def event_stream() -> Iterator[bytes]:
            cursor = 0 if replay else (path.stat().st_size if path.exists() else 0)
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
                yield b": keep-alive\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
