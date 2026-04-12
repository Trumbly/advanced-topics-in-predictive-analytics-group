"""FastAPI app factory for the BirdCLEF dashboard.

`create_app(studies_root, sandbox_root)` returns a ready-to-serve
`FastAPI` instance. Call it from the `agent ui` CLI command or from
`uvicorn agent.ui:create_app --factory` for manual dev launches.

The app is stateless — every request reads the JSON files under
`studies_root` fresh. That means any study written to disk (live or
past) shows up without restarting the server, and there is no
orchestrator coupling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent.models import Study
from agent.submission import (
    NoBestExperimentError,
    SubmissionError,
    SubmissionExporter,
    SubmissionValidationError,
)
from agent.ui.loaders import (
    find_running_experiment,
    list_studies,
    load_experiment_detail,
    load_study_detail,
    read_stdout_tail,
)


_UI_DIR = Path(__file__).parent
_TEMPLATES_DIR = _UI_DIR / "templates"
_STATIC_DIR = _UI_DIR / "static"


def create_app(
    studies_root: Path | str = Path("experiments/studies"),
    sandbox_root: Path | str = Path("sandbox"),
) -> FastAPI:
    """Build the FastAPI app.

    Parameters
    ----------
    studies_root:
        Path to the directory containing per-study subdirectories.
        Defaults to `experiments/studies` (the orchestrator default).
    sandbox_root:
        Path to the sandbox directory, where code.py / stdout.log /
        stderr.log per experiment live. Defaults to `sandbox`.
    """
    studies_root = Path(studies_root).resolve()
    sandbox_root = Path(sandbox_root).resolve()

    app = FastAPI(
        title="BirdCLEF Agent Dashboard",
        description=(
            "Read-only web UI for the autonomous BirdCLEF 2026 ML "
            "research agent. Browses studies, experiments, generated "
            "code, training curves, and the error-recovery task chain."
        ),
        version="0.1.0",
    )

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    # Expose Python helpers to templates.
    templates.env.globals["format_score"] = _format_score
    templates.env.globals["truncate"] = _truncate

    if _STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(_STATIC_DIR)),
            name="static",
        )

    # ---- HTML routes ------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        studies = list_studies(studies_root)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "studies": studies,
                "studies_root": str(studies_root),
            },
        )

    @app.get("/studies/{study_id}", response_class=HTMLResponse)
    def study_detail(request: Request, study_id: str) -> Any:
        detail = load_study_detail(studies_root, study_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Study {study_id} not found")
        return templates.TemplateResponse(
            request,
            "study.html",
            {
                "detail": detail,
                "study": detail.study,
            },
        )

    @app.get(
        "/studies/{study_id}/experiments/{experiment_id}",
        response_class=HTMLResponse,
    )
    def experiment_detail(
        request: Request, study_id: str, experiment_id: str
    ) -> Any:
        detail = load_experiment_detail(
            studies_root, sandbox_root, study_id, experiment_id
        )
        if detail is None:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {experiment_id} not found in {study_id}",
            )
        return templates.TemplateResponse(
            request,
            "experiment.html",
            {
                "detail": detail,
                "study_id": study_id,
            },
        )

    @app.get("/studies/{study_id}/report", response_class=HTMLResponse)
    def study_report(request: Request, study_id: str) -> Any:
        detail = load_study_detail(studies_root, study_id)
        if detail is None or not detail.has_report:
            raise HTTPException(status_code=404, detail="Report not found")
        report_md = detail.report_path.read_text() if detail.report_path else ""
        return templates.TemplateResponse(
            request,
            "report.html",
            {
                "study": detail.study,
                "report_md": report_md,
                "detail": detail,
            },
        )

    # ---- JSON endpoints (chart data) -------------------------------------

    @app.get("/api/studies/{study_id}/score_progression")
    def api_score_progression(study_id: str) -> JSONResponse:
        detail = load_study_detail(studies_root, study_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Study not found")
        return JSONResponse(
            {
                "metric": detail.score_metric,
                "points": detail.score_progression,
            }
        )

    @app.get("/api/studies/{study_id}/failure_breakdown")
    def api_failure_breakdown(study_id: str) -> JSONResponse:
        detail = load_study_detail(studies_root, study_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Study not found")
        return JSONResponse(
            {
                "total": detail.failure_breakdown.total_failed,
                "by_error_type": detail.failure_breakdown.by_error_type,
            }
        )

    @app.get(
        "/api/studies/{study_id}/experiments/{experiment_id}/curves"
    )
    def api_training_curves(
        study_id: str, experiment_id: str
    ) -> JSONResponse:
        detail = load_experiment_detail(
            studies_root, sandbox_root, study_id, experiment_id
        )
        if detail is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return JSONResponse(detail.training_curves)

    @app.get("/api/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "studies_root": str(studies_root),
                "sandbox_root": str(sandbox_root),
            }
        )

    # ---- Live monitoring (HTMX partials + JSON) --------------------------

    @app.get("/api/studies/{study_id}/running")
    def api_running_experiment(study_id: str) -> JSONResponse:
        """JSON snapshot of the currently-running experiment for a study.

        Used by an optional JS poller to hydrate a header badge. The
        main study-detail page uses the HTMX partial variant below.
        """
        snapshot = find_running_experiment(studies_root, sandbox_root, study_id)
        if snapshot is None:
            return JSONResponse({"running": False})
        return JSONResponse(
            {
                "running": True,
                "experiment_id": snapshot.experiment_id,
                "stdout_last_modified": snapshot.stdout_last_modified_iso,
                "seconds_since_last_write": snapshot.seconds_since_last_write,
            }
        )

    @app.get(
        "/partials/studies/{study_id}/running",
        response_class=HTMLResponse,
    )
    def partial_running_card(request: Request, study_id: str) -> Any:
        """HTMX partial: renders the 'currently running' card (or nothing).

        The study-detail page polls this every few seconds via HTMX
        so the card appears as soon as an experiment starts writing
        output and disappears once it completes.
        """
        snapshot = find_running_experiment(studies_root, sandbox_root, study_id)
        return templates.TemplateResponse(
            request,
            "_partials/running_card.html",
            {"running": snapshot, "study_id": study_id},
        )

    @app.get(
        "/partials/studies/{study_id}/experiments/{experiment_id}/stdout",
        response_class=HTMLResponse,
    )
    def partial_stdout_tail(
        request: Request, study_id: str, experiment_id: str
    ) -> Any:
        """HTMX partial: last N lines of a sandbox experiment's stdout.log.

        The experiment-detail page polls this every few seconds so the
        operator can watch training progress live without refreshing
        the whole page.
        """
        tail = read_stdout_tail(sandbox_root, study_id, experiment_id)
        return templates.TemplateResponse(
            request,
            "_partials/stdout_tail.html",
            {"stdout_tail": tail},
        )

    @app.get(
        "/partials/studies/{study_id}/stats",
        response_class=HTMLResponse,
    )
    def partial_study_stats(request: Request, study_id: str) -> Any:
        """HTMX partial: just the stat cards (experiments / completed /
        failed / best score). Cheaper than re-rendering the entire
        study-detail page every 3 seconds."""
        detail = load_study_detail(studies_root, study_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Study not found")
        return templates.TemplateResponse(
            request,
            "_partials/study_stats.html",
            {"detail": detail, "study": detail.study},
        )

    # ---- Kaggle submission -----------------------------------------------

    @app.post("/api/studies/{study_id}/submit")
    def api_submit_study(study_id: str) -> JSONResponse:
        """Export the best experiment of `study_id` as a Kaggle notebook.

        Wraps `SubmissionExporter.export_best` so the UI can offer a
        single 'Export Kaggle submission' button. Returns the path of
        the new notebook on success, or a 4xx with the validation
        message on failure.
        """
        study_dir = studies_root / study_id
        study_json = study_dir / "study.json"
        if not study_json.exists():
            raise HTTPException(status_code=404, detail=f"Study {study_id} not found")

        try:
            study = Study.from_json_file(study_json)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"Could not load study.json: {exc}",
            ) from exc

        if not study.best_experiment_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Study has no best_experiment_id yet — run or finish "
                    "a study with at least one successful experiment first."
                ),
            )

        exporter = SubmissionExporter()
        try:
            output_path = exporter.export_best(
                study=study,
                study_dir=study_dir,
                sandbox_dir=sandbox_root / study_id,
            )
        except NoBestExperimentError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SubmissionValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SubmissionError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Update the study.json's `submissions` list so the UI sees the
        # new entry on the next refresh. We do this defensively: the
        # exporter doesn't touch study.json.
        already = [Path(p) for p in study.submissions]
        if output_path not in already:
            study.submissions.append(output_path)
            study_json.write_text(study.model_dump_json())

        return JSONResponse(
            {
                "ok": True,
                "study_id": study_id,
                "best_experiment_id": study.best_experiment_id,
                "submission_path": str(output_path),
                "submission_filename": output_path.name,
            }
        )

    return app


# ---------------------------------------------------------------------------
# Template globals
# ---------------------------------------------------------------------------


def _format_score(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _truncate(text: str | None, max_len: int = 120) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


__all__ = ["create_app"]
