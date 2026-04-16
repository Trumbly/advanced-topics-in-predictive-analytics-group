"""Experiment detail pages."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from lab.ui import launches, loaders
from lab.ui.charts import history_to_series, line_chart_svg


router = APIRouter()


def _resolve_stdout_path(experiment, sandbox_root: Path) -> Path:
    """Pick the most reliable path to the running training script's stdout.

    The executor stores the absolute sandbox dir on `experiment.sandbox_path`
    once the experiment has actually started running. That's the
    authoritative location — using it avoids drift between the agent
    process's CWD and the UI's CWD.
    """
    if experiment.sandbox_path:
        return Path(experiment.sandbox_path) / "stdout.log"
    return sandbox_root / experiment.id / "stdout.log"


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

    stdout_path = _resolve_stdout_path(exp, sandbox_root)
    live_stdout = ""
    if stdout_path.exists():
        text = stdout_path.read_text(errors="replace")
        live_stdout = "\n".join(text.splitlines()[-500:])

    # Also surface the orchestrator log when a launch is attached to the
    # study — that's where "currently running propose_architecture" lines
    # appear, complementing the training-script stdout above.
    running_launch = launches.launch_for_study(study_id, settings.repo_root)
    launch_log_tail = ""
    if running_launch:
        log_path = settings.repo_root / running_launch.log_path
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            launch_log_tail = "\n".join(text.splitlines()[-200:])

    loss_chart_svg = line_chart_svg(
        history_to_series(exp.history, ("loss",)),
        y_label="train loss",
    )
    # Primary metric + whatever else is numeric in the history — pick the
    # common audio/text metrics that the skeletons actually emit.
    metric_names = tuple(dict.fromkeys([
        exp.primary_metric,
        "f1_macro",
        "f1_binary",
        "roc_auc_macro",
        "roc_auc_binary",
        "accuracy",
    ]))
    metric_chart_svg = line_chart_svg(
        history_to_series(exp.history, metric_names),
        y_label="validation metrics",
    )

    return request.app.state.templates.TemplateResponse(
        request,
        "experiment.html",
        {
            "study": study,
            "experiment": exp,
            "live_stdout": live_stdout,
            "stdout_path_display": str(stdout_path),
            "stdout_exists": stdout_path.exists(),
            "running_launch": running_launch,
            "launch_log_tail": launch_log_tail,
            "loss_chart_svg": loss_chart_svg,
            "metric_chart_svg": metric_chart_svg,
        },
    )
