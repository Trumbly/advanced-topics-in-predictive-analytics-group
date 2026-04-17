"""Experiment detail pages."""
from __future__ import annotations

import json
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
        y_label="train loss (per epoch)",
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
    # Per-batch loss curve: much more informative than a single dot when
    # EPOCHS=1, and shows within-epoch convergence shape (noise, plateau,
    # instability) even on multi-epoch runs.
    batch_loss_series = _batch_loss_series(exp.history)
    batch_loss_chart_svg = line_chart_svg(
        batch_loss_series,
        y_label="train loss (per batch)",
    )

    proposal = _parse_proposal(exp.architecture_proposal)

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
            "batch_loss_chart_svg": batch_loss_chart_svg,
            "proposal": proposal,
        },
    )


def _parse_proposal(raw: str | None) -> dict | None:
    """Parse the architecture_proposal JSON into a structured dict for
    the template. Returns None when parsing fails — the template falls
    back to the raw JSON pre-block."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
    except (json.JSONDecodeError, TypeError):
        return None

    # Known hyperparameter keys → rendered as pills.
    hp_keys = {"lr", "lr_schedule", "epochs", "init_from_experiment_id"}
    hyperparams = {}
    for k in hp_keys:
        v = data.get(k)
        if v is not None:
            hyperparams[k] = v

    # Everything that isn't a "known" top-level field goes into extras.
    known = {
        "architecture_name", "architecture_family", "description",
        "reasoning", "hypothesis", "risk", "risk_note", "risk_mitigation",
    } | hp_keys
    extras = {k: v for k, v in data.items() if k not in known and v}

    return {
        "name": data.get("architecture_name", ""),
        "family": data.get("architecture_family", ""),
        "description": data.get("description", ""),
        "reasoning": data.get("reasoning") or data.get("hypothesis") or "",
        "risk": data.get("risk") or data.get("risk_note") or "",
        "risk_mitigation": data.get("risk_mitigation") or "",
        "hyperparams": hyperparams,
        "extras": extras,
        "raw": raw,
    }


def _batch_loss_series(history):
    """Flatten per-epoch `batch_losses` into one series with cumulative step.

    If multiple epochs carry batch_losses we concatenate them on a global
    step counter so the chart shows the full training trajectory. If no
    history row carries batch_losses we return an empty series and the
    template skips the chart.
    """
    series: list[tuple[float, float]] = []
    step = 0
    for row in history or []:
        if not isinstance(row, dict):
            continue
        bls = row.get("batch_losses")
        if not isinstance(bls, list):
            continue
        for v in bls:
            if isinstance(v, (int, float)):
                step += 1
                series.append((float(step), float(v)))
    if not series:
        return []
    return [("loss", series)]
