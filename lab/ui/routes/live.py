"""Server-Sent Events stream for live stdout tail."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from lab.ui import loaders
from lab.ui.live import format_sse, tail_file


router = APIRouter()


def _resolve_exp_stdout(settings, study_id: str | None, exp_id: str) -> Path:
    """Find the experiment's stdout.log via its stored absolute
    sandbox_path when possible, else fall back to constructing the
    path from the configured sandbox_root + exp_id."""
    sandbox_root = settings.abspath(settings.paths.sandbox)
    if study_id:
        experiments_dir = settings.abspath(settings.paths.experiments)
        study = loaders.load_study(experiments_dir, study_id)
        if study:
            exp = next((e for e in study.experiments if e.id == exp_id), None)
            if exp and exp.sandbox_path:
                return Path(exp.sandbox_path) / "stdout.log"
    return sandbox_root / exp_id / "stdout.log"


@router.get("/live/{exp_id}")
async def live_stdout(exp_id: str, request: Request):
    settings = request.app.state.settings
    if not settings.ui.enable_live_sse:
        return {"error": "disabled"}, 404
    stdout_path = _resolve_exp_stdout(settings, None, exp_id)

    async def gen():
        async for line in tail_file(stdout_path):
            if await request.is_disconnected():
                break
            yield format_sse(line)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/live/study/{study_id}/exp/{exp_id}")
async def live_stdout_via_study(study_id: str, exp_id: str, request: Request):
    """Same as ``/live/<exp_id>`` but resolves the path through the
    experiment's stored ``sandbox_path``, which is authoritative."""
    settings = request.app.state.settings
    if not settings.ui.enable_live_sse:
        return {"error": "disabled"}, 404
    stdout_path = _resolve_exp_stdout(settings, study_id, exp_id)

    async def gen():
        async for line in tail_file(stdout_path):
            if await request.is_disconnected():
                break
            yield format_sse(line)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/live/launches/{launch_id}")
async def live_launch(launch_id: str, request: Request):
    """SSE stream for an agent subprocess log (combined stdout + stderr)."""
    settings = request.app.state.settings
    if not settings.ui.enable_live_sse:
        return {"error": "disabled"}, 404
    log_path = settings.abspath(f"experiments/launches/{launch_id}.log")

    async def gen():
        async for line in tail_file(log_path):
            if await request.is_disconnected():
                break
            yield format_sse(line)

    return StreamingResponse(gen(), media_type="text/event-stream")
