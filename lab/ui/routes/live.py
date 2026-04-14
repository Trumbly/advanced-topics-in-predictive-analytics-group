"""Server-Sent Events stream for live stdout tail."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from lab.ui.live import format_sse, tail_file


router = APIRouter()


@router.get("/live/{exp_id}")
async def live_stdout(exp_id: str, request: Request):
    settings = request.app.state.settings
    if not settings.ui.enable_live_sse:
        return {"error": "disabled"}, 404
    sandbox_root = settings.abspath(settings.paths.sandbox)
    stdout_path = sandbox_root / exp_id / "stdout.log"

    async def gen():
        async for line in tail_file(stdout_path):
            if await request.is_disconnected():
                break
            yield format_sse(line)

    return StreamingResponse(gen(), media_type="text/event-stream")
