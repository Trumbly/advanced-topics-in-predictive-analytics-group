"""Task picker endpoints — the UI dropdown consumes these."""
from __future__ import annotations

from fastapi import APIRouter, Request

from lab.tasks.registry import list_available_tasks


router = APIRouter(prefix="/tasks")


@router.get("")
async def all_tasks(request: Request):
    return {"tasks": list_available_tasks(request.app.state.settings)}
