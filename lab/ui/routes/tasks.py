"""Task picker endpoints — the UI dropdown consumes these."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from lab.tasks.registry import list_available_tasks


router = APIRouter(prefix="/tasks")


@router.get("")
async def all_tasks(request: Request):
    return {"tasks": list_available_tasks(request.app.state.settings)}


@router.get("/{task_name}/eda", response_class=HTMLResponse)
async def task_eda(task_name: str, request: Request):
    """Serve the pre-rendered EDA report for a task.

    404 when the report hasn't been generated yet — the error message
    tells the user the CLI command to run.
    """
    settings = request.app.state.settings
    path = settings.abspath(settings.paths.eda_dir) / f"{task_name}.html"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No EDA report for task '{task_name}'. "
                f"Run `python -m lab eda --task {task_name}` to generate it."
            ),
        )
    return HTMLResponse(path.read_text())
