"""Start & stop agent subprocesses from the dashboard."""
from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from lab.ui import launches


router = APIRouter()


def _form_list(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


@router.post("/run")
async def start_study(
    request: Request,
    task: str = Form(""),
    name: str = Form(""),
    tags: str = Form(""),
    predecessor: str = Form(""),
    report: str = Form("false"),
    executor: str = Form(""),
):
    settings = request.app.state.settings
    if not task:
        raise HTTPException(400, "task is required")

    # Prompt overrides come through as repeated ``prompt_<task>`` form fields
    # (the Form() shorthand can't express a dict parameter, so we pull from
    # the raw form).
    form = await request.form()
    overrides: dict[str, str] = {}
    for key, raw in form.multi_items():
        if key.startswith("prompt_") and isinstance(raw, str) and raw.strip():
            overrides[key[len("prompt_"):]] = raw.strip()

    launch = launches.spawn(
        settings.repo_root,
        task=task,
        name=name,
        tags=tags,
        predecessor=predecessor,
        report=report.lower() in {"1", "true", "on", "yes"},
        prompt_overrides=overrides,
        executor_backend=executor.strip(),
    )
    return RedirectResponse(f"/launches/{launch.id}", status_code=303)


@router.post("/launches/{launch_id}/stop")
async def stop_launch(launch_id: str, request: Request):
    settings = request.app.state.settings
    ok = launches.stop_launch(launch_id, settings.repo_root)
    if not ok:
        raise HTTPException(404, f"No launch {launch_id}")
    return RedirectResponse("/", status_code=303)


@router.get("/launches/{launch_id}")
async def launch_detail(launch_id: str, request: Request):
    """Thin view of a launch — mainly useful right after POST /run before
    the orchestrator has registered a study_id."""
    settings = request.app.state.settings
    launch = launches.load_launch(launch_id, settings.repo_root)
    if launch is None:
        raise HTTPException(404)
    # If the study exists already, hop straight to it.
    if launch.study_id:
        return RedirectResponse(f"/studies/{launch.study_id}", status_code=303)
    return request.app.state.templates.TemplateResponse(
        request, "launch.html", {"launch": launch},
    )
