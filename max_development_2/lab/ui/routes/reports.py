"""Study report viewer."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from lab.ui import loaders


router = APIRouter()


@router.get("/studies/{study_id}/report", response_class=HTMLResponse)
async def view_report(study_id: str, request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    report_md = experiments_dir / study_id / "report" / "report.md"
    if not report_md.exists():
        raise HTTPException(
            404,
            "Report not yet generated. Run `lab report <study_id>` to build it.",
        )
    content = report_md.read_text()
    return request.app.state.templates.TemplateResponse(
        "report.html",
        {"request": request, "study_id": study_id, "markdown": content},
    )


@router.get("/studies/{study_id}/report/figures/{name}")
async def report_figure(study_id: str, name: str, request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    path = experiments_dir / study_id / "report" / "figures" / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)
