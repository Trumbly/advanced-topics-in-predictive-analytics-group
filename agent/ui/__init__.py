"""Web dashboard for the BirdCLEF agent.

A read-only FastAPI application that surfaces the agent's on-disk state:
studies, experiments, tasks, generated code, training curves, and the
auto-generated study report.

Launch via the CLI command `agent ui`, which starts a uvicorn server
on http://localhost:8000/. The dashboard reads directly from the
`experiments/studies/` and `sandbox/` directories — it does not need
a running orchestrator, and any study that has been written to disk
(live or past) will show up.

Scope for v1 (this module):
  - Index with a list of all studies
  - Study detail with experiment table + score progression chart +
    failure breakdown chart + best experiment highlighted
  - Experiment detail with code viewer, prompts, training curves,
    and the full task chain (propose / generate / validate / execute
    / capture / recovery / analyze)

Scope deferred to v2:
  - Live WebSocket monitoring / auto-refresh
  - Start / Stop / Pause controls
  - Prompt A/B testing
  - Kaggle submission button
  - Comparison view (agent vs manual baselines)

The architecture follows the spec in the repo wiki's Architecture page:
FastAPI backend, Jinja2 + HTMX + Chart.js + Tailwind (all CDN-delivered,
no build step), serving server-rendered pages with optional async
refresh partials.
"""

from __future__ import annotations


def create_app(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazy re-export so `agent.ui.loaders` can be imported without
    pulling in FastAPI."""
    from agent.ui.app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
