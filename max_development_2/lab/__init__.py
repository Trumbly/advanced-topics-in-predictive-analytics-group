"""lab — the task-agnostic LLM research agent.

See docs/REDESIGN_PLAN.md for the architecture overview. The public entry
points are:

    from lab.config import load_settings
    from lab.core.orchestrator import Orchestrator
    from lab.tasks.registry import get_task_adapter

Run ``python -m lab --help`` for the CLI.
"""
from __future__ import annotations

__version__ = "2.0.0"
