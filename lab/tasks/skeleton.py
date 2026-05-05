"""Render the audio-multilabel training skeleton from the Jinja template.

Used by the orchestrator (I-06) to materialise the skeleton on disk per
experiment, with the LLM-authored ``build_model`` block spliced into the
START/END markers.
"""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import StrictUndefined, Template

from lab.config import Settings


_BUILD_START = "# --- AGENT_BUILD_MODEL_START ---"
_BUILD_END = "# --- AGENT_BUILD_MODEL_END ---"


def render_skeleton(settings: Settings) -> str:
    """Render the Jinja template for the configured task."""
    path = Path(settings.task.skeleton_path)
    if not path.exists():
        raise FileNotFoundError(f"skeleton template not found: {path}")
    return Template(path.read_text(encoding="utf-8"), undefined=StrictUndefined).render(
        num_classes=settings.task.expected_num_classes,
        input_tensor_shape=list(settings.task.input_tensor_shape),
        primary_metric=settings.task.primary_metric,
    )


def splice_build_model(rendered_skeleton: str, build_model_block: str) -> str:
    """Replace the default build_model block with an LLM-authored block."""
    if _BUILD_START not in rendered_skeleton or _BUILD_END not in rendered_skeleton:
        raise ValueError("rendered skeleton is missing AGENT_BUILD_MODEL markers")

    head, _, rest = rendered_skeleton.partition(_BUILD_START)
    _, _, tail = rest.partition(_BUILD_END)
    return f"{head}{_BUILD_START}\n{build_model_block.strip()}\n{_BUILD_END}{tail}"
