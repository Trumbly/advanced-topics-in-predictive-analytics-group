"""Prompt engine — slot filling for versioned prompt templates.

Templates use Python ``str.format`` single-brace slots (``{slot_name}``).
Missing slots raise ``MissingSlotError`` with the full list so the caller
can't fail in a mysterious way at the LLM API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from lab.prompts.registry import PromptRegistry


class MissingSlotError(KeyError):
    pass


_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def extract_slots(template: str) -> list[str]:
    return sorted(set(_SLOT_RE.findall(template)))


def fill(template: str, slots: dict[str, Any]) -> str:
    missing = [s for s in extract_slots(template) if s not in slots]
    if missing:
        raise MissingSlotError(f"missing prompt slots: {missing}")

    class _Safe(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    # Use format_map so stray braces in user values don't blow up.
    return template.format_map(_Safe({k: _to_str(v) for k, v in slots.items()}))


def _to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        import json
        return json.dumps(value, indent=2, default=str)
    except Exception:  # noqa: BLE001
        return str(value)


@dataclass
class PromptEngine:
    registry: PromptRegistry

    def render(
        self,
        task: str,
        slots: dict[str, Any],
        *,
        version: str | None = None,
    ) -> tuple[str, str]:
        """Return ``(system, user)`` strings for ``task`` with ``slots`` filled."""
        prompt = self.registry.load_prompt(task, version=version)
        system = fill(prompt.get("system", ""), slots)
        user = fill(prompt.get("template", ""), slots)
        return system, user


__all__ = ["PromptEngine", "MissingSlotError", "fill", "extract_slots"]
