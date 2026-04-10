"""Load YAML prompt templates and fill them with runtime slots.

A prompt template is a YAML file under `config/prompts/` that follows the
`PromptTemplate` schema (see `agent.models.PromptTemplate`). The ContextHandler
calls `PromptEngine.fill()` with a dict of slot values to produce the final
system + user messages that are sent to the LLM.

Template format
---------------
```yaml
name: propose_architecture
description: LLM proposes a CNN architecture
system: |
  You are an ML research agent.
template: |
  ## Dataset Profile
  {dataset_profile}

  ## Your Task
  Propose an architecture.
slots:
  - dataset_profile
fallback_when_no_memory: |
  No experiments yet. Start with a small CNN.
response_schema:
  type: object
  ...
```

Design decisions
----------------
- Slot filling uses `str.format_map` with a custom mapping that raises on
  missing keys (so typos surface loudly).
- Unknown keys in the slot dict are silently ignored — the LLM may return
  extra fields we don't care about.
- The template keeps its own declared `slots` list; `fill()` validates that
  every declared slot is provided, but does not validate that the template
  body actually uses them (YAML authors may reference slots inside conditional
  blocks in future).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agent.models import PromptTemplate


class MissingSlotError(KeyError):
    """Raised when a required slot is missing from the provided values."""


class PromptEngine:
    """Loads and fills prompt templates.

    Templates are cached by path after the first load. Call `clear_cache()`
    during development if you edit a YAML file and want the change picked up
    without restarting.
    """

    def __init__(self) -> None:
        self._cache: dict[Path, PromptTemplate] = {}

    # -- loading ------------------------------------------------------------

    def load(self, path: Path | str) -> PromptTemplate:
        """Load and validate a prompt template from disk."""
        path = Path(path)
        if path in self._cache:
            return self._cache[path]
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: prompt YAML must be a mapping, got {type(data)}")

        template = PromptTemplate.model_validate(data)
        self._cache[path] = template
        return template

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- filling ------------------------------------------------------------

    def fill(
        self,
        template: PromptTemplate | Path | str,
        slots: dict[str, Any],
        *,
        use_fallback: bool = False,
    ) -> tuple[str, str]:
        """Fill a template with slot values.

        Args:
            template: A PromptTemplate, or a path to a YAML file to load.
            slots: Mapping of slot name -> value. Non-string values are
                JSON-serialized (e.g. dicts for response_schema).
            use_fallback: If True, return the `fallback_when_no_memory`
                string as the user message instead of the filled template.
                Used when the experiment memory is empty.

        Returns:
            `(system_message, user_message)` as strings.
        """
        if isinstance(template, (str, Path)):
            template = self.load(template)

        if use_fallback:
            if template.fallback_when_no_memory is None:
                raise ValueError(
                    f"Template '{template.name}' has no fallback_when_no_memory"
                )
            return template.system, template.fallback_when_no_memory

        # Validate every declared slot is provided
        missing = [s for s in template.slots if s not in slots]
        if missing:
            raise MissingSlotError(
                f"Template '{template.name}' is missing slots: {missing}"
            )

        # Serialize non-string values (dicts, lists) to readable strings
        str_slots = {k: _to_prompt_string(v) for k, v in slots.items()}

        # Use format_map so unknown keys don't trip us up, but missing keys do
        try:
            user_message = template.template.format_map(_StrictDict(str_slots))
        except KeyError as e:
            raise MissingSlotError(
                f"Template '{template.name}' references slot {e} "
                f"which was not provided."
            ) from e

        return template.system, user_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StrictDict(dict):  # type: ignore[type-arg]
    """A dict subclass that raises KeyError on missing keys from format_map.

    Without this, Python silently substitutes an empty string for missing
    keys when using `format_map`, which would mask template bugs.
    """

    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def _to_prompt_string(value: Any) -> str:
    """Convert a slot value to a string suitable for prompt injection.

    - str → as-is
    - dict / list → pretty-printed JSON (readable by the LLM)
    - Pydantic model → model_dump_json (pretty)
    - None → empty string
    - everything else → str(value)
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # Pydantic BaseModel (duck-typed to avoid a hard import)
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json(indent=2)
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, default=str)
    return str(value)


__all__ = [
    "PromptEngine",
    "MissingSlotError",
]
