"""Slot-filling prompt engine (ADR-004)."""

from __future__ import annotations

import re
from typing import Iterable

from lab.prompts.registry import PromptRegistry, PromptTask, PromptTemplate

_SLOT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class MissingSlotError(KeyError):
    """Raised when a template references a slot the caller did not supply."""

    def __init__(self, slot: str, *, task: str | None = None, where: str | None = None):
        self.slot = slot
        self.task = task
        self.where = where
        msg = f"missing slot '{{{slot}}}'"
        if task:
            msg += f" while rendering task '{task}'"
        if where:
            msg += f" in {where}"
        super().__init__(msg)


class PromptEngine:
    """Render a versioned `PromptTemplate` by substituting `{slot}` placeholders."""

    def __init__(self, registry: PromptRegistry):
        self.registry = registry

    def render(
        self,
        task: PromptTask,
        slots: dict[str, str],
        version: str | None = None,
    ) -> tuple[str, str]:
        tmpl: PromptTemplate = self.registry.load(task, version)
        rendered_system = self._fill(tmpl.system, slots, task=task, where="system")
        rendered_user = self._fill(tmpl.user, slots, task=task, where="user")
        return rendered_system, rendered_user

    def extract_slots(self, template: str) -> set[str]:
        return set(_SLOT_RE.findall(template))

    @staticmethod
    def _fill(template: str, slots: dict[str, str], *, task: str, where: str) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in slots:
                raise MissingSlotError(name, task=task, where=where)
            return str(slots[name])

        return _SLOT_RE.sub(replace, template)

    @staticmethod
    def required_slots(template: str) -> set[str]:
        return set(_SLOT_RE.findall(template))

    @staticmethod
    def validate_slots(template: str, provided: Iterable[str]) -> set[str]:
        """Return slot names referenced by `template` but missing in `provided`."""
        return set(_SLOT_RE.findall(template)) - set(provided)
