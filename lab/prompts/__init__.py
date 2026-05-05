"""Prompt registry + engine (ADR-004)."""
from lab.prompts.engine import MissingSlotError, PromptEngine
from lab.prompts.registry import PromptRegistry, PromptTemplate, PROMPT_TASKS

__all__ = [
    "PROMPT_TASKS",
    "MissingSlotError",
    "PromptEngine",
    "PromptRegistry",
    "PromptTemplate",
]
