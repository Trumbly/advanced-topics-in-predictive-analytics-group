"""Assemble the final LLM prompt from a template + memory + profile + registry.

The ContextHandler is the glue between PromptEngine and ExperimentMemory.
For a given prompt template, it knows how to populate every slot:

- `dataset_profile`    → DatasetProfile serialized as markdown
- `model_registry`     → ModelRegistry.to_markdown()
- `experiment_memory`  → ExperimentMemory.to_markdown(top_k)
- `last_result`        → most recent Experiment serialized as markdown
- `response_schema`    → prompt_template.response_schema as JSON
- ...other slots       → passed through from the caller

It also enforces a rough token budget: if the assembled user message exceeds
`max_prompt_tokens`, the memory slot is progressively trimmed (fewer entries)
until it fits. This is a coarse heuristic — it assumes memory is the biggest
variable-sized slot, which is true in practice for this project.

ExperimentMemory is not implemented in Phase 2 (it's part of Phase 3). This
module uses a small Protocol so the ContextHandler can be tested with a mock
memory today, and we'll plug in the real one in Phase 3 without changing
this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.llm_client import LLMClient
from agent.models import DatasetProfile, Experiment, PromptTemplate
from agent.prompt_engine import PromptEngine


# ---------------------------------------------------------------------------
# Dependency protocols (so Phase 2 doesn't hard-depend on Phase 3 modules)
# ---------------------------------------------------------------------------


class MemoryLike(Protocol):
    """Minimal surface of an ExperimentMemory that ContextHandler needs."""

    def to_markdown(self, top_k: int = 5) -> str: ...
    def is_empty(self) -> bool: ...
    def last(self) -> Experiment | None: ...


class RegistryLike(Protocol):
    """Minimal surface of a ModelRegistry that ContextHandler needs."""

    def to_markdown(self) -> str: ...


# ---------------------------------------------------------------------------
# Simple in-memory stub (useful as default when nothing is passed in)
# ---------------------------------------------------------------------------


@dataclass
class EmptyMemory:
    """A no-op memory for the very first experiment of a study.

    Returns empty markdown and reports `is_empty() == True`, which triggers
    the `fallback_when_no_memory` branch in PromptEngine.
    """

    def to_markdown(self, top_k: int = 5) -> str:
        return ""

    def is_empty(self) -> bool:
        return True

    def last(self) -> Experiment | None:
        return None


# ---------------------------------------------------------------------------
# Context handler
# ---------------------------------------------------------------------------


@dataclass
class ContextHandler:
    """Fills prompt templates with live agent state.

    Required dependencies:
        prompt_engine:  loads and fills YAML templates
        llm_client:     used only for token counting (not for chat calls)
        dataset_profile: the study's DatasetProfile
        registry:       a ModelRegistry (for the `model_registry` slot)
        memory:         an ExperimentMemory (for the `experiment_memory` slot)

    Optional:
        max_prompt_tokens:  token budget for the final user message
        min_memory_entries: never trim memory below this (keep at least K
                            experiments so the LLM retains some history)
    """

    prompt_engine: PromptEngine
    llm_client: LLMClient
    dataset_profile: DatasetProfile
    registry: RegistryLike
    memory: MemoryLike = field(default_factory=EmptyMemory)
    max_prompt_tokens: int = 8000
    initial_memory_top_k: int = 5
    min_memory_entries: int = 1

    # -- public API ---------------------------------------------------------

    def build(
        self,
        template: PromptTemplate,
        *,
        extra_slots: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Assemble and return the final (system, user) messages.

        Args:
            template: the loaded PromptTemplate to fill.
            extra_slots: additional slot values provided by the caller
                (e.g. `architecture_proposal` for the generate_code step).
                These are merged into the base slots and override them
                if there's a name collision.

        Returns:
            `(system_message, user_message)` ready to hand to `LLMClient.chat`.
        """
        # Cold-start: first experiment in the study has no memory → fallback
        if self.memory.is_empty() and template.fallback_when_no_memory is not None:
            return self.prompt_engine.fill(template, slots={}, use_fallback=True)

        # Build the base slots once
        base_slots = self._base_slots(template, self.initial_memory_top_k)
        if extra_slots:
            base_slots.update(extra_slots)

        system, user = self.prompt_engine.fill(template, base_slots)

        # Token budget enforcement — trim memory if we're over the limit
        top_k = self.initial_memory_top_k
        while (
            self.llm_client.count_tokens(system) + self.llm_client.count_tokens(user)
            > self.max_prompt_tokens
            and top_k > self.min_memory_entries
        ):
            top_k -= 1
            base_slots = self._base_slots(template, top_k)
            if extra_slots:
                base_slots.update(extra_slots)
            system, user = self.prompt_engine.fill(template, base_slots)

        return system, user

    # -- base slot assembly -------------------------------------------------

    def _base_slots(self, template: PromptTemplate, top_k: int) -> dict[str, Any]:
        """Populate the slots ContextHandler knows how to fill on its own."""
        slots: dict[str, Any] = {}
        declared = set(template.slots)

        if "dataset_profile" in declared:
            slots["dataset_profile"] = _profile_to_markdown(self.dataset_profile)
        if "model_registry" in declared:
            slots["model_registry"] = self.registry.to_markdown()
        if "experiment_memory" in declared:
            slots["experiment_memory"] = (
                self.memory.to_markdown(top_k=top_k) or "(no previous experiments)"
            )
        if "last_result" in declared:
            last = self.memory.last()
            slots["last_result"] = _experiment_to_markdown(last) if last else "(none)"
        if "response_schema" in declared:
            slots["response_schema"] = (
                json.dumps(template.response_schema, indent=2)
                if template.response_schema
                else "{}"
            )
        return slots


# ---------------------------------------------------------------------------
# Markdown serializers
# ---------------------------------------------------------------------------


def _profile_to_markdown(profile: DatasetProfile) -> str:
    """Compact markdown description of a DatasetProfile for prompt injection.

    Kept terse intentionally — full class stats would blow the token budget
    for 234 classes. We summarize: num classes, shape, imbalance, a few
    extreme classes (most / least populated).
    """
    lines = [
        f"- num_classes: {profile.num_classes}",
        f"- num_samples: {profile.num_samples}",
        f"- spectrogram_shape: {profile.spectrogram_shape}",
        f"- sample_rate: {profile.sample_rate} Hz",
        f"- imbalance_ratio: {profile.imbalance_ratio:.2f} "
        f"(min={profile.min_class_samples}, max={profile.max_class_samples})",
        f"- split: {profile.split_strategy} (seed={profile.split_seed})",
        f"- train/val: {len(profile.train_indices)}/{len(profile.val_indices)}",
    ]
    # Show the 3 most and least populated classes
    if profile.class_stats:
        sorted_stats = sorted(profile.class_stats, key=lambda c: c.sample_count)
        tail = sorted_stats[:3]
        head = sorted_stats[-3:][::-1]
        lines.append("- most-populated classes:")
        for c in head:
            lines.append(f"    - {c.class_id}: {c.sample_count}")
        lines.append("- least-populated classes:")
        for c in tail:
            lines.append(f"    - {c.class_id}: {c.sample_count}")
    return "\n".join(lines)


def _experiment_to_markdown(experiment: Experiment) -> str:
    """One-experiment summary (the `last_result` slot)."""
    lines = [
        f"- id: {experiment.experiment_id}",
        f"- status: {experiment.status}",
        f"- llm_model: {experiment.llm_model}",
    ]
    if experiment.config:
        lines.append(f"- architecture: {experiment.config.architecture}")
        if experiment.config.pretrained_model:
            lines.append(f"- pretrained: {experiment.config.pretrained_model}")
        lines.append(f"- hyperparams: {json.dumps(experiment.config.hyperparams)}")
    if experiment.results:
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(experiment.results.metrics.items())
        )
        lines.append(f"- metrics: {metrics_str}")
        lines.append(f"- duration: {experiment.results.duration_seconds:.1f}s")
    return "\n".join(lines)


__all__ = [
    "ContextHandler",
    "EmptyMemory",
    "MemoryLike",
    "RegistryLike",
]
