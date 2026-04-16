"""Prompt slot filling.

Decides which values flow into the prompt templates. Handles token-budget
trimming by dropping memory entries until the rendered prompt fits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.config import Settings
from lab.core.memory import Memory
from lab.core.models import DatasetProfile


@dataclass
class ContextBuilder:
    settings: Settings
    memory: Memory
    dataset_profile: DatasetProfile | None = None
    model_registry: list[dict[str, Any]] = field(default_factory=list)
    task_slots: dict[str, str] = field(default_factory=dict)

    def build(self, *, last_result: str = "", architecture_proposal: str = "",
              error_type: str = "", error_message: str = "", error_traceback: str = "",
              broken_code: str = "", **extra: Any) -> dict[str, Any]:
        """Produce a dict of slot values for the prompt engine."""
        top_k = self.settings.context.memory_top_k
        memory_md = self.memory.to_markdown(top_k=top_k)

        dataset_md = self._profile_to_markdown(self.dataset_profile)
        registry_md = self._registry_to_markdown(self.model_registry)

        code_skel = self.task_slots.get("code_skeleton_content", "")

        ctx: dict[str, Any] = {
            # Task-level slots from the task YAML
            "task_description": self.task_slots.get("task_description", ""),
            "input_tensor_shape": self.task_slots.get("input_tensor_shape", ""),
            "output_description": self.task_slots.get("output_description", ""),
            "valid_architecture_families": "\n".join(
                f"  - {fam}" for fam in self.task_slots.get("valid_architecture_families", [])
            ) or "  (any)",
            "code_skeleton_content": code_skel,
            # Runtime slots
            "memory_top_k": top_k,
            "experiment_memory": memory_md,
            "dataset_profile": dataset_md,
            "model_registry": registry_md,
            "last_result": last_result or "_first experiment_",
            "primary_metric": self.memory.score_metric,
            "architecture_proposal": architecture_proposal,
            "error_type": error_type,
            "error_message": error_message,
            "error_traceback": error_traceback,
            "broken_code": broken_code,
            "response_schema": _RESPONSE_SCHEMA_JSON,
            "available_checkpoints": extra.pop(
                "available_checkpoints", "_no archived checkpoints yet_",
            ),
            "eda_summary": extra.pop(
                "eda_summary", "",
            ),
        }
        ctx.update(extra)
        # Fit to budget (best-effort)
        self._shrink_to_budget(ctx)
        return ctx

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _profile_to_markdown(profile: DatasetProfile | None) -> str:
        if profile is None:
            return "_no dataset profile available_"
        lines = [f"- kind: {profile.kind}"]
        if profile.num_classes is not None:
            lines.append(f"- num_classes: {profile.num_classes}")
        if profile.num_train_samples is not None:
            lines.append(f"- num_train: {profile.num_train_samples}")
        if profile.num_val_samples is not None:
            lines.append(f"- num_val: {profile.num_val_samples}")
        for k, v in (profile.extras or {}).items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _registry_to_markdown(registry: list[dict[str, Any]]) -> str:
        if not registry:
            return "_registry empty_"
        lines = []
        for e in registry:
            name = e.get("name", "?")
            fam = e.get("family", "?")
            desc = e.get("description", "").replace("\n", " ").strip()
            lines.append(f"- [{fam}] **{name}** — {desc}")

            backbone = e.get("backbone")
            weights = e.get("weights")
            timm_name = e.get("timm")
            pretrained = e.get("pretrained")
            in_ch = e.get("input_channels")
            how = (e.get("how_to_use") or "").replace("\n", " ").strip()

            sub_parts: list[str] = []
            if backbone:
                w = f'(weights="{weights}")' if weights else "()"
                sub_parts.append(f"torchvision: `{backbone}{w}`")
            if timm_name:
                sub_parts.append(f'timm: `timm.create_model("{timm_name}", pretrained=True)`')
            if in_ch is not None:
                sub_parts.append(f"expects {in_ch}-channel input")
            if pretrained is False:
                sub_parts.append("no pretrained weights")
            if sub_parts:
                lines.append("  · " + "; ".join(sub_parts))
            if how:
                lines.append(f"  · how: {how}")
        return "\n".join(lines)

    def _shrink_to_budget(self, ctx: dict[str, Any]) -> None:
        """Best-effort trim to stay under settings.context.max_prompt_tokens."""
        budget = self.settings.context.max_prompt_tokens
        trimmed = True
        while trimmed:
            rendered = json.dumps(ctx, default=str)
            approx_tokens = max(1, len(rendered) // 4)
            if approx_tokens <= budget:
                return
            trimmed = False
            # IMPORTANT: never trim `code_skeleton_content` here. If we chop the
            # skeleton, model-generation can regress to tiny scripts that define
            # only `build_model()` and never run training.
            for field_name, hard_cap in (
                ("broken_code", 4000),
                ("error_traceback", 3000),
                ("experiment_memory", 2500),
                ("model_registry", 2500),
                ("last_result", 1500),
                ("dataset_profile", 1200),
            ):
                v = ctx.get(field_name, "")
                if isinstance(v, str) and len(v) > hard_cap:
                    ctx[field_name] = v[:hard_cap] + "\n# ... [truncated] ..."
                    trimmed = True
                    break


_RESPONSE_SCHEMA_JSON = (
    '{\n'
    '  "architecture_name": "string",\n'
    '  "architecture_family": "string (one of the valid families above)",\n'
    '  "description": "one-sentence description",\n'
    '  "reasoning": "why this architecture is worth trying next",\n'
    '  "lr": 1e-3,                     // optional float in [1e-5, 1e-1]\n'
    '  "lr_schedule": "constant",      // optional: constant | cosine | onecycle\n'
    '  "init_from_experiment_id": null // optional: continue from a past run\n'
    '}'
)


def load_model_registry(path: Path) -> list[dict[str, Any]]:
    import yaml
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or []
    if isinstance(data, dict):
        data = data.get("models", [])
    return data


__all__ = ["ContextBuilder", "load_model_registry"]
