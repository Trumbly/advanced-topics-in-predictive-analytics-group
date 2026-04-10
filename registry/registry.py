"""Model Registry loader and lookup API.

Loads `registry/models.yaml` into a list of `ModelRegistryEntry` objects
and provides lookup + markdown serialization for LLM prompt injection.

The LLM must reference models by name. At architecture-proposal time,
the ContextHandler calls `ModelRegistry.to_markdown()` to inject the full
catalog into the prompt. When the LLM's response names a pretrained_model,
the orchestrator calls `registry.get(name)` to fetch the import snippet
and pass it into the code generation step.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agent.models import ModelRegistryEntry, ModelRegistryFile


class ModelRegistry:
    """In-memory catalog of ModelRegistryEntry."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._entries: dict[str, ModelRegistryEntry] = {}
        self.load()

    def load(self) -> None:
        """(Re)load the registry from disk."""
        if not self.path.exists():
            raise FileNotFoundError(f"Model registry not found at {self.path}")
        data = yaml.safe_load(self.path.read_text())
        file_model = ModelRegistryFile.model_validate(data)
        self._entries = {entry.name: entry for entry in file_model.models}

    def get(self, name: str) -> ModelRegistryEntry:
        """Look up a model by name. Raises KeyError if not found."""
        if name not in self._entries:
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available: {sorted(self._entries.keys())}"
            )
        return self._entries[name]

    def list_all(self) -> list[ModelRegistryEntry]:
        """Return all entries, sorted by name."""
        return [self._entries[name] for name in sorted(self._entries.keys())]

    def names(self) -> list[str]:
        """Return just the names of all registered models."""
        return sorted(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def to_markdown(self) -> str:
        """Compact markdown catalog for LLM prompt injection.

        Output is intentionally terse to minimize tokens. Each model is
        described in three lines: name, basic specs, and suitability notes.
        """
        lines: list[str] = []
        for entry in self.list_all():
            shape = "x".join(str(d) for d in entry.input_shape)
            lines.append(f"### {entry.name}")
            lines.append(
                f"- family: {entry.family} | framework: {entry.framework} | "
                f"params: {entry.parameters_millions}M | pretrained: {entry.pretrained_on}"
            )
            lines.append(f"- input_shape: {shape} | output_dim: {entry.output_dim}")
            lines.append(f"- notes: {entry.suitability_notes.strip()}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


__all__ = ["ModelRegistry"]
