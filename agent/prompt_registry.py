"""Versioned prompt management for A/B testing.

Prompts are organized as:
    config/prompts/<task_name>/v1.yaml
    config/prompts/<task_name>/v2.yaml
    config/prompts/_registry.yaml

Versions are IMMUTABLE — once created, the YAML file is never modified.
This prevents retroactive score tampering. Only the `_registry.yaml`
metadata (default_version pointer) is mutable.

The `PromptRegistryManager` loads the registry, creates new versions,
resolves prompt paths for a study, and provides the data for the
prompt A/B testing dashboard.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent.models import (
    PromptRegistry,
    PromptTaskEntry,
    PromptVersionMeta,
)

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = "_registry.yaml"


class PromptRegistryManager:
    """Manage versioned, immutable prompt templates on disk."""

    def __init__(self, prompts_dir: Path = Path("config/prompts")) -> None:
        self.prompts_dir = Path(prompts_dir).resolve()
        self._registry_path = self.prompts_dir / _REGISTRY_FILENAME

    # -- Loading -----------------------------------------------------------

    def load_registry(self) -> PromptRegistry:
        """Load _registry.yaml. Returns empty registry if file missing."""
        if not self._registry_path.exists():
            return PromptRegistry()
        data = yaml.safe_load(self._registry_path.read_text()) or {}
        return PromptRegistry.model_validate(data)

    def save_registry(self, registry: PromptRegistry) -> None:
        """Write _registry.yaml back to disk."""
        self._registry_path.write_text(
            yaml.safe_dump(
                registry.model_dump(mode="json"),
                default_flow_style=False,
                sort_keys=False,
            )
        )

    # -- Queries -----------------------------------------------------------

    def list_tasks(self) -> list[str]:
        """Return all task names that have prompt versions."""
        registry = self.load_registry()
        return sorted(registry.tasks.keys())

    def list_versions(self, task_name: str) -> list[PromptVersionMeta]:
        """Return all versions for a task, sorted by version number."""
        registry = self.load_registry()
        entry = registry.tasks.get(task_name)
        if entry is None:
            return []
        versions = list(entry.versions.values())
        versions.sort(key=lambda v: _version_sort_key(v.version))
        return versions

    def get_version_path(self, task_name: str, version: str) -> Path:
        """Return the absolute path to a specific version's YAML file."""
        return self.prompts_dir / task_name / f"{version}.yaml"

    def get_default_version(self, task_name: str) -> str | None:
        """Return the default version string for a task."""
        registry = self.load_registry()
        entry = registry.tasks.get(task_name)
        return entry.default_version if entry else None

    def get_default_path(self, task_name: str) -> Path | None:
        """Return the file path for the default version of a task."""
        version = self.get_default_version(task_name)
        if version is None:
            return None
        path = self.get_version_path(task_name, version)
        return path if path.exists() else None

    # -- Mutations ---------------------------------------------------------

    def create_version(
        self,
        task_name: str,
        content: str,
        *,
        description: str = "",
        created_by: str = "",
    ) -> str:
        """Create a new immutable version. Returns the version string.

        Auto-increments from the highest existing version number.
        Validates that the content is valid YAML before writing.
        Raises ValueError if the task does not exist in the registry.
        """
        registry = self.load_registry()
        entry = registry.tasks.get(task_name)
        if entry is None:
            raise ValueError(
                f"Task {task_name!r} not found in registry. "
                f"Available: {sorted(registry.tasks.keys())}"
            )

        # Validate YAML
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML: {exc}") from exc

        # Auto-increment version
        existing = sorted(entry.versions.keys(), key=_version_sort_key)
        if existing:
            last_num = _version_sort_key(existing[-1])
            new_version = f"v{last_num + 1}"
        else:
            new_version = "v1"

        # Write the immutable file
        task_dir = self.prompts_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        version_path = task_dir / f"{new_version}.yaml"
        if version_path.exists():
            raise ValueError(f"{version_path} already exists!")
        version_path.write_text(content)

        # Update registry
        entry.versions[new_version] = PromptVersionMeta(
            version=new_version,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            description=description,
        )
        self.save_registry(registry)

        logger.info(
            "Created prompt version %s/%s: %s",
            task_name,
            new_version,
            description[:80],
        )
        return new_version

    def set_default(self, task_name: str, version: str) -> None:
        """Set a version as the default for a task."""
        registry = self.load_registry()
        entry = registry.tasks.get(task_name)
        if entry is None:
            raise ValueError(f"Task {task_name!r} not found in registry")
        if version not in entry.versions:
            raise ValueError(
                f"Version {version!r} not found for task {task_name!r}"
            )
        entry.default_version = version
        self.save_registry(registry)

    # -- Resolution (for study start) --------------------------------------

    def resolve_prompt_paths(
        self,
        selection: dict[str, str],
    ) -> dict[str, Path]:
        """Resolve a selection map into concrete file paths.

        Selection values:
          - "default" → use the default version
          - "vN"      → use a specific version
          - "best"    → use the highest-scoring version (requires score data)

        Returns {task_name: Path} for each LLM task.
        """
        registry = self.load_registry()
        result: dict[str, Path] = {}

        for task_name, choice in selection.items():
            entry = registry.tasks.get(task_name)
            if entry is None:
                continue

            if choice == "default":
                version = entry.default_version
            elif choice == "best":
                # "best" needs external score data — fall back to default
                # The actual "best" resolution happens in the UI layer
                # where score aggregation data is available.
                version = entry.default_version
            else:
                version = choice

            path = self.get_version_path(task_name, version)
            if path.exists():
                result[task_name] = path
            else:
                logger.warning(
                    "Prompt version %s/%s not found at %s, skipping",
                    task_name,
                    version,
                    path,
                )

        return result

    # -- Migration ---------------------------------------------------------

    @staticmethod
    def migrate_flat_to_versioned(prompts_dir: Path) -> PromptRegistry:
        """One-time migration: move flat YAML files into task directories.

        `config/prompts/generate_code.yaml` → `config/prompts/generate_code/v1.yaml`

        Returns the generated PromptRegistry.
        """
        now = datetime.now(timezone.utc)
        registry = PromptRegistry()

        for yaml_file in sorted(prompts_dir.glob("*.yaml")):
            if yaml_file.name.startswith("_"):
                continue  # skip _registry.yaml if it already exists

            task_name = yaml_file.stem
            task_dir = prompts_dir / task_name

            # Don't re-migrate if directory already exists
            if task_dir.is_dir() and (task_dir / "v1.yaml").exists():
                logger.info("Already migrated: %s", task_name)
                # Still register it
                registry.tasks[task_name] = PromptTaskEntry(
                    default_version="v1",
                    versions={
                        "v1": PromptVersionMeta(
                            version="v1",
                            created_at=now,
                            created_by="migration",
                            description=f"Original {task_name} prompt (migrated)",
                        )
                    },
                )
                continue

            task_dir.mkdir(exist_ok=True)
            dest = task_dir / "v1.yaml"
            shutil.copy2(yaml_file, dest)
            yaml_file.unlink()

            registry.tasks[task_name] = PromptTaskEntry(
                default_version="v1",
                versions={
                    "v1": PromptVersionMeta(
                        version="v1",
                        created_at=now,
                        created_by="migration",
                        description=f"Original {task_name} prompt (migrated)",
                    )
                },
            )
            logger.info("Migrated %s.yaml → %s/v1.yaml", task_name, task_name)

        # Write registry
        registry_path = prompts_dir / _REGISTRY_FILENAME
        registry_path.write_text(
            yaml.safe_dump(
                registry.model_dump(mode="json"),
                default_flow_style=False,
                sort_keys=False,
            )
        )
        logger.info("Wrote %s with %d tasks", registry_path, len(registry.tasks))

        return registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _version_sort_key(version: str) -> int:
    """Extract the numeric part of a version string (e.g. 'v3' → 3)."""
    try:
        return int(version.lstrip("v"))
    except ValueError:
        return 0


__all__ = [
    "PromptRegistryManager",
]
