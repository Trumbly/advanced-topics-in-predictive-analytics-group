"""Immutable versioned prompt registry (ADR-004).

Layout::

    config/prompts/
      _registry.yaml                # active version pointer per task
      <task>/v1.yaml v2.yaml ...    # immutable version files

`_registry.yaml` is the only mutable file. Version files never overwrite.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel

PromptTask = Literal[
    "propose_architecture",
    "generate_code",
    "recover_from_error",
    "analyze_result",
    "judge_experiment",
    "judge_study",
]

PROMPT_TASKS: tuple[str, ...] = get_args(PromptTask)

_REGISTRY_FILE = "_registry.yaml"
_VERSION_RE = re.compile(r"^v(\d+)\.yaml$")


class PromptTemplate(BaseModel):
    """A loaded prompt with both system and user halves and provenance."""

    system: str
    user: str
    task: str
    version: str
    path: Path

    model_config = {"arbitrary_types_allowed": True}


class PromptRegistry:
    """Versioned prompt store under `root` (typically `config/prompts/`)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    # ---------- read ----------

    def active_version(self, task: PromptTask) -> str:
        registry = self._load_registry()
        try:
            return registry[task]
        except KeyError as exc:
            raise KeyError(f"no active version for task '{task}' in {self._registry_path}") from exc

    def list_versions(self, task: PromptTask) -> list[str]:
        d = self.root / task
        if not d.exists():
            return []
        versions = []
        for entry in d.iterdir():
            m = _VERSION_RE.match(entry.name)
            if m:
                versions.append(entry.stem)
        versions.sort(key=lambda v: int(v[1:]))
        return versions

    def load(self, task: PromptTask, version: str | None = None) -> PromptTemplate:
        v = version or self.active_version(task)
        path = self.root / task / f"{v}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"prompt not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            return PromptTemplate(
                system=data["system"],
                user=data["user"],
                task=task,
                version=v,
                path=path,
            )
        except KeyError as exc:
            raise ValueError(f"prompt {path} missing required key: {exc}") from exc

    # ---------- write ----------

    def set_active(self, task: PromptTask, version: str) -> None:
        if (self.root / task / f"{version}.yaml").exists() is False:
            raise FileNotFoundError(f"version not on disk: {task}/{version}.yaml")
        registry = self._load_registry()
        registry[task] = version
        self._atomic_write_registry(registry)

    def save_new_version(self, task: PromptTask, system: str, user: str) -> str:
        d = self.root / task
        d.mkdir(parents=True, exist_ok=True)

        existing = self.list_versions(task)
        next_n = (max(int(v[1:]) for v in existing) + 1) if existing else 1
        new_version = f"v{next_n}"
        path = d / f"{new_version}.yaml"

        if path.exists():  # paranoia: never overwrite
            raise FileExistsError(f"refusing to overwrite {path}")

        path.write_text(
            yaml.safe_dump({"system": system, "user": user}, sort_keys=False),
            encoding="utf-8",
        )
        return new_version

    # ---------- internals ----------

    @property
    def _registry_path(self) -> Path:
        return self.root / _REGISTRY_FILE

    def _load_registry(self) -> dict[str, str]:
        if not self._registry_path.exists():
            return {}
        return yaml.safe_load(self._registry_path.read_text(encoding="utf-8")) or {}

    def _atomic_write_registry(self, payload: dict[str, str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # write-to-temp + rename for atomic replace
        fd, tmp_path = tempfile.mkstemp(
            prefix=".registry-", suffix=".tmp", dir=str(self.root)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.safe_dump(payload, fh, sort_keys=True)
            os.replace(tmp_path, self._registry_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
