"""Versioned prompt registry.

Layout on disk (immutable file versions + one mutable pointer file):

    config/prompts/
      _registry.yaml                  # mutable: which version is active
      propose_architecture/
        v1.yaml                       # immutable
        v2.yaml                       # immutable
      generate_code/
        v1.yaml
      ...

The public API resolves the active version for a given task name and
returns a ``Path`` to the YAML file. It also lists all versions so the UI
A/B dashboard can show a dropdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PromptVersion:
    name: str
    version: str
    path: Path
    description: str = ""
    created_at: str | None = None


@dataclass
class PromptRegistry:
    root: Path
    _cache: dict[str, Any] = field(default_factory=dict)

    @property
    def pointer_path(self) -> Path:
        return self.root / "_registry.yaml"

    # ---- read ---------------------------------------------------------

    def _load_pointer(self) -> dict[str, Any]:
        if not self.pointer_path.exists():
            raise FileNotFoundError(f"No prompt registry at {self.pointer_path}")
        with self.pointer_path.open() as fh:
            data = yaml.safe_load(fh) or {}
        return data

    def list_tasks(self) -> list[str]:
        return sorted(self._load_pointer().keys())

    def list_versions(self, task: str) -> list[PromptVersion]:
        data = self._load_pointer().get(task, {})
        versions = data.get("versions", {})
        out: list[PromptVersion] = []
        for vname, meta in sorted(versions.items()):
            out.append(PromptVersion(
                name=task,
                version=vname,
                path=self.root / meta["path"],
                description=meta.get("description", ""),
                created_at=meta.get("created_at"),
            ))
        return out

    def active_version(self, task: str) -> str:
        data = self._load_pointer().get(task)
        if data is None:
            raise KeyError(f"No prompt entry for task {task!r}")
        return data.get("active", "v1")

    def resolve_active_path(self, task: str) -> Path:
        v = self.active_version(task)
        data = self._load_pointer()[task]
        rel = data["versions"][v]["path"]
        return self.root / rel

    def load_prompt(self, task: str, version: str | None = None) -> dict[str, Any]:
        v = version or self.active_version(task)
        data = self._load_pointer()[task]
        rel = data["versions"][v]["path"]
        with (self.root / rel).open() as fh:
            return yaml.safe_load(fh) or {}

    # ---- write --------------------------------------------------------

    def delete_version(self, task: str, version: str) -> None:
        """Delete a prompt version.

        Refuses if:
          * The task has only one version left (we never leave a task
            with zero prompts — the registry would be unusable).
          * The version being deleted is the active one (must `set_active`
            to a different version first).
        """
        data = self._load_pointer()
        if task not in data:
            raise KeyError(task)
        versions = data[task].get("versions", {})
        if version not in versions:
            raise KeyError(version)
        if len(versions) <= 1:
            raise ValueError(
                f"cannot delete the last remaining version of {task!r}"
            )
        if data[task].get("active") == version:
            raise ValueError(
                f"cannot delete the active version {version!r} of {task!r}; "
                "activate a different version first"
            )
        # Remove the file and the pointer entry.
        rel = versions[version].get("path")
        if rel:
            abs_path = self.root / rel
            if abs_path.exists():
                abs_path.unlink()
        del versions[version]
        data[task]["versions"] = versions
        with self.pointer_path.open("w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)

    def set_active(self, task: str, version: str) -> None:
        data = self._load_pointer()
        if task not in data:
            raise KeyError(task)
        if version not in data[task].get("versions", {}):
            raise KeyError(version)
        data[task]["active"] = version
        with self.pointer_path.open("w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)

    def save_new_version(
        self,
        task: str,
        *,
        content: dict[str, Any],
        description: str = "",
        make_active: bool = False,
    ) -> PromptVersion:
        """Write a new immutable version file and update the pointer.

        Returns the new ``PromptVersion``.
        """
        data = self._load_pointer()
        existing = data.get(task, {"versions": {}, "active": "v1"})
        versions = existing.setdefault("versions", {})
        next_num = max((int(v[1:]) for v in versions if v.startswith("v")), default=0) + 1
        new_v = f"v{next_num}"
        rel = f"{task}/{new_v}.yaml"
        abs_path = self.root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        content.setdefault("name", task)
        content.setdefault("version", new_v)
        if description:
            content.setdefault("description", description)
        with abs_path.open("w") as fh:
            yaml.safe_dump(content, fh, sort_keys=False, allow_unicode=True)
        versions[new_v] = {
            "path": rel,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if make_active:
            existing["active"] = new_v
        data[task] = existing
        with self.pointer_path.open("w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)
        return PromptVersion(
            name=task, version=new_v, path=abs_path, description=description
        )


__all__ = ["PromptRegistry", "PromptVersion"]
