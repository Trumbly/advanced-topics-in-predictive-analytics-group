"""Task adapter registry.

Every task YAML has an ``adapter:`` key of the form
``module.path:ClassName``. This module resolves that string to a class and
instantiates it with the merged ``Settings``.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from lab.config import Settings, load_settings
from lab.tasks.base import TaskAdapter


def list_available_tasks(settings: Settings | None = None) -> list[dict[str, str]]:
    """Enumerate tasks by scanning the tasks directory. Used by the UI."""
    settings = settings or load_settings()
    tasks_dir = settings.abspath(settings.paths.tasks_dir)
    out: list[dict[str, str]] = []
    if not tasks_dir.exists():
        return out
    for path in sorted(tasks_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue
        metrics_cfg = data.get("metrics", {}) or {}
        primary_metric = metrics_cfg.get("primary") or ""
        others = metrics_cfg.get("others", []) or []
        all_metrics: list[str] = []
        if isinstance(primary_metric, str) and primary_metric.strip():
            all_metrics.append(primary_metric.strip())
        for m in others:
            if isinstance(m, str) and m.strip() and m.strip() not in all_metrics:
                all_metrics.append(m.strip())
        out.append({
            "name": data.get("name", path.stem),
            "kind": data.get("kind", ""),
            "description": (data.get("description") or "").strip(),
            "yaml_path": str(path),
            "primary_metric": primary_metric,
            "metrics": ",".join(all_metrics),
        })
    return out


def _resolve_adapter_class(spec: str) -> type[TaskAdapter]:
    module_path, _, class_name = spec.partition(":")
    if not module_path or not class_name:
        raise ValueError(
            f"adapter spec must be 'module.path:ClassName', got {spec!r}"
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, TaskAdapter):
        raise TypeError(f"{spec} is not a TaskAdapter subclass")
    return cls


def get_task_adapter(task_name: str | None = None, *, settings: Settings | None = None) -> TaskAdapter:
    """Load settings for ``task_name`` (or the default), and instantiate its adapter."""
    settings = settings or load_settings(task=task_name)
    spec = settings.task_config.get("adapter")
    if not spec:
        raise ValueError(
            f"task {settings.task_config.get('name', '?')}: adapter spec missing in task YAML"
        )
    cls = _resolve_adapter_class(spec)
    return cls(settings)


__all__ = ["get_task_adapter", "list_available_tasks"]
