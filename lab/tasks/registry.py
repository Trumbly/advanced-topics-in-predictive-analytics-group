"""Resolve a TaskAdapter from a `module.path:ClassName` import path."""

from __future__ import annotations

import importlib

from lab.config import Settings
from lab.tasks.base import TaskAdapter


def get_task_adapter(settings: Settings) -> TaskAdapter:
    spec = settings.task.adapter
    if ":" not in spec:
        raise ValueError(
            f"task.adapter must be of the form 'module.path:ClassName', got {spec!r}"
        )
    module_path, _, class_name = spec.partition(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"class {class_name!r} not found in module {module_path!r}")
    instance = cls(settings=settings)
    if not isinstance(instance, TaskAdapter):
        raise TypeError(
            f"{spec} resolved to {type(instance).__name__}, not a TaskAdapter"
        )
    return instance
