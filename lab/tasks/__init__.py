"""Task adapters: encapsulate task-specific knowledge (ADR-003)."""
from lab.tasks.base import DatasetProfile, TaskAdapter
from lab.tasks.registry import get_task_adapter

__all__ = ["DatasetProfile", "TaskAdapter", "get_task_adapter"]
