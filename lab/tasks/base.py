"""TaskAdapter ABC (ADR-003).

Each adapter encapsulates a task's data shape, prompt slot values, training
skeleton signature, and submission artifact builder. The agent loop is
task-agnostic and reads everything through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from lab.core.models import DatasetProfile


class TaskAdapter(ABC):
    name: str
    kind: str
    primary_metric: str

    # ------- introspection -------

    @abstractmethod
    def profile(self) -> DatasetProfile: ...

    @abstractmethod
    def prompt_slots(self) -> dict[str, str]: ...

    @abstractmethod
    def model_block_signature(self) -> tuple[str, str]:
        """Return (function_name, required_arg_name)."""

    @abstractmethod
    def spawn_triggering_calls(self) -> Iterable[str]:
        """Names of dataloader / multiprocessing calls that need a __main__ guard."""

    # ------- submission -------

    @abstractmethod
    def build_submission(
        self, code: str, experiment_id: str, out_dir: Path
    ) -> Path: ...
