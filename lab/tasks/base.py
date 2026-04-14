"""TaskAdapter ABC.

An adapter binds the task-agnostic agent to a specific track. Everything
that differs between tracks — data paths, metric names, input shapes,
submission format, prompt slot values — is provided by an adapter instance.

Nothing in ``lab.core`` is allowed to special-case a task kind. If a value
varies by track, it MUST come from an adapter method.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from lab.config import Settings
from lab.core.models import DatasetProfile


class TaskAdapter(ABC):
    #: Short name of the task (e.g. "track_a"). Must match the task YAML file name.
    name: str = ""

    #: Pretty kind tag (e.g. "text_classification_binary").
    kind: str = ""

    #: The metric that the agent optimises (e.g. "f1_binary", "f1_macro").
    primary_metric: str = ""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.task_cfg: dict[str, Any] = settings.task_config

    # ------------------------------------------------------------------
    # Paths & resources
    # ------------------------------------------------------------------

    def code_skeleton_path(self) -> Path:
        rel = self.task_cfg.get("prompt_slots", {}).get("code_skeleton")
        if rel is None:
            raise ValueError(
                f"task {self.name}: prompt_slots.code_skeleton missing in task YAML"
            )
        return self.settings.abspath(rel)

    def model_registry_path(self) -> Path:
        rel = self.task_cfg.get("model", {}).get("registry")
        if rel is None:
            raise ValueError(f"task {self.name}: model.registry missing in task YAML")
        return self.settings.abspath(rel)

    # ------------------------------------------------------------------
    # Prompt slots — the only task-specific knowledge the prompts see
    # ------------------------------------------------------------------

    def prompt_slot_values(self) -> dict[str, Any]:
        return dict(self.task_cfg.get("prompt_slots", {}))

    # ------------------------------------------------------------------
    # Validator hook
    # ------------------------------------------------------------------

    def spawn_triggering_calls(self) -> Iterable[str]:
        """Extra function names that, when called at module scope,
        must be inside ``if __name__ == "__main__":``. The validator's
        default set already includes ``DataLoader``; adapters add their
        own data-loader entry-point names here."""
        return ()

    # ------------------------------------------------------------------
    # Env vars injected into the sandbox subprocess
    # ------------------------------------------------------------------

    def env_vars(self, settings: Settings) -> dict[str, str]:
        """Extra env vars merged on top of the default training env.

        Default impl injects absolute paths to the raw/processed data so
        generated code can find its dataset without any hardcoding.
        """
        prefix = settings.env_prefix
        raw = self.task_cfg.get("data", {}).get("raw", {})
        env: dict[str, str] = {}
        for k, v in raw.items():
            env[f"{prefix}_{k.upper()}"] = str(settings.abspath(v))
        processed = self.task_cfg.get("data", {}).get("processed_dir")
        if processed:
            env[f"{prefix}_PROCESSED_DIR"] = str(settings.abspath(processed))
        env[f"{prefix}_TASK"] = self.name
        return env

    # ------------------------------------------------------------------
    # Contracts the adapter MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def build_profile(self) -> DatasetProfile:
        """Scan the raw data on disk and return a typed profile."""

    def load_profile(self) -> DatasetProfile:
        """Return the cached profile, (re)building on disk miss.

        Default implementation reads ``processed_dir/dataset_profile.json``
        if present and calls ``build_profile`` otherwise.
        """
        processed = self.settings.abspath(
            self.task_cfg.get("data", {}).get("processed_dir", "data/processed")
        )
        cache = processed / "dataset_profile.json"
        if cache.exists():
            import json
            try:
                data = json.loads(cache.read_text())
            except json.JSONDecodeError:
                data = {}
            # Backward-compat: profiles from earlier builds didn't carry
            # task_name / kind. Fill them in from the adapter so old
            # caches still load instead of forcing a re-preprocess of
            # multi-GB datasets just to populate two strings.
            if isinstance(data, dict):
                data.setdefault("task_name", self.name)
                data.setdefault("kind", self.kind)
                try:
                    return DatasetProfile.model_validate(data)
                except Exception:  # noqa: BLE001
                    pass  # fall through to a fresh rebuild
        profile = self.build_profile()
        processed.mkdir(parents=True, exist_ok=True)
        cache.write_text(profile.model_dump_json(indent=2))
        return profile

    def validate_training_output(self, results_json: dict[str, Any]) -> list[str]:
        """Return a list of error strings if the training output is invalid.

        Empty list = ok. Default implementation checks the primary metric is
        present and numeric; adapters can add more checks.
        """
        errors: list[str] = []
        if results_json.get("primary_metric") != self.primary_metric:
            errors.append(
                f"results.primary_metric={results_json.get('primary_metric')!r} but "
                f"task requires {self.primary_metric!r}"
            )
        score = results_json.get("primary_score")
        if not isinstance(score, (int, float)):
            errors.append("results.primary_score missing or non-numeric")
        return errors

    @abstractmethod
    def build_submission(self, experiment_code: str, experiment_id: str, out_dir: Path) -> Path:
        """Write a runnable submission artifact and return its path."""


__all__ = ["TaskAdapter"]
