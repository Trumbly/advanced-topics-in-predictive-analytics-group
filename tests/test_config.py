"""I-01 acceptance: YAML-only loader, no env overrides, strict validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lab.config import (
    AgentConfig,
    ComputeBudget,
    LLMConfig,
    Settings,
    load_settings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_default_track_b_settings():
    s = load_settings("track_b", repo_root=REPO_ROOT)
    assert isinstance(s, Settings)
    assert s.project == "lab"
    assert s.task_name == "track_b"
    assert s.task.name == "track_b"
    assert s.task.expected_num_classes == 206
    assert s.compute_budget.max_experiments > 0
    assert s.llm.provider in {"ollama", "openai", "anthropic"}


def test_env_var_is_ignored(monkeypatch):
    """Env overrides must NOT take effect (ADR-002)."""
    monkeypatch.setenv("AGENT__COMPUTE_BUDGET__MAX_EXPERIMENTS", "999")
    monkeypatch.setenv("AGENT__LLM__MODEL", "spy-model")
    s = load_settings("track_b", repo_root=REPO_ROOT)
    assert s.compute_budget.max_experiments != 999
    assert s.llm.model != "spy-model"


def test_unknown_task_raises_filenotfound():
    with pytest.raises(FileNotFoundError) as exc:
        load_settings("does_not_exist", repo_root=REPO_ROOT)
    msg = str(exc.value)
    assert "does_not_exist.yaml" in msg


def test_missing_required_key_raises_validation_error(tmp_path: Path):
    """Removing a required key from the global YAML triggers ValidationError at load time."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "tasks").mkdir()

    # Copy task YAML verbatim
    src_task = REPO_ROOT / "config" / "tasks" / "track_b.yaml"
    (cfg_dir / "tasks" / "track_b.yaml").write_text(src_task.read_text())

    # Drop a required key from global config
    src_cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    src_cfg.pop("compute_budget")
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(src_cfg))

    with pytest.raises(ValidationError):
        load_settings("track_b", repo_root=tmp_path)


def test_extra_field_is_rejected(tmp_path: Path):
    """Strict schema: unknown YAML keys must fail validation, not silently pass."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "tasks").mkdir()
    (cfg_dir / "tasks" / "track_b.yaml").write_text(
        (REPO_ROOT / "config" / "tasks" / "track_b.yaml").read_text()
    )

    src_cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    src_cfg["unknown_top_level_field"] = "boom"
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(src_cfg))

    with pytest.raises(ValidationError):
        load_settings("track_b", repo_root=tmp_path)


def test_settings_round_trip_dict():
    s = load_settings("track_b", repo_root=REPO_ROOT)
    payload = s.model_dump()
    again = Settings(**payload)
    assert again == s


def test_individual_models_are_strict():
    with pytest.raises(ValidationError):
        LLMConfig(provider="bogus", base_url="x", model="y", temperature=0.1, max_tokens=1, retry_attempts=0, retry_backoff_seconds=0.0)
    with pytest.raises(ValidationError):
        ComputeBudget(max_experiments=-1, max_wallclock_minutes=1, max_experiment_seconds=1, max_epochs_per_run=1, max_codegen_retries=0, max_recovery_attempts=0, lr_min=1e-5, lr_max=1e-3)
    with pytest.raises(ValidationError):
        AgentConfig(personality="rogue")  # type: ignore[arg-type]
