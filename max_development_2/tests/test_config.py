from __future__ import annotations

import os
from pathlib import Path

from lab.config import load_settings


REPO = Path(__file__).resolve().parent.parent


def test_load_default_settings():
    s = load_settings(repo_root=REPO)
    assert s.project == "lab"
    assert s.default_task in {"track_a", "track_b"}
    assert s.env_prefix == "AGENT"


def test_task_config_overrides_compute_budget():
    s = load_settings(repo_root=REPO, task="track_a")
    # Track A is fast; it overrides max_experiment_seconds.
    assert s.compute_budget.max_experiment_seconds == 600


def test_env_overrides_apply():
    os.environ["AGENT__COMPUTE_BUDGET__MAX_EXPERIMENTS"] = "7"
    try:
        s = load_settings(repo_root=REPO)
        assert s.compute_budget.max_experiments == 7
    finally:
        del os.environ["AGENT__COMPUTE_BUDGET__MAX_EXPERIMENTS"]


def test_env_var_helper():
    s = load_settings(repo_root=REPO)
    assert s.env_var("epochs") == "AGENT_EPOCHS"
    assert s.env_var("BATCH_SIZE") == "AGENT_BATCH_SIZE"
