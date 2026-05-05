"""Ollama tag discovery + UI integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from lab.config import load_settings
from lab.core.llm_catalog import candidate_models, list_ollama_models
from lab.ui.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fake_response(payload: dict):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    return _Resp()


# ---- list_ollama_models ----

def test_list_ollama_models_returns_sorted_unique_names():
    payload = {
        "models": [
            {"name": "qwen3-coder:30b"},
            {"name": "gemma4:e4b"},
            {"name": "qwen3-coder:30b"},  # duplicate
            {"name": ""},                  # empty filtered
        ]
    }
    with patch(
        "lab.core.llm_catalog.urllib.request.urlopen",
        return_value=_fake_response(payload),
    ):
        names = list_ollama_models()
    assert names == ["gemma4:e4b", "qwen3-coder:30b"]


def test_list_ollama_models_returns_empty_when_unreachable():
    import urllib.error

    with patch(
        "lab.core.llm_catalog.urllib.request.urlopen",
        side_effect=urllib.error.URLError("no route"),
    ):
        assert list_ollama_models() == []


# ---- candidate_models ----

def test_candidate_models_puts_default_first(monkeypatch):
    monkeypatch.setattr(
        "lab.core.llm_catalog.list_ollama_models",
        lambda *a, **k: ["a:latest", "b:latest", "c:latest"],
    )
    out = candidate_models("b:latest")
    assert out[0] == "b:latest"
    assert set(out) == {"a:latest", "b:latest", "c:latest"}


def test_candidate_models_includes_default_when_missing_from_live(monkeypatch):
    monkeypatch.setattr(
        "lab.core.llm_catalog.list_ollama_models",
        lambda *a, **k: ["a:latest"],
    )
    out = candidate_models("custom-model:tag")
    assert out[0] == "custom-model:tag"
    assert "a:latest" in out


def test_candidate_models_handles_unreachable(monkeypatch):
    """When Ollama is down, fall back to just the configured default."""
    monkeypatch.setattr(
        "lab.core.llm_catalog.list_ollama_models",
        lambda *a, **k: [],
    )
    assert candidate_models("default:tag") == ["default:tag"]


# ---- UI form ----

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lab.core.llm_catalog.list_ollama_models",
        lambda *a, **k: ["gemma4:e4b", "qwen3-coder:30b"],
    )
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"experiments_dir": str(tmp_path / "studies")})
    s = s.model_copy(update={"paths": new_paths})
    Path(new_paths.experiments_dir).mkdir(parents=True)
    return TestClient(create_app(s))


def test_new_form_renders_model_dropdown(client):
    r = client.get("/new")
    assert r.status_code == 200
    assert 'name="llm_model"' in r.text
    assert "gemma4:e4b" in r.text
    assert "qwen3-coder:30b" in r.text


def test_run_form_threads_llm_model_into_args(client, monkeypatch, tmp_path):
    captured = {}
    import lab.cli
    from lab.config import load_settings

    def fake_cmd_run(args):
        captured["llm_model"] = args.llm_model
        return 0

    monkeypatch.setattr(lab.cli, "cmd_run", fake_cmd_run)

    # Seed a fake train.pt so the pre-flight gate passes.
    s = load_settings("track_b", repo_root=REPO_ROOT)
    Path(s.task.processed_data_dir).mkdir(parents=True, exist_ok=True)
    train_pt = Path(s.task.processed_data_dir) / "train.pt"
    cleanup = not train_pt.exists()
    if cleanup:
        train_pt.touch()
    try:
        r = client.post(
            "/run-form",
            data={
                "task": "track_b",
                "personality": "exploratory",
                "max_experiments": "1",
                "max_wallclock_min": "5",
                "llm_model": "qwen3-coder:30b",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303
        import time
        time.sleep(0.1)
        assert captured.get("llm_model") == "qwen3-coder:30b"
    finally:
        if cleanup:
            train_pt.unlink(missing_ok=True)


# ---- CLI override ----

def test_cli_run_override_changes_settings_llm_model():
    """`--llm-model` flag overrides settings.llm.model in _apply_run_overrides."""
    from lab.cli import _apply_run_overrides

    settings = load_settings("track_b", repo_root=REPO_ROOT)

    class _Args:
        max_experiments = None
        max_wallclock_min = None
        personality = None
        agent_memory = False
        llm_model = "qwen3-coder:30b"

    new_settings = _apply_run_overrides(settings, _Args())
    assert new_settings.llm.model == "qwen3-coder:30b"
    # other fields untouched
    assert new_settings.llm.provider == settings.llm.provider
    assert new_settings.llm.base_url == settings.llm.base_url


def test_cli_run_no_override_leaves_model():
    from lab.cli import _apply_run_overrides

    settings = load_settings("track_b", repo_root=REPO_ROOT)

    class _Args:
        max_experiments = None
        max_wallclock_min = None
        personality = None
        agent_memory = False
        llm_model = None

    new_settings = _apply_run_overrides(settings, _Args())
    assert new_settings.llm.model == settings.llm.model
