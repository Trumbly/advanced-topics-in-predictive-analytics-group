"""Settings page: GET renders the form, POST persists + validates."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from lab.config import load_settings
from lab.ui.app import create_app
from lab.ui.settings_io import (
    parse_form_into_yaml,
    read_global_yaml,
    validate_yaml,
    write_global_yaml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root(tmp_path):
    """Stage a minimal repo layout under tmp so the settings round-trip
    does not clobber the real config.yaml during the test suite."""
    (tmp_path / "config" / "tasks").mkdir(parents=True)
    (tmp_path / "config" / "config.yaml").write_text(
        (REPO_ROOT / "config" / "config.yaml").read_text()
    )
    (tmp_path / "config" / "tasks" / "track_b.yaml").write_text(
        (REPO_ROOT / "config" / "tasks" / "track_b.yaml").read_text()
    )
    return tmp_path


# ---- io helpers ----

def test_read_global_yaml(repo_root):
    payload = read_global_yaml(repo_root)
    assert payload["project"] == "lab"
    assert "compute_budget" in payload


def test_parse_form_into_yaml_coerces_types(repo_root):
    current = read_global_yaml(repo_root)
    form = {
        "compute_budget.max_experiments": "33",
        "compute_budget.lr_min": "1.0e-5",
        "compute_budget.device": "mps",
        "compute_budget.num_workers": "4",
        "agent.memory_enabled": "true",
        "logging.level": "DEBUG",
    }
    out = parse_form_into_yaml(current, form)
    assert out["compute_budget"]["max_experiments"] == 33
    assert out["compute_budget"]["lr_min"] == pytest.approx(1.0e-5)
    assert out["compute_budget"]["device"] == "mps"
    assert out["compute_budget"]["num_workers"] == 4
    assert out["agent"]["memory_enabled"] is True
    assert out["logging"]["level"] == "DEBUG"


def test_validate_yaml_passes_for_valid_payload(repo_root):
    payload = read_global_yaml(repo_root)
    # Force the new fields to known defaults so the test is independent of
    # whatever the live config.yaml has been edited to via the UI.
    payload["compute_budget"]["device"] = "cpu"
    payload["compute_budget"]["num_workers"] = 0
    payload["compute_budget"]["batch_size"] = 8
    settings = validate_yaml(repo_root, payload)
    assert settings.compute_budget.device == "cpu"
    assert settings.compute_budget.num_workers == 0
    assert settings.compute_budget.batch_size == 8


def test_validate_yaml_rejects_invalid_device(repo_root):
    from pydantic import ValidationError

    payload = read_global_yaml(repo_root)
    payload["compute_budget"]["device"] = "tpu"  # not in Literal
    with pytest.raises(ValidationError):
        validate_yaml(repo_root, payload)


def test_write_global_yaml_persists(repo_root):
    payload = read_global_yaml(repo_root)
    payload["compute_budget"]["max_experiments"] = 99
    write_global_yaml(repo_root, payload)
    reread = yaml.safe_load((repo_root / "config" / "config.yaml").read_text())
    assert reread["compute_budget"]["max_experiments"] == 99


# ---- routes ----

@pytest.fixture
def client(tmp_path, monkeypatch):
    # Run the test against a copy so we never touch the real config.yaml.
    work = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "config", work / "config")

    monkeypatch.setattr(
        "lab.config._default_root", lambda: work
    )
    settings = load_settings("track_b", repo_root=work)
    return TestClient(create_app(settings))


def test_settings_get_renders_with_current_config(client):
    r = client.get("/settings")
    assert r.status_code == 200
    assert "Compute budget" in r.text
    assert 'name="compute_budget.device"' in r.text
    assert 'name="compute_budget.num_workers"' in r.text
    assert 'name="llm.model"' in r.text


def test_settings_post_persists_and_redirects(client, monkeypatch):
    r = client.post(
        "/settings",
        data={
            "compute_budget.max_experiments": "42",
            "compute_budget.device": "mps",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/settings?saved=1"


def test_settings_post_invalid_redirects_with_error(client):
    r = client.post(
        "/settings",
        data={"compute_budget.device": "tpu"},  # invalid
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "/settings?error=" in r.headers["location"]


def test_nav_link_present(client):
    r = client.get("/dashboard")
    assert 'href="/settings"' in r.text
