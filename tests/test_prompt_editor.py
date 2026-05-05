"""Prompt editor UI: view, save-as-new, save-and-activate."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from lab.config import load_settings
from lab.ui.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client(tmp_path):
    """Point both `prompts_dir` and `experiments_dir` at tmp so the test
    cannot mutate the repo's real prompt files."""
    s = load_settings("track_b", repo_root=REPO_ROOT)

    prompts_root = tmp_path / "prompts"
    (prompts_root / "propose_architecture").mkdir(parents=True)
    (prompts_root / "propose_architecture" / "v1.yaml").write_text(
        yaml.safe_dump(
            {"system": "system v1", "user": "user v1 with {personality}"}
        ),
        encoding="utf-8",
    )
    (prompts_root / "propose_architecture" / "v2.yaml").write_text(
        yaml.safe_dump({"system": "system v2", "user": "user v2"}),
        encoding="utf-8",
    )
    (prompts_root / "_registry.yaml").write_text(
        yaml.safe_dump({"propose_architecture": "v1"}),
        encoding="utf-8",
    )

    new_paths = s.paths.model_copy(
        update={
            "experiments_dir": str(tmp_path / "studies"),
            "prompts_dir": str(prompts_root),
        }
    )
    s = s.model_copy(update={"paths": new_paths})
    Path(new_paths.experiments_dir).mkdir(parents=True)

    return TestClient(create_app(s)), prompts_root


# ---------- view ----------


def test_prompts_dashboard_links_to_editor(client):
    c, _ = client
    r = c.get("/prompts")
    assert r.status_code == 200
    assert "/prompts/propose_architecture/v1" in r.text
    assert "/prompts/propose_architecture/new/blank" in r.text


def test_view_specific_version_renders_textareas(client):
    c, _ = client
    r = c.get("/prompts/propose_architecture/v1")
    assert r.status_code == 200
    assert "system v1" in r.text
    assert "user v1 with {personality}" in r.text
    assert 'name="system"' in r.text
    assert 'name="user"' in r.text


def test_view_unknown_task_404(client):
    c, _ = client
    r = c.get("/prompts/does_not_exist/v1")
    assert r.status_code == 404


def test_view_unknown_version_404(client):
    c, _ = client
    r = c.get("/prompts/propose_architecture/v99")
    assert r.status_code == 404


def test_root_task_redirects_to_active(client):
    c, _ = client
    r = c.get("/prompts/propose_architecture", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/prompts/propose_architecture/v1"


def test_blank_form_seeded_from_active(client):
    c, _ = client
    r = c.get("/prompts/propose_architecture/new/blank")
    assert r.status_code == 200
    # seeded from active (v1), not from empty
    assert "system v1" in r.text


# ---------- save ----------


def test_save_creates_new_version_and_redirects(client):
    c, root = client
    r = c.post(
        "/prompts/propose_architecture/edit",
        data={"system": "system v3 new", "user": "user v3 new"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/prompts/propose_architecture/v3?saved=1"

    # disk: new immutable file at v3, others untouched
    v3 = (root / "propose_architecture" / "v3.yaml").read_text()
    assert "system v3 new" in v3
    assert "user v3 new" in v3
    v1 = (root / "propose_architecture" / "v1.yaml").read_text()
    assert "system v1" in v1  # never overwritten

    # registry NOT changed (no `activate` form value)
    registry = yaml.safe_load((root / "_registry.yaml").read_text())
    assert registry["propose_architecture"] == "v1"


def test_save_and_activate_flips_registry(client):
    c, root = client
    r = c.post(
        "/prompts/propose_architecture/edit",
        data={"system": "sys", "user": "usr", "activate": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    registry = yaml.safe_load((root / "_registry.yaml").read_text())
    assert registry["propose_architecture"] == "v3"


def test_save_for_new_task_creates_dir(client):
    c, root = client
    r = c.post(
        "/prompts/judge_experiment/edit",
        data={"system": "j sys", "user": "j usr"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/prompts/judge_experiment/v1?saved=1"
    assert (root / "judge_experiment" / "v1.yaml").exists()


def test_save_returns_400_when_field_missing(client):
    c, _ = client
    r = c.post(
        "/prompts/propose_architecture/edit",
        data={"system": "only system"},
    )
    # FastAPI validation error
    assert r.status_code in (400, 422)
