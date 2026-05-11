"""I-04 acceptance: prompt registry + engine."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lab.prompts import (
    MissingSlotError,
    PROMPT_TASKS,
    PromptEngine,
    PromptRegistry,
    PromptTemplate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_PROMPTS = REPO_ROOT / "config" / "prompts"


def _make_empty_registry(tmp_path: Path) -> PromptRegistry:
    return PromptRegistry(tmp_path / "config" / "prompts")


# -------- shipped prompts --------

def test_all_shipped_tasks_have_loadable_active():
    """The active version for every shipped task points at a real, non-empty
    template. Specific version pins live in `config/prompts/_registry.yaml`
    and are intentionally allowed to evolve — the test only enforces that
    whatever is pinned actually exists and renders both system + user
    blocks."""
    reg = PromptRegistry(SHIPPED_PROMPTS)
    for task in PROMPT_TASKS:
        active = reg.active_version(task)
        assert active, task
        tmpl = reg.load(task)
        assert isinstance(tmpl, PromptTemplate)
        assert tmpl.system.strip()
        assert tmpl.user.strip()


def test_engine_renders_propose_architecture_with_full_slots():
    reg = PromptRegistry(SHIPPED_PROMPTS)
    eng = PromptEngine(reg)
    slots = {
        "task_description": "Multi-label bird classification.",
        "num_classes": "234",
        "input_tensor_shape": "(1,128,313)",
        "valid_architecture_families": "cnn_scratch, efficientnet_pretrained",
        "personality": "exploratory",
        "experiment_memory": "## Top-5 experiments\n(none yet)",
        "eda_summary": "234 classes, 15320 samples.",
        "allow_transfer_learning": "false",
        "code_skeleton_content": "(skeleton here)",
    }
    sys, usr = eng.render("propose_architecture", slots)
    assert "{task_description}" not in sys + usr
    assert "Multi-label bird classification." in usr
    assert "exploratory" in sys


# -------- missing slot --------

def test_missing_slot_raises_named_error():
    reg = PromptRegistry(SHIPPED_PROMPTS)
    eng = PromptEngine(reg)
    incomplete = {"task_description": "x"}  # all other slots missing
    with pytest.raises(MissingSlotError) as exc:
        eng.render("propose_architecture", incomplete)
    assert exc.value.slot  # slot name populated
    assert exc.value.task == "propose_architecture"


def test_extract_slots_finds_all_braces():
    eng = PromptEngine(PromptRegistry(SHIPPED_PROMPTS))
    template = "Hi {a}, welcome {b}, score {c123}."
    assert eng.extract_slots(template) == {"a", "b", "c123"}


# -------- save_new_version --------

def test_save_new_version_increments_and_does_not_overwrite(tmp_path: Path):
    reg = _make_empty_registry(tmp_path)
    v1 = reg.save_new_version("propose_architecture", "S1", "U1")
    v2 = reg.save_new_version("propose_architecture", "S2", "U2")
    v3 = reg.save_new_version("propose_architecture", "S3", "U3")
    assert (v1, v2, v3) == ("v1", "v2", "v3")

    versions = reg.list_versions("propose_architecture")
    assert versions == ["v1", "v2", "v3"]

    # contents preserved per version
    assert reg.load("propose_architecture", "v1").system == "S1"
    assert reg.load("propose_architecture", "v3").user == "U3"


# -------- set_active --------

def test_set_active_writes_atomically_and_persists(tmp_path: Path):
    reg = _make_empty_registry(tmp_path)
    reg.save_new_version("generate_code", "s1", "u1")
    reg.save_new_version("generate_code", "s2", "u2")
    reg.set_active("generate_code", "v2")
    assert reg.active_version("generate_code") == "v2"

    # registry file is valid YAML and reflects update
    payload = yaml.safe_load((reg.root / "_registry.yaml").read_text())
    assert payload == {"generate_code": "v2"}


def test_set_active_rejects_unknown_version(tmp_path: Path):
    reg = _make_empty_registry(tmp_path)
    reg.save_new_version("generate_code", "s1", "u1")
    with pytest.raises(FileNotFoundError):
        reg.set_active("generate_code", "v9")


def test_load_unknown_version_raises(tmp_path: Path):
    reg = _make_empty_registry(tmp_path)
    reg.save_new_version("analyze_result", "s", "u")
    with pytest.raises(FileNotFoundError):
        reg.load("analyze_result", "v99")


def test_active_version_unknown_task_raises(tmp_path: Path):
    reg = _make_empty_registry(tmp_path)
    with pytest.raises(KeyError):
        reg.active_version("propose_architecture")


def test_render_with_explicit_version(tmp_path: Path):
    reg = _make_empty_registry(tmp_path)
    reg.save_new_version("analyze_result", "old-{x}", "old-{y}")
    reg.save_new_version("analyze_result", "new-{x}", "new-{y}")
    reg.set_active("analyze_result", "v2")
    eng = PromptEngine(reg)

    sys, usr = eng.render("analyze_result", {"x": "1", "y": "2"}, version="v1")
    assert sys == "old-1" and usr == "old-2"

    sys2, usr2 = eng.render("analyze_result", {"x": "1", "y": "2"})  # active
    assert sys2 == "new-1" and usr2 == "new-2"
