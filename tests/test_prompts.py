from __future__ import annotations

from pathlib import Path

import pytest

from lab.prompts.engine import MissingSlotError, PromptEngine, extract_slots, fill
from lab.prompts.registry import PromptRegistry


REPO = Path(__file__).resolve().parent.parent


def test_extract_slots_finds_single_braces():
    assert extract_slots("hi {name}, {age}") == ["age", "name"]


def test_fill_raises_on_missing_slot():
    with pytest.raises(MissingSlotError):
        fill("hello {name}", {})


def test_fill_substitutes_values():
    assert fill("{a} + {b}", {"a": 1, "b": "two"}) == "1 + two"


def test_registry_can_list_and_resolve_tasks():
    reg = PromptRegistry(REPO / "config" / "prompts")
    tasks = reg.list_tasks()
    assert "propose_architecture" in tasks
    assert "generate_code" in tasks
    path = reg.resolve_active_path("propose_architecture")
    assert path.exists()


def test_prompt_engine_renders_v1(tmp_path):
    reg = PromptRegistry(REPO / "config" / "prompts")
    engine = PromptEngine(reg)
    # Pull real v1 slot list from the file so the test is self-updating.
    content = reg.load_prompt("analyze_result")
    slots_needed = set(extract_slots(content.get("system", "")) + extract_slots(content.get("template", "")))
    slots = {k: f"<{k}>" for k in slots_needed}
    system, user = engine.render("analyze_result", slots)
    assert "<task_description>" in system or "<task_description>" in user


def test_cli_parses_prompt_overrides():
    from lab.cli import _parse_prompt_overrides
    assert _parse_prompt_overrides(["propose_architecture=v2", "generate_code=v1"]) == {
        "propose_architecture": "v2",
        "generate_code": "v1",
    }
    assert _parse_prompt_overrides(None) == {}
    assert _parse_prompt_overrides([]) == {}


def test_cli_rejects_malformed_prompt_overrides():
    from lab.cli import _parse_prompt_overrides
    with pytest.raises(ValueError):
        _parse_prompt_overrides(["no_equals_here"])
    with pytest.raises(ValueError):
        _parse_prompt_overrides(["=v1"])
    with pytest.raises(ValueError):
        _parse_prompt_overrides(["task="])
