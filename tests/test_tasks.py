from __future__ import annotations

from pathlib import Path

from lab.tasks.registry import get_task_adapter, list_available_tasks


REPO = Path(__file__).resolve().parent.parent


def test_list_available_tasks_includes_both_tracks():
    from lab.config import load_settings
    tasks = list_available_tasks(load_settings(repo_root=REPO))
    names = {t["name"] for t in tasks}
    assert "track_a" in names
    assert "track_b" in names


def test_track_a_adapter_loads():
    adapter = get_task_adapter(task_name="track_a")
    assert adapter.name == "track_a"
    assert adapter.primary_metric == "f1_binary"
    # Prompt slot values must include the task description.
    slots = adapter.prompt_slot_values()
    assert "task_description" in slots
    assert "code_skeleton" in slots


def test_track_b_adapter_loads():
    adapter = get_task_adapter(task_name="track_b")
    assert adapter.name == "track_b"
    assert adapter.primary_metric == "f1_macro"


def test_track_a_spawn_triggers_include_loader():
    adapter = get_task_adapter(task_name="track_a")
    assert "load_text_dataset" in tuple(adapter.spawn_triggering_calls())


def test_env_vars_are_prefixed():
    adapter = get_task_adapter(task_name="track_a")
    env = adapter.env_vars(adapter.settings)
    for k in env:
        assert k.startswith("AGENT_"), k
