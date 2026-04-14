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


def test_load_profile_tolerates_legacy_cache_without_task_name(tmp_path):
    """Old dataset_profile.json files don't have task_name/kind fields.
    The adapter must fill them from itself instead of crashing."""
    import json
    from lab.tasks.track_b_birdclef import BirdclefAdapter
    from lab.config import load_settings

    settings = load_settings(task="track_b")
    # Point processed_dir at an isolated temp tree.
    settings.task_config["data"]["processed_dir"] = str(tmp_path)
    (tmp_path / "dataset_profile.json").write_text(json.dumps({
        "num_classes": 206,
        "num_train_samples": 12345,
        # Crucially: NO task_name, NO kind — that's the whole point.
    }))

    adapter = BirdclefAdapter(settings)
    profile = adapter.load_profile()
    assert profile.task_name == "track_b"
    assert profile.kind == "audio_multilabel"
    assert profile.num_classes == 206
