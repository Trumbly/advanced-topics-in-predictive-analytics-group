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


def test_track_b_first_present_picks_known_column():
    from lab.tasks.track_b_birdclef import _first_present
    assert _first_present(["sample_id", "class_id"], ("filename", "sample_id")) == "sample_id"
    assert _first_present(["filename", "labels"], ("filename", "sample_id")) == "filename"
    assert _first_present(["other"], ("filename", "sample_id")) is None


def _try_run_loader_on(tmp_path, csv_text: str) -> int:
    """Run load_audio_dataset against a synthetic labels.csv and return
    the detected number of classes. Skips quietly when torch isn't
    importable in the test env."""
    import os
    pytest_obj = __import__("pytest")
    try:
        import numpy as np
        import torch  # noqa: F401
    except ImportError:
        pytest_obj.skip("torch / numpy not installed in test env")

    (tmp_path / "labels.csv").write_text(csv_text)
    spectrograms = tmp_path / "spectrograms"
    spectrograms.mkdir()
    # Write three tiny spectrogram files so the DataLoader can iterate.
    for name in ("a.npy", "b.npy", "c.npy"):
        np.save(spectrograms / name, np.zeros((1, 4, 8), dtype=np.float32))

    os.environ["AGENT_PROCESSED_DIR"] = str(tmp_path)
    try:
        from lab.tasks.track_b_birdclef import load_audio_dataset
        _, _, n_classes = load_audio_dataset(batch_size=1, num_workers=0, val_fraction=0.34)
        return n_classes
    finally:
        os.environ.pop("AGENT_PROCESSED_DIR", None)


def test_track_b_loader_accepts_filename_labels_schema(tmp_path):
    """The original schema: columns 'filename' + pipe-separated 'labels'."""
    n = _try_run_loader_on(tmp_path,
        "filename,labels\na.npy,c1|c2\nb.npy,c1\nc.npy,c3\n",
    )
    assert n == 3


def test_track_b_loader_accepts_sample_id_class_id_schema(tmp_path):
    """The team's actual schema: columns 'sample_id' + 'class_id'.
    Bare sample id (no .npy suffix) — loader appends it."""
    n = _try_run_loader_on(tmp_path,
        "sample_id,class_id\na,7\nb,7\nc,3\n",
    )
    assert n == 2  # classes 7 and 3


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
