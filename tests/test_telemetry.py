"""I-21 acceptance: structured JSONL + human sinks, identical from CLI or UI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.config import load_settings
from lab.core import telemetry as t

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def configured(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"experiments_dir": str(tmp_path / "studies")})
    s = s.model_copy(update={"paths": new_paths})
    t._reset_for_tests()
    t.configure(s, study_id="study_test_xxxx")
    yield s
    t._reset_for_tests()


def _read_jsonl(study_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (study_dir / "run.log.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_event_writes_both_sinks(configured, tmp_path, capsys):
    t.log_event("study_start", task="track_b")
    study_dir = tmp_path / "studies" / "study_test_xxxx"
    assert (study_dir / "run.log.jsonl").exists()
    assert (study_dir / "run.log").exists()

    records = _read_jsonl(study_dir)
    assert records[0]["event"] == "study_start"
    assert records[0]["fields"]["task"] == "track_b"
    assert records[0]["study_id"] == "study_test_xxxx"

    captured = capsys.readouterr()
    assert "study_start" in captured.out


def test_unknown_event_rejected(configured):
    with pytest.raises(ValueError):
        t.log_event("not_a_real_event")  # type: ignore[arg-type]


def test_set_experiment_attaches_id_to_subsequent_events(configured, tmp_path):
    t.set_experiment("exp_0001")
    t.log_event("experiment_start")
    records = _read_jsonl(tmp_path / "studies" / "study_test_xxxx")
    assert records[-1]["experiment_id"] == "exp_0001"


def test_non_json_field_is_coerced_via_repr(configured, tmp_path):
    class Weird:
        def __repr__(self):
            return "<weird>"

    t.log_event("validate", obj=Weird())
    records = _read_jsonl(tmp_path / "studies" / "study_test_xxxx")
    assert records[-1]["fields"]["obj"] == "<weird>"


def test_no_sink_when_study_id_omitted(tmp_path, capsys):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    t._reset_for_tests()
    t.configure(s)  # no study_id
    t.log_event("study_start", x=1)
    captured = capsys.readouterr()
    assert "study_start" in captured.out
    t._reset_for_tests()


def test_log_human_writes_message(configured, tmp_path, capsys):
    t._emit_human("hello world", "info")
    out = capsys.readouterr().out
    assert "hello world" in out
    human = (tmp_path / "studies" / "study_test_xxxx" / "run.log").read_text()
    assert "hello world" in human


def test_identical_event_set_from_two_callers(configured, tmp_path):
    """Repeating the same event from multiple call sites yields the same schema."""
    t.log_event("execute", duration_ms=42, succeeded=True)
    t.log_event("execute", duration_ms=42, succeeded=True)
    records = _read_jsonl(tmp_path / "studies" / "study_test_xxxx")
    assert len(records) == 2
    assert records[0]["fields"] == records[1]["fields"]
    assert set(records[0].keys()) == set(records[1].keys())
