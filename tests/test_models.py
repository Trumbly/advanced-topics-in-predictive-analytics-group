"""I-02 acceptance: typed pydantic models + Study save/load round-trip."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.core.loaders import list_studies, load_all, load_many
from lab.core.models import (
    Experiment,
    Proposal,
    Study,
    StudyNotFoundError,
    Task,
    TaskError,
    Verdict,
    new_study_id,
)


# -------- Study ID format --------

def test_new_study_id_pattern():
    sid = new_study_id()
    assert re.fullmatch(r"study_\d{8}_\d{6}_[a-z0-9]{4}", sid), sid


def test_new_study_id_uses_provided_clock():
    fixed = datetime(2026, 5, 4, 13, 12, 11)
    sid = new_study_id(now=fixed)
    assert sid.startswith("study_20260504_131211_")


# -------- Round-trip --------

def _make_study() -> Study:
    proposal = Proposal(
        architecture_name="EffNetB0-transfer",
        family="efficientnet_pretrained",
        lr=3e-4,
        lr_schedule="cosine",
        epochs=4,
    )
    err = TaskError(
        error_type="ShapeMismatch",
        message="expected (1,128,313) got (3,128,313)",
        traceback="...",
    )
    task = Task(
        name="execute",
        status="FAILED",
        input={"epochs": 4},
        output={},
        error=err,
    )
    verdict = Verdict(
        verdict="discard",
        score=0.12,
        rationale="Shape error blocks training.",
    )
    exp = Experiment(
        id="exp_0001",
        index=0,
        status="FAILED",
        proposal=proposal,
        primary_metric="roc_auc_macro",
        primary_score=None,
        tasks=[task],
        verdict=verdict,
    )
    return Study(
        id=new_study_id(),
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        prompt_template_paths={"propose_architecture": Path("config/prompts/propose_architecture/v1.yaml")},
        experiments=[exp],
        best_experiment_id=None,
        best_score=None,
        created_at=datetime(2026, 5, 4, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_study_save_and_load_round_trip(tmp_path: Path):
    study = _make_study()
    study.save(tmp_path)

    out_file = tmp_path / study.id / "study.json"
    assert out_file.exists()

    loaded = Study.load(tmp_path, study.id)
    assert loaded == study
    # Nested objects preserved
    assert loaded.experiments[0].verdict == study.experiments[0].verdict
    assert loaded.experiments[0].tasks[0].error == study.experiments[0].tasks[0].error


def test_load_unknown_raises_study_not_found(tmp_path: Path):
    with pytest.raises(StudyNotFoundError):
        Study.load(tmp_path, "study_does_not_exist_0000")


def test_forward_compat_extra_keys_in_json_are_ignored(tmp_path: Path):
    study = _make_study()
    study.save(tmp_path)
    path = tmp_path / study.id / "study.json"

    payload = json.loads(path.read_text())
    payload["future_field_we_added_later"] = {"answer": 42}
    payload["experiments"][0]["another_unknown"] = "ignored"
    path.write_text(json.dumps(payload))

    loaded = Study.load(tmp_path, study.id)
    assert loaded.id == study.id
    assert loaded.experiments[0].id == study.experiments[0].id


# -------- bulk loaders --------

def test_list_studies_only_returns_dirs_with_study_json(tmp_path: Path):
    s1 = _make_study()
    s1.save(tmp_path)
    # noise: empty dir + dir with wrong file
    (tmp_path / "study_garbage").mkdir()
    junk = tmp_path / "study_partial"
    junk.mkdir()
    (junk / "not_study.json").write_text("{}")

    found = list_studies(tmp_path)
    assert found == [s1.id]


def test_load_many_skips_missing(tmp_path: Path):
    s1 = _make_study()
    s1.save(tmp_path)
    loaded = load_many(tmp_path, [s1.id, "study_missing"])
    assert len(loaded) == 1 and loaded[0].id == s1.id


def test_load_all_returns_every_saved_study(tmp_path: Path):
    s1 = _make_study()
    s1.save(tmp_path)
    s2 = _make_study()
    s2.save(tmp_path)
    loaded = load_all(tmp_path)
    assert {s.id for s in loaded} == {s1.id, s2.id}


# -------- field validation --------

def test_proposal_rejects_invalid_lr_schedule():
    with pytest.raises(Exception):
        Proposal(
            architecture_name="X",
            family="cnn_scratch",
            lr=1e-3,
            lr_schedule="bogus",  # type: ignore[arg-type]
            epochs=1,
        )


def test_verdict_rationale_capped_at_500_chars():
    with pytest.raises(Exception):
        Verdict(verdict="keep", score=0.5, rationale="x" * 501)


def test_verdict_kind_must_be_known():
    with pytest.raises(Exception):
        Verdict(verdict="rogue", score=0.5, rationale="ok")  # type: ignore[arg-type]
