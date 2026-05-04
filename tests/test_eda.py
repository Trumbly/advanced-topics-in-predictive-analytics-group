"""I-12 acceptance: EDA step produces compact markdown."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.config import load_settings
from lab.core.models import DatasetProfile
from lab.tasks.base import TaskAdapter
from lab.tasks.eda import run_eda

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeAdapter(TaskAdapter):
    name = "fake"
    kind = "audio_multilabel"
    primary_metric = "roc_auc_macro"

    def __init__(self, profile: DatasetProfile):
        self._profile = profile

    def profile(self) -> DatasetProfile:
        return self._profile

    def prompt_slots(self) -> dict[str, str]:
        return {}

    def model_block_signature(self) -> tuple[str, str]:
        return ("build_model", "num_classes")

    def spawn_triggering_calls(self):
        return ()

    def build_submission(self, code, experiment_id, out_dir):
        return Path(out_dir) / "submission.ipynb"


@pytest.fixture
def settings():
    return load_settings("track_b", repo_root=REPO_ROOT)


def test_imbalance_ratio_two_classes(settings):
    profile = DatasetProfile(
        num_classes=2,
        num_train=55,
        input_tensor_shape=(1, 128, 313),
        class_imbalance={"a": 5, "b": 50},
    )
    report = run_eda(_FakeAdapter(profile), settings)
    assert report.imbalance_ratio == pytest.approx(10.0)
    assert "5" in report.markdown and "50" in report.markdown


def test_long_tail_warning_when_class_under_threshold(settings):
    profile = DatasetProfile(
        num_classes=3,
        num_train=120,
        input_tensor_shape=(1, 128, 313),
        class_imbalance={"a": 50, "b": 60, "c": 10},  # c is long-tail
    )
    report = run_eda(_FakeAdapter(profile), settings)
    assert any("long-tail" in n.lower() or "<25" in n for n in report.notes) or any(
        "samples" in n for n in report.notes
    )


def test_markdown_under_4kb(settings):
    counts = {f"class_{i}": (i + 1) * 5 for i in range(234)}
    profile = DatasetProfile(
        num_classes=234,
        num_train=sum(counts.values()),
        input_tensor_shape=(1, 128, 313),
        class_imbalance=counts,
    )
    report = run_eda(_FakeAdapter(profile), settings)
    assert len(report.markdown.encode("utf-8")) <= 4096


def test_no_imbalance_data_falls_back_gracefully(settings):
    profile = DatasetProfile(
        num_classes=234,
        num_train=15000,
        input_tensor_shape=(1, 128, 313),
        class_imbalance=None,
    )
    report = run_eda(_FakeAdapter(profile), settings)
    assert report.num_classes == 234
    assert report.imbalance_ratio == pytest.approx(1.0)


def test_severe_imbalance_emits_warning(settings):
    counts = {"a": 10, "b": 200}
    profile = DatasetProfile(
        num_classes=2,
        num_train=210,
        input_tensor_shape=(1, 128, 313),
        class_imbalance=counts,
    )
    report = run_eda(_FakeAdapter(profile), settings)
    assert any("imbalance is severe" in n for n in report.notes)
