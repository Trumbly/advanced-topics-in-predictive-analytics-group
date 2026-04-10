"""Unit tests for `agent.metrics.MetricsCollector`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.metrics import MetricsCollector, MetricsParseError, ScriptReportedError
from agent.models import TrainingResults


def _write_results(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


class TestParseResults:
    def test_minimal_valid(self, tmp_path: Path) -> None:
        path = _write_results(
            tmp_path / "results.json",
            {
                "metrics": {"roc_auc_macro": 0.72, "loss": 0.38},
                "duration_seconds": 145.3,
            },
        )
        collector = MetricsCollector()
        results = collector.parse_results(path)
        assert isinstance(results, TrainingResults)
        assert results.metrics["roc_auc_macro"] == 0.72
        assert results.duration_seconds == 145.3

    def test_with_training_curves(self, tmp_path: Path) -> None:
        path = _write_results(
            tmp_path / "results.json",
            {
                "metrics": {"roc_auc_macro": 0.8, "loss": 0.2},
                "training_curves": {
                    "loss": [1.0, 0.5, 0.2],
                    "roc_auc_macro": [0.4, 0.6, 0.8],
                },
                "duration_seconds": 100.0,
            },
        )
        results = MetricsCollector().parse_results(path)
        assert results.training_curves["loss"] == [1.0, 0.5, 0.2]
        assert results.training_curves["roc_auc_macro"] == [0.4, 0.6, 0.8]

    def test_extra_metrics_preserved(self, tmp_path: Path) -> None:
        path = _write_results(
            tmp_path / "results.json",
            {
                "metrics": {
                    "roc_auc_macro": 0.7,
                    "loss": 0.3,
                    "precision": 0.65,
                    "recall": 0.72,
                }
            },
        )
        results = MetricsCollector().parse_results(path)
        assert "precision" in results.metrics
        assert results.metrics["precision"] == 0.65

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MetricsParseError, match="not found"):
            MetricsCollector().parse_results(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json}")
        with pytest.raises(MetricsParseError, match="invalid JSON"):
            MetricsCollector().parse_results(path)

    def test_missing_required_metric_raises(self, tmp_path: Path) -> None:
        path = _write_results(tmp_path / "results.json", {"metrics": {"loss": 0.5}})
        with pytest.raises(MetricsParseError, match="missing required metrics"):
            MetricsCollector().parse_results(path)

    def test_non_object_root_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(MetricsParseError, match="expected a JSON object"):
            MetricsCollector().parse_results(path)

    def test_unparseable_metric_skipped(self, tmp_path: Path) -> None:
        path = _write_results(
            tmp_path / "results.json",
            {
                "metrics": {
                    "roc_auc_macro": 0.7,
                    "loss": 0.3,
                    "bogus": "not_a_number",
                }
            },
        )
        results = MetricsCollector().parse_results(path)
        assert "bogus" not in results.metrics

    def test_script_error_key_surfaces_cleanly(self, tmp_path: Path) -> None:
        """Regression: when LLM code hits its own except branch and writes
        `{"error": "..."}`, we must surface the actual error — not report
        a confusing 'missing required metrics' message."""
        path = _write_results(
            tmp_path / "results.json",
            {"error": "RuntimeError: shape mismatch (32,1,128,313) vs (32,3,224,224)"},
        )
        with pytest.raises(ScriptReportedError) as exc_info:
            MetricsCollector().parse_results(path)
        assert exc_info.value.script_error.startswith("RuntimeError")
        # ScriptReportedError IS a MetricsParseError subclass — callers can
        # still catch MetricsParseError for a uniform fallback path.
        assert isinstance(exc_info.value, MetricsParseError)

    def test_empty_metrics_with_error_key_reports_error(
        self, tmp_path: Path
    ) -> None:
        """Reality check: the failing runs in the production logs wrote
        `{"metrics": {}, "error": "..."}`. The error takes precedence."""
        path = _write_results(
            tmp_path / "results.json",
            {"metrics": {}, "error": "NameError: CnnSmallV1 is not defined"},
        )
        with pytest.raises(ScriptReportedError, match="NameError"):
            MetricsCollector().parse_results(path)


class TestComputeDelta:
    def test_delta_against_none_is_empty(self) -> None:
        current = TrainingResults(metrics={"roc_auc_macro": 0.7})
        assert MetricsCollector().compute_delta(current, None) == {}

    def test_delta_against_previous(self) -> None:
        previous = TrainingResults(metrics={"roc_auc_macro": 0.5, "loss": 0.6})
        current = TrainingResults(metrics={"roc_auc_macro": 0.7, "loss": 0.4})
        delta = MetricsCollector().compute_delta(current, previous)
        assert delta["roc_auc_macro"] == pytest.approx(0.2)
        assert delta["loss"] == pytest.approx(-0.2)

    def test_delta_skips_missing_previous(self) -> None:
        previous = TrainingResults(metrics={"roc_auc_macro": 0.5})
        current = TrainingResults(
            metrics={"roc_auc_macro": 0.7, "new_metric": 0.9}
        )
        delta = MetricsCollector().compute_delta(current, previous)
        assert "new_metric" not in delta
        assert "roc_auc_macro" in delta
