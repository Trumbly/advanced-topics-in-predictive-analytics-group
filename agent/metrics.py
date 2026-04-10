"""Parse the `results.json` files written by LLM-generated training scripts.

Contract with LLM-generated code
--------------------------------
The LLM is instructed (in `config/prompts/generate_code.yaml`) to write a
`results.json` file with at minimum:

    {
      "metrics": {
        "roc_auc_macro": 0.72,
        "loss": 0.38
      },
      "training_curves": {
        "loss": [...],
        "roc_auc_macro": [...]
      },
      "duration_seconds": 145.3
    }

Additional keys are tolerated and silently passed through. If the file is
missing a required key, `parse_results` raises a clear error that the
orchestrator turns into a TaskError so the LLM gets useful feedback on
its next iteration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.models import TrainingResults


class MetricsParseError(ValueError):
    """Raised when a results.json file cannot be parsed into TrainingResults."""


@dataclass
class MetricsCollector:
    """Parses results.json and optionally computes deltas vs. a baseline."""

    required_metrics: tuple[str, ...] = ("roc_auc_macro", "loss")

    # -- parsing ------------------------------------------------------------

    def parse_results(self, results_json_path: Path) -> TrainingResults:
        """Load and validate a results.json file.

        Missing optional fields (`training_curves`, `peak_ram_mb`,
        `val_predictions_path`) default to empty/None. Missing REQUIRED
        metrics (`roc_auc_macro`, `loss` by default) raise MetricsParseError.
        """
        path = Path(results_json_path)
        if not path.exists():
            raise MetricsParseError(f"results.json not found at {path}")

        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise MetricsParseError(f"{path}: invalid JSON ({e})") from e

        if not isinstance(raw, dict):
            raise MetricsParseError(
                f"{path}: expected a JSON object, got {type(raw).__name__}"
            )

        metrics_raw = raw.get("metrics", {})
        if not isinstance(metrics_raw, dict):
            raise MetricsParseError(
                f"{path}: 'metrics' must be an object, got {type(metrics_raw).__name__}"
            )

        # Coerce metric values to float so downstream sorting is consistent
        metrics: dict[str, float] = {}
        for key, value in metrics_raw.items():
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue  # skip unparseable entries silently

        missing = [m for m in self.required_metrics if m not in metrics]
        if missing:
            raise MetricsParseError(
                f"{path}: missing required metrics {missing}. "
                f"Got: {sorted(metrics.keys())}"
            )

        curves = raw.get("training_curves", {}) or {}
        parsed_curves: dict[str, list[float]] = {}
        if isinstance(curves, dict):
            for key, values in curves.items():
                if isinstance(values, list):
                    try:
                        parsed_curves[key] = [float(v) for v in values]
                    except (TypeError, ValueError):
                        continue

        duration = _safe_float(raw.get("duration_seconds"), default=0.0)
        peak_ram = (
            _safe_float(raw.get("peak_ram_mb"), default=None)
            if "peak_ram_mb" in raw
            else None
        )
        val_pred_path = raw.get("val_predictions_path")
        val_pred_path_obj = Path(val_pred_path) if val_pred_path else None

        return TrainingResults(
            metrics=metrics,
            training_curves=parsed_curves,
            duration_seconds=duration,
            peak_ram_mb=peak_ram,
            val_predictions_path=val_pred_path_obj,
        )

    # -- deltas -------------------------------------------------------------

    def compute_delta(
        self,
        current: TrainingResults,
        previous: TrainingResults | None,
    ) -> dict[str, float]:
        """Return a per-metric delta (current - previous).

        If `previous` is None or lacks a metric present in `current`, the
        delta for that metric is omitted. This makes the result safe to
        feed directly into the LLM prompt without worrying about cold-starts.
        """
        if previous is None:
            return {}
        delta: dict[str, float] = {}
        for metric, value in current.metrics.items():
            if metric in previous.metrics:
                delta[metric] = value - previous.metrics[metric]
        return delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, *, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["MetricsCollector", "MetricsParseError"]
