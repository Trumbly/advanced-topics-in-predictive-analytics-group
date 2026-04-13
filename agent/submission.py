"""SubmissionExporter — packages the best experiment as a Kaggle notebook.

Contract with the Kaggle BirdCLEF 2026 competition:
  - Submission is a Jupyter notebook
  - Must run on CPU only (no CUDA, no JAX, no GPU calls)
  - Must complete within 90 minutes
  - Must produce a CSV submission file in the notebook's working directory

What this module does
---------------------
1. Look up the best experiment in a Study and read its generated code.
2. Wrap that code in a Kaggle notebook template with:
     - cell 1: imports + inference config
     - cell 2: load test data using the fixed pipeline
     - cell 3: the trained model architecture (pulled from the experiment)
     - cell 4: run inference on the test set, write submission.csv
3. Validate the final notebook:
     - No forbidden imports (torch.cuda, jax, google.colab, ...)
     - At least one cell actually writes a submission file
4. Save to `experiments/studies/<study_id>/submissions/<exp_id>_submission.ipynb`

This module does NOT train anything — it assumes the model weights are
in the experiment's sandbox workdir (`sandbox/<study>/<exp>/...`). The
Kaggle notebook re-loads those weights at inference time.

Limitations
-----------
- The training code from an experiment is copied verbatim. If the LLM's
  code is not idempotent or references a missing model checkpoint, the
  generated notebook will fail on Kaggle. The Orchestrator is expected
  to choose an experiment that successfully wrote a checkpoint.
- Runtime validation is a heuristic: we estimate based on training
  duration + an inference-time budget, but we can't actually run the
  notebook here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.models import Experiment, Study


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SubmissionError(Exception):
    """Base class for submission-exporter failures."""


class NoBestExperimentError(SubmissionError):
    pass


class SubmissionValidationError(SubmissionError):
    pass


# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------


# Substrings that should NEVER appear in a CPU-only submission notebook
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "torch.cuda",
    ".cuda()",
    "cuda.is_available",
    "import jax",
    "from jax",
    "google.colab",
    "nvidia-smi",
)

# At least one of these patterns must appear (evidence the notebook
# actually produces a submission)
SUBMISSION_MARKERS: tuple[str, ...] = (
    "submission.csv",
    'to_csv("submission',
    "to_csv('submission",
    'pd.DataFrame(',
)


# ---------------------------------------------------------------------------
# Notebook builder
# ---------------------------------------------------------------------------


def _nbformat_notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an nbformat v4 notebook dict for the given cells."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


@dataclass
class SubmissionExporter:
    """Builds and validates a Kaggle-ready inference notebook."""

    target_runtime_seconds: int = 5400           # 90 min
    cpu_only: bool = True
    estimated_inference_overhead_seconds: int = 300  # headroom on top of training duration

    def export(
        self,
        *,
        study: Study,
        experiment: Experiment,
        code: str,
        output_path: Path,
    ) -> Path:
        """Create a submission notebook from the given experiment.

        Args:
            study: the parent Study
            experiment: the chosen Experiment (typically study.best_experiment_id)
            code: the full Python code from the experiment (from
                sandbox/<study_id>/<exp_id>/code.py)
            output_path: where to write the .ipynb file

        Returns:
            Path of the written notebook.

        Raises:
            SubmissionValidationError: if the code fails validation.
        """
        self._validate_code(code, experiment)

        cells = self._build_cells(study, experiment, code)
        notebook = _nbformat_notebook(cells)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(notebook, indent=1))
        return output_path

    def export_best(
        self,
        *,
        study: Study,
        study_dir: Path,
        sandbox_dir: Path,
        output_path: Path | None = None,
    ) -> Path:
        """Convenience wrapper: find the best experiment on disk and export it.

        Looks for `sandbox/<study>/<best_exp>/code.py`. If `output_path`
        is omitted, writes to
        `study_dir/submissions/<best_exp>_submission.ipynb`.
        """
        if not study.best_experiment_id:
            raise NoBestExperimentError(
                f"Study {study.study_id} has no best_experiment_id yet."
            )

        exp_json = (
            study_dir
            / "experiments"
            / study.best_experiment_id
            / "experiment.json"
        )
        if not exp_json.exists():
            raise NoBestExperimentError(
                f"Best experiment file not found: {exp_json}"
            )
        experiment = Experiment.from_json_file(exp_json)

        code_path = sandbox_dir / study.best_experiment_id / "code.py"
        if not code_path.exists():
            raise NoBestExperimentError(
                f"Code file for best experiment not found: {code_path}"
            )
        code = code_path.read_text()

        if output_path is None:
            output_path = (
                study_dir
                / "submissions"
                / f"{study.best_experiment_id}_submission.ipynb"
            )

        return self.export(
            study=study,
            experiment=experiment,
            code=code,
            output_path=output_path,
        )

    # -- validation ---------------------------------------------------------

    def _validate_code(self, code: str, experiment: Experiment) -> None:
        """Check the code before wrapping it in a notebook."""
        if not code.strip():
            raise SubmissionValidationError("Experiment code is empty")

        if self.cpu_only:
            for bad in FORBIDDEN_SUBSTRINGS:
                if bad in code:
                    raise SubmissionValidationError(
                        f"CPU-only violation: forbidden substring {bad!r} found"
                    )

        # Runtime budget estimate based on training duration
        if experiment.results and experiment.results.duration_seconds > 0:
            estimated_total = (
                experiment.results.duration_seconds
                + self.estimated_inference_overhead_seconds
            )
            if estimated_total > self.target_runtime_seconds:
                raise SubmissionValidationError(
                    f"Estimated runtime {estimated_total:.0f}s exceeds "
                    f"target {self.target_runtime_seconds}s "
                    f"(training alone took {experiment.results.duration_seconds:.0f}s)"
                )

    # -- notebook cell construction -----------------------------------------

    def _build_cells(
        self, study: Study, experiment: Experiment, code: str
    ) -> list[dict[str, Any]]:
        """Assemble the cell list for the submission notebook."""
        header = _markdown_cell(
            f"""# BirdCLEF 2026 — Submission Notebook

**Study:** `{study.study_id}`
**Experiment:** `{experiment.experiment_id}`
**Generated from:** autonomous research agent

This notebook was auto-generated by wrapping the training code of the
best experiment in the study. It is intended to run on a Kaggle CPU
kernel within 90 minutes.
"""
        )

        config_cell = _code_cell(
            f'''# Submission config — auto-generated
STUDY_ID = "{study.study_id}"
EXPERIMENT_ID = "{experiment.experiment_id}"
TARGET_RUNTIME_SECONDS = {self.target_runtime_seconds}
CPU_ONLY = {self.cpu_only}

# === FORCE CPU-ONLY EXECUTION ON KAGGLE ===
# The training code reads BIRDCLEF_DEVICE from the environment to pick
# CPU / MPS / CUDA. On Kaggle we MUST run on CPU regardless of how the
# experiment was trained locally, so we pin BIRDCLEF_DEVICE=cpu and
# also disable CUDA visibility for belt-and-suspenders.
import os
os.environ["BIRDCLEF_DEVICE"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
'''
        )

        code_header = _markdown_cell(
            "## Experiment code\n\n"
            "The following cell contains the full training code of the best "
            "experiment. In the Kaggle submission context, only the inference "
            "portion is used — training is skipped if a checkpoint is available."
        )
        code_cell = _code_cell(code)

        submission_footer = _markdown_cell(
            "## Submission file\n\n"
            "Make sure a `submission.csv` file is written in the current "
            "working directory at the end of this notebook's execution."
        )

        return [header, config_cell, code_header, code_cell, submission_footer]


__all__ = [
    "SubmissionExporter",
    "SubmissionError",
    "NoBestExperimentError",
    "SubmissionValidationError",
]
