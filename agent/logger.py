"""ExperimentLogger — persistent logs for studies, experiments, and tasks.

Every artifact the agent produces is written here so we can reconstruct
what happened for the report, for debugging, and for handing successful
runs to the SubmissionExporter.

Layout
------
    experiments/studies/<study_id>/
        study.json
        study.md
        experiments/
            <experiment_id>/
                experiment.json
                experiment.md
                tasks/
                    <task_id>.json
                    <task_id>.md
        memory.json   (written by ExperimentMemory, not this module)
        memory.md

This is a stateless writer — callers own the Study, Experiment, and Task
objects and hand them to the logger whenever something changes. That
makes the logger trivially testable and lets the orchestrator decide when
to flush.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.models import Experiment, Study, Task


@dataclass
class ExperimentLogger:
    """Writes JSON + Markdown artifacts for a Study, its Experiments, and Tasks."""

    study_dir: Path

    def __post_init__(self) -> None:
        self.study_dir = Path(self.study_dir)
        self.study_dir.mkdir(parents=True, exist_ok=True)

    # -- Study --------------------------------------------------------------

    @property
    def study_json_path(self) -> Path:
        return self.study_dir / "study.json"

    @property
    def study_markdown_path(self) -> Path:
        return self.study_dir / "study.md"

    def write_study(self, study: Study) -> None:
        study.to_json_file(self.study_json_path)
        self.study_markdown_path.write_text(_render_study(study))

    # -- Experiment ---------------------------------------------------------

    def experiment_dir(self, experiment_id: str) -> Path:
        path = self.study_dir / "experiments" / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_experiment(self, experiment: Experiment) -> None:
        exp_dir = self.experiment_dir(experiment.experiment_id)
        experiment.to_json_file(exp_dir / "experiment.json")
        (exp_dir / "experiment.md").write_text(_render_experiment(experiment))

    # -- Task ---------------------------------------------------------------

    def task_dir(self, experiment_id: str) -> Path:
        path = self.experiment_dir(experiment_id) / "tasks"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_task(self, task: Task) -> None:
        tdir = self.task_dir(task.experiment_id)
        task.to_json_file(tdir / f"{task.task_id}.json")
        (tdir / f"{task.task_id}.md").write_text(_render_task(task))


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------


def _render_study(study: Study) -> str:
    lines = [
        f"# Study: {study.name}",
        "",
        f"- **ID:** {study.study_id}",
        f"- **Status:** {study.status}",
        f"- **Mode:** {study.mode}",
        f"- **Hypothesis:** {study.hypothesis}",
        f"- **Created:** {study.created_at}",
        f"- **Updated:** {study.updated_at}",
        "",
        "## Compute Budget",
        f"- max_experiments: {study.compute_budget.max_experiments}",
        f"- max_wallclock_minutes: {study.compute_budget.max_wallclock_minutes}",
        f"- max_experiment_seconds: {study.compute_budget.max_experiment_seconds}",
        f"- max_epochs_per_run: {study.compute_budget.max_epochs_per_run}",
        "",
        "## Configuration",
        f"- pipeline: `{study.pipeline_config_path}`",
        f"- dataset profile: `{study.dataset_profile_path}`",
        f"- model registry: `{study.model_registry_path}`",
        "",
        "## Progress",
        f"- experiments run: {len(study.experiment_ids)}",
        f"- best experiment: {study.best_experiment_id or '(none)'}",
        f"- best score: {study.best_score if study.best_score is not None else '(none)'}",
        f"- submissions: {len(study.submissions)}",
    ]
    return "\n".join(lines) + "\n"


def _render_experiment(experiment: Experiment) -> str:
    lines = [
        f"# Experiment {experiment.experiment_id}",
        "",
        f"- **Study:** {experiment.study_id}",
        f"- **Status:** {experiment.status}",
        f"- **LLM:** {experiment.llm_model}",
        f"- **Created:** {experiment.created_at}",
    ]
    if experiment.started_at:
        lines.append(f"- **Started:** {experiment.started_at}")
    if experiment.completed_at:
        lines.append(f"- **Completed:** {experiment.completed_at}")

    if experiment.config:
        lines += [
            "",
            "## Model Config",
            f"- architecture: `{experiment.config.architecture}`",
        ]
        if experiment.config.pretrained_model:
            lines.append(f"- pretrained: `{experiment.config.pretrained_model}`")
        if experiment.config.hyperparams:
            lines.append("- hyperparams:")
            for k, v in experiment.config.hyperparams.items():
                lines.append(f"  - {k}: {v}")
        if experiment.config.augmentation:
            lines.append("- augmentation:")
            for k, v in experiment.config.augmentation.items():
                lines.append(f"  - {k}: {v}")

    if experiment.results:
        lines += ["", "## Results"]
        for k, v in sorted(experiment.results.metrics.items()):
            lines.append(f"- {k}: {v:.4f}")
        lines.append(f"- duration: {experiment.results.duration_seconds:.1f}s")
        if experiment.results.peak_ram_mb:
            lines.append(f"- peak RAM: {experiment.results.peak_ram_mb:.0f} MB")
        if experiment.results.training_curves:
            lines.append("- training curves:")
            for metric, values in experiment.results.training_curves.items():
                lines.append(f"  - {metric} (len={len(values)}): {_fmt_curve(values)}")

    if experiment.task_ids:
        lines += ["", "## Tasks", *[f"- {tid}" for tid in experiment.task_ids]]

    return "\n".join(lines) + "\n"


def _render_task(task: Task) -> str:
    lines = [
        f"# Task {task.task_id}",
        "",
        f"- **Experiment:** {task.experiment_id}",
        f"- **Type:** {task.task_type}",
        f"- **Name:** {task.task_name}",
        f"- **Status:** {task.status}",
    ]
    if task.started_at:
        lines.append(f"- **Started:** {task.started_at}")
    if task.completed_at:
        lines.append(f"- **Completed:** {task.completed_at}")

    if task.prompt_used:
        lines += ["", "## Prompt Used", "```", task.prompt_used, "```"]
    if task.llm_response:
        lines += ["", "## LLM Response", "```", task.llm_response, "```"]
    if task.code_used:
        lines += ["", "## Code Used", "```python", task.code_used, "```"]
    if task.output:
        lines += ["", "## Output"]
        for k, v in task.output.items():
            lines.append(f"- **{k}:** {v}")
    if task.error:
        lines += [
            "",
            "## Error",
            f"- **type:** {task.error.error_type}",
            f"- **message:** {task.error.message}",
        ]
        if task.error.traceback:
            lines += ["", "```", task.error.traceback, "```"]

    return "\n".join(lines) + "\n"


def _fmt_curve(values: list[float]) -> str:
    """Compact curve summary: first, last, min, max."""
    if not values:
        return "(empty)"
    return (
        f"first={values[0]:.4f}, last={values[-1]:.4f}, "
        f"min={min(values):.4f}, max={max(values):.4f}"
    )


__all__ = ["ExperimentLogger"]
