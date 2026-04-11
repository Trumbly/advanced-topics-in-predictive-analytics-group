"""ReportGenerator — writes a post-study Markdown report with figures.

Called at the end of `Orchestrator.run()` when the study has at least one
successful experiment, and also via the CLI `report` command to generate a
report for a completed study after the fact.

What it produces
----------------
    experiments/studies/<study_id>/report/
        report.md                      — LLM-authored Markdown report
        figures/
            score_progression.png      — per-experiment score, success/fail color coding
            failure_breakdown.png      — bar chart of error types
            best_learning_curve.png    — optional, only when the best run has >1 epoch

The figures are always rendered (even without a working LLM). If the LLM
call fails, a minimal placeholder Markdown is written instead, with the
figures still intact so the user can build their own report on top.

Design notes
------------
- matplotlib is imported lazily inside the figure methods so `import
  agent.report` stays fast and works in environments without matplotlib.
- All methods are best-effort. Exceptions during figure rendering or the
  LLM call are logged and swallowed — a broken report must never bring
  down an otherwise-successful study.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.llm_client import LLMClient, LLMError
from agent.memory import ExperimentMemory
from agent.models import Experiment, ExperimentStatus, Study, Task, TaskError, TaskStatus
from agent.prompt_engine import PromptEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


@dataclass
class ExperimentSummary:
    """Per-experiment summary used to populate the report and figures."""

    experiment_id: str
    index: int
    status: str
    architecture: str
    score: float | None
    loss: float | None
    duration_seconds: float | None
    error_type: str | None
    error_message: str | None
    training_curves: dict[str, list[float]] = field(default_factory=dict)
    hyperparams: dict[str, Any] = field(default_factory=dict)


def _summarize_experiment(index: int, experiment: Experiment) -> ExperimentSummary:
    """Flatten an Experiment into the fields the report needs."""
    arch = "(unknown)"
    hyperparams: dict[str, Any] = {}
    if experiment.config:
        arch = experiment.config.architecture
        hyperparams = experiment.config.hyperparams

    score: float | None = None
    loss: float | None = None
    duration: float | None = None
    curves: dict[str, list[float]] = {}
    if experiment.results:
        score = experiment.results.metrics.get("roc_auc_macro")
        loss = experiment.results.metrics.get("loss")
        duration = experiment.results.duration_seconds
        curves = dict(experiment.results.training_curves or {})

    return ExperimentSummary(
        experiment_id=experiment.experiment_id,
        index=index,
        status=experiment.status if isinstance(experiment.status, str) else experiment.status.value,
        architecture=arch,
        score=score,
        loss=loss,
        duration_seconds=duration,
        error_type=None,  # filled by caller from task logs
        error_message=None,
        training_curves=curves,
        hyperparams=hyperparams,
    )


def _extract_first_failed_task_error(exp_dir: Path) -> tuple[str | None, str | None]:
    """Walk the task logs and return the first failed task's error info."""
    tasks_dir = exp_dir / "tasks"
    if not tasks_dir.exists():
        return None, None
    import json as _json
    for tj in sorted(tasks_dir.glob("*.json")):
        try:
            data = _json.loads(tj.read_text())
        except Exception:  # noqa: BLE001
            continue
        if data.get("status") != "failed":
            continue
        err = data.get("error") or {}
        return err.get("error_type"), err.get("message")
    return None, None


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


@dataclass
class ReportGenerator:
    """Produces a Markdown + figures report for a completed Study."""

    study: Study
    memory: ExperimentMemory
    llm_client: LLMClient
    prompt_engine: PromptEngine
    study_dir: Path
    prompt_path: Path = Path("config/prompts/report.yaml")

    # -- public API ---------------------------------------------------------

    def generate(self) -> Path:
        """Gather data, render figures, call the LLM, write report.md.

        Returns the path to the written report. Always returns a path —
        even on failure, a minimal placeholder is written so the user
        sees SOMETHING rather than nothing.
        """
        report_dir = self.study_dir / "report"
        figures_dir = report_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        summaries = self._collect_summaries()
        available_figures = self._render_figures(summaries, figures_dir)

        try:
            markdown = self._ask_llm_for_report(summaries, available_figures)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Report LLM call failed: %s", exc)
            markdown = self._placeholder_markdown(summaries, available_figures)

        report_path = report_dir / "report.md"
        report_path.write_text(markdown)
        logger.info("Study report written to %s", report_path)
        return report_path

    # -- data gathering -----------------------------------------------------

    def _collect_summaries(self) -> list[ExperimentSummary]:
        """Load every experiment's Experiment object + first failing task error."""
        summaries: list[ExperimentSummary] = []
        experiments_root = self.study_dir / "experiments"
        for idx, exp_id in enumerate(self.study.experiment_ids, start=1):
            exp_file = experiments_root / exp_id / "experiment.json"
            if not exp_file.exists():
                continue
            try:
                experiment = Experiment.from_json_file(exp_file)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s: %s", exp_file, exc)
                continue
            summary = _summarize_experiment(idx, experiment)
            if summary.status != ExperimentStatus.COMPLETED.value:
                err_type, err_msg = _extract_first_failed_task_error(
                    experiments_root / exp_id
                )
                summary.error_type = err_type
                summary.error_message = err_msg
            summaries.append(summary)
        return summaries

    # -- figures ------------------------------------------------------------

    def _render_figures(
        self, summaries: list[ExperimentSummary], figures_dir: Path
    ) -> list[str]:
        """Render the three available figures. Returns list of generated file names."""
        available: list[str] = []
        try:
            self._render_score_progression(summaries, figures_dir / "score_progression.png")
            available.append("score_progression.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning("score_progression figure failed: %s", exc)

        try:
            if self._render_failure_breakdown(
                summaries, figures_dir / "failure_breakdown.png"
            ):
                available.append("failure_breakdown.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning("failure_breakdown figure failed: %s", exc)

        try:
            best = self._best_successful(summaries)
            if best and self._render_learning_curve(
                best, figures_dir / "best_learning_curve.png"
            ):
                available.append("best_learning_curve.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning("best_learning_curve figure failed: %s", exc)

        return available

    @staticmethod
    def _best_successful(summaries: list[ExperimentSummary]) -> ExperimentSummary | None:
        successes = [
            s
            for s in summaries
            if s.status == ExperimentStatus.COMPLETED.value and s.score is not None
        ]
        if not successes:
            return None
        return max(successes, key=lambda s: s.score or float("-inf"))

    def _render_score_progression(
        self, summaries: list[ExperimentSummary], path: Path
    ) -> None:
        """Scatter + line of scores across experiments, color-coded by status."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        indices = [s.index for s in summaries]
        scores = [s.score if s.score is not None else 0.0 for s in summaries]
        colors = [
            "#2ecc71"
            if s.status == ExperimentStatus.COMPLETED.value and s.score is not None
            else "#e74c3c"
            for s in summaries
        ]

        fig, ax = plt.subplots(figsize=(9, 5))

        # Line through successful experiments only
        successes = [
            (s.index, s.score)
            for s in summaries
            if s.status == ExperimentStatus.COMPLETED.value and s.score is not None
        ]
        if len(successes) >= 2:
            xs = [x for x, _ in successes]
            ys = [y for _, y in successes]
            ax.plot(xs, ys, color="#2ecc71", alpha=0.35, linewidth=2, zorder=1)

        ax.scatter(indices, scores, c=colors, s=120, edgecolors="black", zorder=2)

        # Running best line
        best_so_far: list[float] = []
        current_best = float("-inf")
        for s in summaries:
            if (
                s.status == ExperimentStatus.COMPLETED.value
                and s.score is not None
                and s.score > current_best
            ):
                current_best = s.score
            best_so_far.append(current_best if current_best > float("-inf") else 0.0)
        ax.plot(
            indices,
            best_so_far,
            linestyle="--",
            color="#2c3e50",
            linewidth=1.5,
            label="running best",
            zorder=0,
        )

        ax.set_xlabel("Experiment number")
        ax.set_ylabel("ROC-AUC macro")
        ax.set_title(f"Score progression — {self.study.name}")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(indices)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower right")

        # Legend for point colors
        from matplotlib.patches import Patch

        legend_patches = [
            Patch(color="#2ecc71", label="completed"),
            Patch(color="#e74c3c", label="failed / no score"),
        ]
        ax.legend(handles=legend_patches + [ax.get_lines()[-1]], loc="lower right")

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    def _render_failure_breakdown(
        self, summaries: list[ExperimentSummary], path: Path
    ) -> bool:
        """Bar chart of error_type frequencies. Returns False if no failures."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        failures = [
            s.error_type or "Unknown"
            for s in summaries
            if s.status != ExperimentStatus.COMPLETED.value
        ]
        if not failures:
            return False

        counts = Counter(failures)
        types = list(counts.keys())
        values = [counts[t] for t in types]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(types, values, color="#e74c3c", edgecolor="black")
        ax.set_ylabel("Number of failed experiments")
        ax.set_title(f"Failure breakdown — {self.study.name}")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        for i, v in enumerate(values):
            ax.text(i, v + 0.05, str(v), ha="center", va="bottom", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return True

    def _render_learning_curve(
        self, best: ExperimentSummary, path: Path
    ) -> bool:
        """Render the best run's training curves. Returns False if <2 epochs."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        curves = best.training_curves
        loss_curve = curves.get("loss", [])
        auc_curve = curves.get("roc_auc_macro", [])
        if len(loss_curve) < 2 and len(auc_curve) < 2:
            return False

        epochs = list(range(1, max(len(loss_curve), len(auc_curve)) + 1))
        fig, ax1 = plt.subplots(figsize=(9, 5))

        if loss_curve:
            ax1.plot(
                epochs[: len(loss_curve)],
                loss_curve,
                color="#e67e22",
                marker="o",
                label="loss",
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss", color="#e67e22")
            ax1.tick_params(axis="y", labelcolor="#e67e22")
            ax1.grid(True, alpha=0.3)

        if auc_curve:
            ax2 = ax1.twinx()
            ax2.plot(
                epochs[: len(auc_curve)],
                auc_curve,
                color="#2ecc71",
                marker="s",
                label="roc_auc_macro",
            )
            ax2.set_ylabel("ROC-AUC macro", color="#2ecc71")
            ax2.tick_params(axis="y", labelcolor="#2ecc71")
            ax2.set_ylim(0, 1.05)

        ax1.set_title(
            f"Best run learning curve — {best.experiment_id} ({best.architecture})"
        )
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return True

    # -- LLM prompt assembly -----------------------------------------------

    def _ask_llm_for_report(
        self,
        summaries: list[ExperimentSummary],
        available_figures: list[str],
    ) -> str:
        """Fill the report prompt, call the LLM, return the Markdown body."""
        template = self.prompt_engine.load(self.prompt_path)

        slots = {
            "study_metadata": self._format_metadata(),
            "experiment_table": self._format_experiment_table(summaries),
            "best_experiment": self._format_best_experiment(summaries),
            "failure_breakdown": self._format_failure_breakdown(summaries),
            "available_figures": self._format_available_figures(available_figures),
        }

        system, user = self.prompt_engine.fill(template, slots)
        response = self.llm_client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        return _strip_wrapping_fence(response.strip()) + "\n"

    # -- formatting helpers (kept pure so they are trivially testable) -----

    def _format_metadata(self) -> str:
        bud = self.study.compute_budget
        lines = [
            f"- name: {self.study.name}",
            f"- id: {self.study.study_id}",
            f"- status: {self.study.status}",
            f"- hypothesis: {self.study.hypothesis}",
            f"- pipeline: {self.study.pipeline_config_path}",
            f"- compute budget: {bud.max_experiments} experiments, "
            f"{bud.max_wallclock_minutes} min wallclock, "
            f"{bud.max_experiment_seconds}s/run, "
            f"{bud.max_epochs_per_run} epochs/run, "
            f"up to {bud.max_recovery_attempts} recovery attempts",
            f"- experiments attempted: {len(self.study.experiment_ids)}",
            f"- best experiment: {self.study.best_experiment_id or '(none)'}",
            f"- best score: "
            f"{self.study.best_score:.4f}" if self.study.best_score is not None
            else "- best score: (none)",
            f"- created: {self.study.created_at}",
            f"- completed: {self.study.updated_at}",
        ]
        return "\n".join(lines)

    def _format_experiment_table(self, summaries: list[ExperimentSummary]) -> str:
        if not summaries:
            return "(no experiments recorded)"
        lines = ["| Exp | Status | Architecture | ROC-AUC | Loss | Duration | Notes |"]
        lines.append("|---|---|---|---|---|---|---|")
        for s in summaries:
            score = f"{s.score:.4f}" if s.score is not None else "—"
            loss = f"{s.loss:.4f}" if s.loss is not None else "—"
            dur = f"{s.duration_seconds:.0f}s" if s.duration_seconds else "—"
            notes = ""
            if s.error_type:
                msg = (s.error_message or "").splitlines()[0][:80]
                notes = f"{s.error_type}: {msg}"
            arch = s.architecture[:60]
            lines.append(
                f"| {s.experiment_id} | {s.status} | {arch} | {score} | "
                f"{loss} | {dur} | {notes} |"
            )
        return "\n".join(lines)

    def _format_best_experiment(self, summaries: list[ExperimentSummary]) -> str:
        best = self._best_successful(summaries)
        if best is None:
            return "(no successful experiments)"
        lines = [
            f"- id: {best.experiment_id}",
            f"- architecture: {best.architecture}",
            f"- roc_auc_macro: {best.score:.4f}" if best.score is not None else "- roc_auc_macro: —",
            f"- loss: {best.loss:.4f}" if best.loss is not None else "- loss: —",
            f"- duration: {best.duration_seconds:.0f}s" if best.duration_seconds else "- duration: —",
        ]
        if best.hyperparams:
            lines.append(f"- hyperparams: {best.hyperparams}")
        if best.training_curves:
            for metric, values in best.training_curves.items():
                if values:
                    head = ", ".join(f"{v:.4f}" for v in values[:3])
                    tail = "..." if len(values) > 3 else ""
                    lines.append(
                        f"  - {metric} ({len(values)} epoch values): [{head}{tail}]"
                    )
        return "\n".join(lines)

    def _format_failure_breakdown(self, summaries: list[ExperimentSummary]) -> str:
        failures = [s for s in summaries if s.status != ExperimentStatus.COMPLETED.value]
        if not failures:
            return "(no failures)"
        by_type: dict[str, list[ExperimentSummary]] = {}
        for s in failures:
            by_type.setdefault(s.error_type or "Unknown", []).append(s)
        lines: list[str] = []
        for err_type in sorted(by_type.keys()):
            entries = by_type[err_type]
            lines.append(f"- **{err_type}** ({len(entries)} experiments):")
            for s in entries:
                msg = (s.error_message or "").splitlines()[0][:100]
                lines.append(f"  - {s.experiment_id}: {msg}")
        return "\n".join(lines)

    def _format_available_figures(self, available: list[str]) -> str:
        if not available:
            return "(no figures were rendered)"
        return "\n".join(f"- figures/{f}" for f in available)

    # -- placeholder when LLM is unavailable -------------------------------

    def _placeholder_markdown(
        self,
        summaries: list[ExperimentSummary],
        available_figures: list[str],
    ) -> str:
        """Minimal report when the LLM call fails. The user still gets the
        figures and the raw data — they just have to write the prose."""
        parts = [
            f"# Study: {self.study.name}",
            "",
            "> ⚠ The LLM call for automatic report generation failed. This is",
            "> a data-only placeholder. Figures have still been rendered.",
            "",
            "## Metadata",
            self._format_metadata(),
            "",
            "## Experiments",
            self._format_experiment_table(summaries),
            "",
            "## Best Experiment",
            self._format_best_experiment(summaries),
            "",
            "## Failure Breakdown",
            self._format_failure_breakdown(summaries),
            "",
            "## Figures",
        ]
        if available_figures:
            for f in available_figures:
                parts.append(f"![{f}](figures/{f})")
        else:
            parts.append("(no figures)")
        return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _strip_wrapping_fence(text: str) -> str:
    """If the LLM wrapped its whole response in ```markdown ... ```, strip it."""
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text


__all__ = ["ReportGenerator", "ExperimentSummary"]
