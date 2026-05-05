"""Study runner — the agent loop (ADR-005)."""

from __future__ import annotations

import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lab.core import telemetry
from lab.core.experiment import RunContext, run_experiment
from lab.core.models import Experiment, Study, new_study_id


class StudyRunner:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self._aborted = False

    # ------------------------------------------------------------------

    def run(self, *, predecessor: Study | None = None) -> Study:
        cb = self.ctx.settings.compute_budget
        study = Study(
            id=new_study_id(),
            task_name=self.ctx.settings.task_name,
            status="RUNNING",
            personality=self.ctx.settings.agent.personality,
            agent_memory_enabled=self.ctx.settings.agent.memory_enabled,
            predecessor_id=predecessor.id if predecessor else None,
            prompt_template_paths=_active_prompt_paths(self.ctx),
            experiments=[],
            created_at=datetime.now(timezone.utc),
        )

        telemetry.configure(self.ctx.settings, study_id=study.id)
        telemetry.log_event("study_start", task=study.task_name)
        study.save(Path(self.ctx.settings.paths.experiments_dir))

        self._wire_agent_memory(study, predecessor)

        self._install_sigint(study)

        deadline = time.monotonic() + cb.max_wallclock_minutes * 60
        for index in range(cb.max_experiments):
            if self._aborted:
                break
            if time.monotonic() > deadline:
                telemetry.log_event("study_end", reason="wallclock_exceeded", level="warn")
                break

            exp = Experiment(
                id=f"exp_{index:04d}",
                index=index,
                status="PROPOSED",
                primary_metric=self.ctx.settings.task.primary_metric,
            )
            try:
                run_experiment(exp, self.ctx)
            except Exception as exc:  # pragma: no cover - defensive
                telemetry.log_event(
                    "error", phase="run_experiment", message=str(exc), level="error"
                )
                exp.status = "FAILED"

            study.experiments.append(exp)

            if exp.primary_score is not None and (
                study.best_score is None or exp.primary_score > study.best_score
            ):
                study.best_score = exp.primary_score
                study.best_experiment_id = exp.id

            study.save(Path(self.ctx.settings.paths.experiments_dir))

            if (
                exp.verdict is not None
                and exp.verdict.verdict == "abort_study"
            ):
                telemetry.log_event(
                    "study_end", reason="judge_abort", level="warn"
                )
                break

        study.status = "ABORTED" if self._aborted else "COMPLETED"
        study.finished_at = datetime.now(timezone.utc)

        try:
            study.study_verdict = self.ctx.judge.judge_study(
                study,
                budget_used=len(study.experiments),
                budget_total=cb.max_experiments,
            )
        except Exception as exc:  # pragma: no cover - defensive
            telemetry.log_event(
                "error", phase="judge_study", message=str(exc), level="error"
            )

        study.save(Path(self.ctx.settings.paths.experiments_dir))
        telemetry.log_event("study_end", status=study.status)
        return study

    def abort(self) -> None:
        self._aborted = True

    # ------------------------------------------------------------------

    def _wire_agent_memory(self, study: Study, predecessor: Study | None) -> None:
        """Honour settings.agent.memory_enabled at study start (ADR-007)."""
        if predecessor is not None:
            self.ctx.memory.seed_from_predecessor(predecessor)
        if self.ctx.settings.agent.memory_enabled:
            self.ctx.memory.seed_from_agent_memory(
                Path(self.ctx.settings.paths.experiments_dir),
                task=self.ctx.settings.task_name,
            )

    # ------------------------------------------------------------------

    def _install_sigint(self, study: Study) -> None:
        def _handler(signum, frame):  # pragma: no cover - signal path
            self.abort()

        try:
            signal.signal(signal.SIGINT, _handler)
        except ValueError:  # pragma: no cover - non-main thread
            pass


def _active_prompt_paths(ctx: RunContext) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for task in (
        "propose_architecture",
        "generate_code",
        "recover_from_error",
        "analyze_result",
        "judge_experiment",
        "judge_study",
    ):
        try:
            tmpl = ctx.engine.registry.load(task)
            out[task] = Path(tmpl.path)
        except Exception:  # pragma: no cover
            pass
    return out
