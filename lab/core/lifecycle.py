"""Study runner — the agent loop (ADR-005)."""

from __future__ import annotations

import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lab.core import telemetry
from lab.core.experiment import RunContext, run_experiment
from lab.core.models import Experiment, Study, new_study_id
from lab.core.watchdog import Watchdog


_WATCHDOG_TIMEOUT_ENV = "AGENT_WATCHDOG_TIMEOUT_S"
_WATCHDOG_DEFAULT_TIMEOUT_S = 1800


# Registry of in-flight runners keyed by study_id. Populated when a runner
# enters its run loop and removed on exit, so the UI / external callers
# can flip the abort flag mid-run without poking thread internals. Lives
# at module scope on purpose -- the FastAPI process keeps it across
# requests; kill the process and any registered runners go with it.
_RUNNING_RUNNERS: dict[str, "StudyRunner"] = {}


def get_running_runner(study_id: str) -> "StudyRunner | None":
    """Return the live StudyRunner for ``study_id`` if one is registered.

    Returns None when the study is no longer running (already finished,
    aborted, or never started in this process). The UI's stop button
    uses this to know whether the request can do anything useful.
    """
    return _RUNNING_RUNNERS.get(study_id)


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
            llm_model=f"{self.ctx.settings.llm.provider}:{self.ctx.settings.llm.model}",
        )

        telemetry.configure(self.ctx.settings, study_id=study.id)
        telemetry.log_event("study_start", task=study.task_name)
        study.save(Path(self.ctx.settings.paths.experiments_dir))

        # Make ourselves reachable by the UI's stop endpoint. The slot is
        # cleared in the finally below regardless of how the run exits.
        _RUNNING_RUNNERS[study.id] = self
        try:
            return self._run_body(study, predecessor, cb)
        finally:
            _RUNNING_RUNNERS.pop(study.id, None)

    def _run_body(self, study: Study, predecessor: Study | None, cb) -> Study:
        self._wire_agent_memory(study, predecessor)

        self._install_sigint(study)

        watchdog = Watchdog(
            study.id,
            Path(self.ctx.settings.paths.experiments_dir),
            timeout_s=int(
                os.environ.get(_WATCHDOG_TIMEOUT_ENV, _WATCHDOG_DEFAULT_TIMEOUT_S)
            ),
            sandbox_root=Path(self.ctx.settings.paths.sandbox),
        )
        watchdog.start()

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
            # Make the experiment visible BEFORE run_experiment so the UI's
            # pipeline view sees it the moment work starts. Wire a progress
            # callback that re-saves study.json after every meaningful state
            # change inside run_experiment.
            study.experiments.append(exp)
            studies_dir = Path(self.ctx.settings.paths.experiments_dir)
            study.save(studies_dir)

            previous_progress = self.ctx.on_progress
            previous_study_id = self.ctx.study_id
            self.ctx.on_progress = lambda: study.save(studies_dir)
            self.ctx.study_id = study.id
            try:
                run_experiment(exp, self.ctx)
            except Exception as exc:  # pragma: no cover - defensive
                telemetry.log_event(
                    "error", phase="run_experiment", message=str(exc), level="error"
                )
                exp.status = "FAILED"
            finally:
                self.ctx.on_progress = previous_progress
                self.ctx.study_id = previous_study_id

            if exp.primary_score is not None and (
                study.best_score is None or exp.primary_score > study.best_score
            ):
                study.best_score = exp.primary_score
                study.best_experiment_id = exp.id

            study.save(studies_dir)

            if (
                exp.verdict is not None
                and exp.verdict.verdict == "abort_study"
            ):
                telemetry.log_event(
                    "study_end", reason="judge_abort", level="warn"
                )
                break

        watchdog.stop()
        if watchdog.stalled:
            telemetry.log_event(
                "study_end",
                reason="watchdog_stalled",
                level="error",
            )
            study.status = "FAILED"
            study.finished_at = datetime.now(timezone.utc)
            study.save(Path(self.ctx.settings.paths.experiments_dir))
            return study

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
