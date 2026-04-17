"""Agent orchestrator.

Runs the experiment loop:

    for i in range(max_experiments):
        proposal = propose_architecture(memory)      # LLM
        code     = generate_code(proposal)            # LLM
        for attempt in range(max_codegen_retries):    # Layer 1 retry
            if validator.validate(code).ok: break
            code = regenerate_with_error(code, err)  # LLM
        for attempt in range(max_recovery_attempts):  # Layer 2 retry
            result = executor.run(code)
            if result.succeeded: break
            code = recover_from_error(code, result.error)
        memory.add(experiment)
        save_study()

Everything task-specific comes from ``TaskAdapter`` and the context builder.
"""
from __future__ import annotations

import json
import logging
import re
import signal
import time
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from lab.config import Settings
from lab.core import validator as validator_mod
from lab.core.context import ContextBuilder, load_model_registry
from lab.core.executor import Executor, ExecutionResult, LocalExecutor
from lab.core.llm import LLMClient, LLMError, format_messages
from lab.core.memory import Memory
from lab.core.models import (
    Experiment,
    ExperimentStatus,
    Study,
    StudyStatus,
    Task,
    TaskError,
    TaskStatus,
)
from lab.prompts.engine import PromptEngine
from lab.prompts.registry import PromptRegistry
from lab.tasks.base import TaskAdapter


logger = logging.getLogger("lab.orchestrator")


# Failure categories that recovery cannot fix from inside the training
# script. The LLM has no way to summon missing data, allocate more RAM,
# or extend a wallclock budget — we mark the experiment failed and move
# on instead of burning N retries on a guaranteed lost cause.
_HARD_FAILURE_ERROR_TYPES = frozenset({
    "Timeout",
    "OOM",
    "FileNotFound",
    "Aborted",
})
_NO_RUNTIME_RECOVERY_BACKENDS = frozenset({"kaggle", "modal"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorHooks:
    """Optional observability hooks. Each returns nothing; exceptions ignored."""
    on_study_start: Callable[[Study], None] | None = None
    on_experiment_start: Callable[[Experiment], None] | None = None
    on_experiment_end: Callable[[Experiment], None] | None = None
    on_study_end: Callable[[Study], None] | None = None


class Orchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        adapter: TaskAdapter,
        llm: LLMClient | None = None,
        hooks: OrchestratorHooks | None = None,
        predecessor: Study | None = None,
        prompt_overrides: dict[str, str] | None = None,
        launch_id: str | None = None,
        executor_backend: str | None = None,
        primary_metrics: list[str] | None = None,
    ):
        self.settings = settings
        self.adapter = adapter
        self.llm = llm or LLMClient(
            base_url=settings.llm.base_url,
            model=settings.llm.default_model,
            api_key=settings.llm.api_key,
            provider=settings.llm.provider,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout_seconds=settings.llm.timeout_seconds,
            retry_attempts=settings.llm.retry_attempts,
            retry_backoff_seconds=settings.llm.retry_backoff_seconds,
        )
        self.hooks = hooks or OrchestratorHooks()
        self.predecessor = predecessor
        # Per-study prompt version overrides: {"propose_architecture": "v2", ...}
        # Empty/missing entries fall back to the registry's active version.
        self.prompt_overrides: dict[str, str] = dict(prompt_overrides or {})
        # When spawned from the UI, launch_id is set so we can report our
        # study_id back to the launch record for UI attribution.
        self.launch_id = launch_id

        self.registry = PromptRegistry(settings.abspath(settings.paths.prompts_dir))
        self.engine = PromptEngine(self.registry)

        self.memory = Memory(score_metric=adapter.primary_metric)
        if predecessor is not None:
            self.memory.seed_from_predecessor(predecessor)

        skeleton_path = settings.abspath(adapter.code_skeleton_path())
        skeleton_content = skeleton_path.read_text() if skeleton_path.exists() else ""
        self._code_skeleton = skeleton_content
        slots = dict(adapter.prompt_slot_values())
        slots["code_skeleton_content"] = skeleton_content

        model_reg = load_model_registry(settings.abspath(adapter.model_registry_path()))

        self.context = ContextBuilder(
            settings=settings,
            memory=self.memory,
            dataset_profile=adapter.load_profile(),
            model_registry=model_reg,
            task_slots=slots,
        )
        schedule = [m.strip() for m in (primary_metrics or []) if m and m.strip()]
        self._primary_metric_schedule: list[str] = schedule or [adapter.primary_metric]
        self._study_primary_metric: str = self._primary_metric_schedule[0]
        self._validator_smoke_input_shape = _resolve_validator_smoke_input_shape(adapter)
        self._validator_smoke_num_classes = _resolve_validator_smoke_num_classes(
            self.context.dataset_profile,
        )
        try:
            fn_name, first_arg = adapter.model_block_signature()
        except Exception:  # noqa: BLE001
            fn_name, first_arg = ("build_model", "num_classes")
        self._model_block_fn_name = fn_name
        self._model_block_first_arg = first_arg

        sandbox_root = settings.abspath(settings.paths.sandbox)
        backend = (executor_backend or settings.executor.backend or "local").strip().lower()
        self.executor: Executor = _build_executor(
            backend=backend,
            settings=settings,
            sandbox_root=sandbox_root,
            training_env=self._training_env(),
        )
        logger.info("Executor backend: %s", self.executor.backend)

        # Graceful SIGINT handling — set by run()
        self._abort_requested = False
        # Live-save hook — set while an experiment is in flight so
        # `_run_experiment` can re-persist the study on every status
        # change (so the UI sees "running" / "failed" live).
        self._live_save: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_study(self, name: str = "", tags: list[str] | None = None) -> Study:
        study = Study(
            name=name or f"{self.adapter.name}-{_now().strftime('%Y%m%d_%H%M%S')}",
            task_name=self.adapter.name,
            primary_metric=self._study_primary_metric,
            predecessor_id=self.predecessor.id if self.predecessor else None,
            prompt_template_paths=self._resolve_prompt_paths(),
            publish=self.settings.publishing.default_publish,
            tags=tags or list(self.settings.publishing.default_tags),
            executor_backend=self.executor.backend,
            executor_infrastructure=self.executor.infrastructure(),
        )
        study.status = StudyStatus.RUNNING
        study.started_at = _now()
        experiments_dir = self.settings.abspath(self.settings.paths.experiments)
        study.save(experiments_dir)

        # Let the UI link a running launch to the study it ended up creating.
        if self.launch_id:
            try:
                from lab.ui import launches  # local import to avoid cycles
                launches.set_study_id(self.launch_id, study.id, self.settings.repo_root)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to write launch breadcrumb for %s", self.launch_id)

        self._fire(self.hooks.on_study_start, study)

        # SIGINT → graceful abort
        prev = signal.signal(signal.SIGINT, lambda *_: self._request_abort())

        start_wall = time.monotonic()
        budget = self.settings.compute_budget
        try:
            for i in range(budget.max_experiments):
                if self._abort_requested:
                    logger.warning("Abort requested — stopping study.")
                    break
                elapsed_minutes = (time.monotonic() - start_wall) / 60.0
                if elapsed_minutes >= budget.max_wallclock_minutes:
                    logger.warning("Wallclock budget exhausted after %.1f minutes.", elapsed_minutes)
                    break

                exp = Experiment(
                    study_id=study.id,
                    index=i,
                    primary_metric=self._metric_for_experiment(i),
                )
                logger.info("── experiment %d/%d: %s", i + 1, budget.max_experiments, exp.id)
                self._fire(self.hooks.on_experiment_start, exp)
                # Append immediately so the UI can show the experiment
                # as "running" during long LLM / Kaggle steps. The live
                # save hook re-persists the study every time the current
                # experiment updates.
                study.experiments.append(exp)
                self._live_save = lambda: study.save(experiments_dir)
                self._live_save()
                try:
                    self._run_experiment(exp)
                finally:
                    self._live_save = None
                self.memory.add(exp)
                self.memory.save(experiments_dir / study.id / "memory.json")

                self._update_best(study)
                study.save(experiments_dir)
                self._fire(self.hooks.on_experiment_end, exp)

            study.status = self._derive_final_status(
                study.experiments, aborted=self._abort_requested,
            )
        except Exception:  # noqa: BLE001 — we want to persist even on crash
            logger.exception("Orchestrator crashed")
            study.status = StudyStatus.FAILED
        finally:
            signal.signal(signal.SIGINT, prev)
            study.completed_at = _now()
            study.save(experiments_dir)
            self._fire(self.hooks.on_study_end, study)
        return study

    def _request_abort(self) -> None:
        """SIGINT handler — must be async-signal-safe and never block.

        We do two things:
          1. Set the abort flag so the main loop breaks at its next
             check and `_execute_with_recovery` stops retrying.
          2. Immediately kill the currently-running training subprocess
             if any. That subprocess lives in its own process group
             (``preexec_fn=os.setsid``), so the SIGINT delivered to the
             agent's pgid does NOT reach it on its own. Without this
             call the agent would keep blocking in `process.wait()`
             until the training finished naturally — defeating the
             whole point of a Stop button.
        """
        already = self._abort_requested
        self._abort_requested = True
        logger.warning(
            "%s received — stopping.%s",
            "Second abort" if already else "Abort",
            " Killing training subprocess." if already else "",
        )
        killer = getattr(self.executor, "kill_running", None)
        if callable(killer):
            try:
                killer()
            except Exception:  # noqa: BLE001
                logger.exception("executor.kill_running failed")

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def _run_experiment(self, exp: Experiment) -> None:
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = _now()
        self._persist()

        # 1) Propose architecture
        propose_task = self._begin_task(exp, "propose_architecture")
        try:
            proposal_text = self._propose_architecture(primary_metric=exp.primary_metric)
            proposal_data = self._parse_proposal(proposal_text)
            exp.architecture_name = proposal_data.get("architecture_name", "unknown")
            exp.architecture_family = proposal_data.get("architecture_family")
            exp.architecture_proposal = json.dumps(proposal_data, indent=2)
            self._finish_task(propose_task, status=TaskStatus.COMPLETED, output=proposal_data)
        except Exception as exc:  # noqa: BLE001
            logger.exception("propose_architecture failed")
            err = TaskError(error_type="LLMError", message=str(exc))
            self._finish_task(propose_task, status=TaskStatus.FAILED, error=err)
            exp.status = ExperimentStatus.FAILED
            exp.error = err
            exp.completed_at = _now()
            self._persist()
            return

        # 2) Generate code with Layer-1 retry (codegen + validator)
        gen_task = self._begin_task(exp, "generate_code")
        try:
            code = self._generate_code(
                exp.architecture_proposal or "",
                primary_metric=exp.primary_metric,
            )
        except LLMError as exc:
            err = TaskError(error_type="LLMError", message=str(exc))
            self._finish_task(gen_task, status=TaskStatus.FAILED, error=err)
            exp.status = ExperimentStatus.FAILED
            exp.error = err
            exp.completed_at = _now()
            self._persist()
            return
        self._finish_task(gen_task, status=TaskStatus.COMPLETED, code=code)

        # 2b) Validate generated code with Layer-1 retry (validator + repair)
        validate_task = self._begin_task(exp, "validate_code")
        val_ok = False
        gen_err: TaskError | None = None
        validator_attempts = 0
        for attempt in range(1, self.settings.compute_budget.max_codegen_retries + 1):
            validator_attempts = attempt
            result = validator_mod.validate(
                code,
                extra_spawn_triggers=self.adapter.spawn_triggering_calls(),
                require_build_model=True,
                require_training_contract=True,
                smoke_input_shape=self._validator_smoke_input_shape,
                smoke_num_classes=self._validator_smoke_num_classes,
            )
            if result.ok:
                val_ok = True
                break
            auto_fixed = _apply_validator_autofix(code, result)
            if auto_fixed is not None and auto_fixed != code:
                logger.info(
                    "Applied validator auto-fix (%s): %s",
                    result.error_type,
                    result.message[:140],
                )
                code = auto_fixed
                continue
            logger.warning("Validator rejected code (attempt %d): %s", attempt, result.message)
            try:
                code = self._recover_code(
                    code,
                    result.error_type,
                    result.message,
                    traceback="",
                    primary_metric=exp.primary_metric,
                )
            except LLMError as exc:
                gen_err = TaskError(error_type="LLMError", message=str(exc))
                break

        exp.code = code
        if not val_ok:
            if gen_err is None:
                gen_err = TaskError(
                    error_type="ValidationFailed",
                    message="Codegen retries exhausted",
                )
            self._finish_task(
                validate_task, status=TaskStatus.FAILED, error=gen_err, code=code,
                output={"validator_attempts": validator_attempts},
            )
            exp.status = ExperimentStatus.FAILED
            exp.error = gen_err
            exp.completed_at = _now()
            self._persist()
            return
        self._finish_task(
            validate_task, status=TaskStatus.COMPLETED, code=code,
            output={"validator_attempts": validator_attempts},
        )

        # 3) Execute with Layer-2 retry (runtime error recovery)
        run_task = self._begin_task(exp, "execute_training")
        extra_env = _proposal_hyperparam_env(
            proposal_data,
            env_prefix=self.settings.env_prefix,
        )
        # Resolve a prior-experiment warm-start reference, if any. Only
        # wired on local backend today — remote kernels can't reach the
        # orchestrator's filesystem and staging checkpoints into Kaggle
        # datasets is a larger change we haven't landed.
        init_exp_id = extra_env.pop(
            f"{self.settings.env_prefix}_INIT_FROM_EXPERIMENT_ID",
            "",
        )
        if init_exp_id and self.executor.backend == "local":
            ckpt = self._resolve_checkpoint_path(init_exp_id)
            if ckpt is not None:
                extra_env[f"{self.settings.env_prefix}_INIT_FROM_CHECKPOINT"] = str(ckpt)
                logger.info("Warm-starting from %s (%s)", init_exp_id, ckpt)
            else:
                logger.warning(
                    "Warm-start requested from %s but no checkpoint found; skipping",
                    init_exp_id,
                )
        elif init_exp_id:
            logger.info(
                "Warm-start from %s ignored on %s backend (not yet supported).",
                init_exp_id, self.executor.backend,
            )
        exec_result = self._execute_with_recovery(code, exp, extra_env=extra_env)
        exp.sandbox_path = str(exec_result.workdir)
        exp.duration_seconds = exec_result.duration_seconds
        run_status = TaskStatus.COMPLETED if exec_result.succeeded else TaskStatus.FAILED
        self._finish_task(
            run_task, status=run_status, error=exec_result.error,
            output={
                "duration_seconds": exec_result.duration_seconds,
                "exit_code": exec_result.exit_code,
                "timed_out": exec_result.timed_out,
            },
        )

        if not exec_result.succeeded:
            exp.status = (
                ExperimentStatus.ABORTED
                if exec_result.error and exec_result.error.error_type == "Aborted"
                else ExperimentStatus.FAILED
            )
            exp.error = exec_result.error
            exp.completed_at = _now()
            self._persist()
            return

        # 4) Capture metrics
        metrics_task = self._begin_task(exp, "capture_metrics")
        try:
            metrics = self._capture_metrics(
                exec_result,
                expected_primary_metric=exp.primary_metric,
            )
            exp.primary_metric = str(metrics.get("primary_metric") or exp.primary_metric)
            exp.metrics = metrics.get("metrics", {})
            exp.history = metrics.get("history", [])
            exp.primary_score = metrics.get("primary_score")
            raw = metrics.get("raw", {}) or {}
            in_band_error = raw.get("error")
            extra_errors = self.adapter.validate_training_output(
                raw,
                expected_primary_metric=exp.primary_metric,
            )
            if in_band_error:
                err = TaskError(
                    error_type="ScriptReportedError",
                    message=str(in_band_error)[:300],
                )
                self._finish_task(metrics_task, status=TaskStatus.FAILED, error=err)
                exp.status = ExperimentStatus.FAILED
                exp.error = err
                # Drop the fake 0-score so study.best_score reflects only
                # genuine results.
                exp.primary_score = None
            elif extra_errors:
                err = TaskError(
                    error_type="TaskValidationFailed",
                    message="; ".join(extra_errors),
                )
                self._finish_task(metrics_task, status=TaskStatus.FAILED, error=err)
                exp.status = ExperimentStatus.FAILED
                exp.error = err
            else:
                self._finish_task(metrics_task, status=TaskStatus.COMPLETED)
                exp.status = ExperimentStatus.COMPLETED
                self._archive_best_checkpoint(exp, exec_result)
        except Exception as exc:  # noqa: BLE001
            err = TaskError(error_type="MetricsError", message=str(exc))
            self._finish_task(metrics_task, status=TaskStatus.FAILED, error=err)
            exp.status = ExperimentStatus.FAILED
            exp.error = err
        exp.completed_at = _now()
        self._persist()

    def _execute_with_recovery(
        self,
        code: str,
        exp: Experiment,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        max_attempts = self.settings.compute_budget.max_recovery_attempts
        # Remote executors already run full jobs out-of-process; blindly
        # re-pushing the same experiment on failure burns quota and keeps
        # the local orchestrator alive even though the remote run is over.
        # For these backends, one execute attempt per experiment is enough.
        if getattr(self.executor, "backend", "") in _NO_RUNTIME_RECOVERY_BACKENDS:
            max_attempts = 1
        for attempt in range(1, max_attempts + 1):
            if self._abort_requested:
                # User hit Stop — don't spawn another training subprocess.
                return self._aborted_result(exp)
            result = self.executor.run(code, experiment_id=exp.id, extra_env=extra_env)
            if result.succeeded:
                return result
            if self._abort_requested:
                # The last execute was killed by our own abort. Surface
                # it as an Aborted failure so the recovery loop doesn't
                # keep burning LLM calls trying to "fix" the crash.
                return self._aborted_result(exp, base=result)
            if result.error is None or result.error.error_type in _HARD_FAILURE_ERROR_TYPES:
                # Hard failures the LLM can't fix from inside the script —
                # stop retrying and surface the failure cleanly.
                return result
            if attempt >= max_attempts:
                return result
            logger.warning(
                "execute failed (%s): %s — recovering (attempt %d/%d)",
                result.error.error_type, result.error.message, attempt, max_attempts,
            )
            try:
                code = self._recover_code(
                    code,
                    error_type=result.error.error_type,
                    error_message=result.error.message,
                    traceback=result.error.traceback or "",
                    primary_metric=exp.primary_metric,
                )
                # Re-validate the recovered code BEFORE shipping it to the
                # executor. The recovery LLM occasionally returns prose
                # ("Looking at the error, the issue is …") which would
                # otherwise blow up at runtime as a SyntaxError. We give
                # the LLM up to 3 chances to correct itself before giving
                # up on this experiment.
                code = self._revalidate_with_retries(code, exp)
                exp.code = code
            except LLMError as exc:
                logger.error("recover_from_error LLM call failed: %s", exc)
                return result
        return result  # type: ignore[return-value]

    @staticmethod
    def _aborted_result(exp: Experiment, *, base: ExecutionResult | None = None) -> ExecutionResult:
        """Uniform ExecutionResult for abort paths — so the caller
        can treat "user pressed Stop" identically to any other failure."""
        from pathlib import Path
        return ExecutionResult(
            exit_code=-1,
            stdout=base.stdout if base else "",
            stderr=base.stderr if base else "",
            duration_seconds=base.duration_seconds if base else 0.0,
            workdir=Path(base.workdir) if base else Path("."),
            results_json_path=None,
            error=TaskError(error_type="Aborted", message="User requested abort"),
            timed_out=False,
        )

    def _revalidate_with_retries(self, code: str, exp: Experiment) -> str:
        """Run the validator on `code`; on failure, re-prompt the LLM
        (up to 3 attempts) to fix the validation error. Returns the
        last code we have, validated or not — the executor will then
        produce the actual error which the outer recovery loop handles.
        """
        for revalidation_attempt in range(1, 4):
            val = validator_mod.validate(
                code,
                extra_spawn_triggers=self.adapter.spawn_triggering_calls(),
                require_build_model=True,
                require_training_contract=True,
                smoke_input_shape=self._validator_smoke_input_shape,
                smoke_num_classes=self._validator_smoke_num_classes,
            )
            if val.ok:
                return code
            auto_fixed = _apply_validator_autofix(code, val)
            if auto_fixed is not None and auto_fixed != code:
                logger.info(
                    "Applied validator auto-fix during runtime recovery (%s): %s",
                    val.error_type,
                    val.message[:120],
                )
                code = auto_fixed
                continue
            logger.warning(
                "Recovered code rejected by validator (%s): %s — re-prompting (%d/3)",
                val.error_type, val.message[:120], revalidation_attempt,
            )
            try:
                code = self._recover_code(
                    code,
                    error_type=val.error_type,
                    error_message=val.message,
                    traceback="",
                    primary_metric=exp.primary_metric,
                )
            except LLMError:
                break
        return code

    # ------------------------------------------------------------------
    # LLM calls — thin wrappers around prompt engine + llm client
    # ------------------------------------------------------------------

    def _metric_for_experiment(self, index: int) -> str:
        schedule = self._primary_metric_schedule
        if not schedule:
            return self.adapter.primary_metric
        return schedule[index % len(schedule)]

    def _propose_architecture(self, *, primary_metric: str) -> str:
        slots = self.context.build(
            primary_metric=primary_metric,
            available_checkpoints=self._available_checkpoints_markdown(),
            eda_summary=self._eda_summary_markdown(),
        )
        system, user = self.engine.render(
            "propose_architecture", slots,
            version=self.prompt_overrides.get("propose_architecture"),
        )
        return self.llm.chat(format_messages(system, user))

    def _generate_code(self, architecture_proposal: str, *, primary_metric: str) -> str:
        slots = self.context.build(
            architecture_proposal=architecture_proposal,
            primary_metric=primary_metric,
        )
        system, user = self.engine.render(
            "generate_code", slots,
            version=self.prompt_overrides.get("generate_code"),
        )
        raw = self.llm.chat(format_messages(system, user))
        return _compose_code_from_model_response(
            raw,
            template_code=self._code_skeleton,
            fn_name=self._model_block_fn_name,
            first_arg_name=self._model_block_first_arg,
        )

    def _recover_code(
        self,
        code: str,
        error_type: str,
        error_message: str,
        traceback: str,
        *,
        primary_metric: str | None = None,
    ) -> str:
        slots = self.context.build(
            error_type=error_type,
            error_message=error_message,
            error_traceback=traceback,
            broken_code=code,
            primary_metric=primary_metric or self._study_primary_metric,
        )
        system, user = self.engine.render(
            "recover_from_error", slots,
            version=self.prompt_overrides.get("recover_from_error"),
        )
        raw = self.llm.chat(format_messages(system, user))
        return _compose_code_from_model_response(
            raw,
            template_code=code,
            fn_name=self._model_block_fn_name,
            first_arg_name=self._model_block_first_arg,
        )

    # ------------------------------------------------------------------
    # Metrics + helpers
    # ------------------------------------------------------------------

    def _capture_metrics(
        self,
        exec_result: ExecutionResult,
        *,
        expected_primary_metric: str | None = None,
    ) -> dict:
        assert exec_result.results_json_path is not None
        raw = json.loads(exec_result.results_json_path.read_text())
        primary_metric = (
            str(expected_primary_metric).strip()
            if expected_primary_metric
            else str(raw.get("primary_metric") or self.adapter.primary_metric)
        )
        raw_primary_metric = str(raw.get("primary_metric") or "")
        raw_primary_score = raw.get("primary_score")
        primary_score = (
            float(raw_primary_score)
            if (
                raw_primary_metric == primary_metric
                and isinstance(raw_primary_score, (int, float))
            )
            else None
        )
        history = raw.get("history", [])
        final = raw.get("final", history[-1] if history else {})
        top_metrics = raw.get("metrics", {})
        best = raw.get("best", {})
        metrics: dict[str, float] = {}
        if isinstance(top_metrics, dict):
            for k, v in top_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        if isinstance(final, dict):
            for k, v in final.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        if not metrics and isinstance(history, list) and history:
            last_hist = history[-1] if isinstance(history[-1], dict) else {}
            if isinstance(last_hist, dict):
                for k, v in last_hist.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)
        if isinstance(best, dict):
            for k, v in best.items():
                if isinstance(v, (int, float)):
                    metrics.setdefault(k, float(v))
        if primary_score is None:
            score_from_best = _extract_metric_from_dict(best, primary_metric)
            if score_from_best is None:
                score_from_best = _extract_metric_from_history(history, primary_metric)
            if score_from_best is None:
                score_from_best = metrics.get(primary_metric)
            if isinstance(score_from_best, (int, float)):
                primary_score = float(score_from_best)
        if primary_score is None and isinstance(raw_primary_score, (int, float)):
            primary_score = float(raw_primary_score)
        return {
            "primary_metric": primary_metric,
            "metrics": metrics,
            "history": history,
            "primary_score": primary_score,
            "raw": raw,
        }

    def _update_best(self, study: Study) -> None:
        ok = [
            e
            for e in study.experiments
            if e.primary_score is not None
            and (not study.primary_metric or e.primary_metric == study.primary_metric)
        ]
        if not ok:
            ok = [e for e in study.experiments if e.primary_score is not None]
        if not ok:
            return
        best = max(ok, key=lambda e: e.primary_score)  # type: ignore[arg-type]
        study.best_experiment_id = best.id
        study.best_score = best.primary_score

    def _archive_best_checkpoint(self, exp: Experiment, exec_result: ExecutionResult) -> None:
        """Copy ``best.pt`` from the sandbox into the study's checkpoint dir.

        No-op when the skeleton didn't save a file (e.g. all-zero history)
        or the copy fails — we never want a successful training run to be
        flipped to failed because of a checkpoint bookkeeping issue.
        """
        try:
            src = Path(exec_result.workdir) / "best.pt"
            if not src.is_file():
                return
            dest_dir = self.settings.abspath(self.settings.paths.checkpoints) / exp.study_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{exp.id}.pt"
            import shutil as _sh
            _sh.copy2(src, dest)
            try:
                exp.checkpoint_path = str(dest.relative_to(self.settings.repo_root))
            except ValueError:
                exp.checkpoint_path = str(dest)
        except Exception:  # noqa: BLE001
            logger.exception("checkpoint archive failed for %s", exp.id)

    def _resolve_checkpoint_path(self, experiment_id: str) -> Path | None:
        """Find the archived checkpoint file for a prior experiment id."""
        for entry in self.memory.entries:
            candidate = entry.checkpoint_path if hasattr(entry, "checkpoint_path") else None
            if entry.experiment_id == experiment_id and candidate:
                p = self.settings.abspath(candidate)
                if p.is_file():
                    return p
        # Predecessor study's experiments: walk the study file if present.
        if self.predecessor is not None:
            for e in self.predecessor.experiments:
                if e.id == experiment_id and e.checkpoint_path:
                    p = self.settings.abspath(e.checkpoint_path)
                    if p.is_file():
                        return p
        # Fall back to scanning this study's live list — memory is added
        # only after the experiment completes, so the very next experiment
        # in the same study has to look here instead.
        ck_dir = self.settings.abspath(self.settings.paths.checkpoints)
        for study_dir in (ck_dir.iterdir() if ck_dir.is_dir() else ()):
            candidate = study_dir / f"{experiment_id}.pt"
            if candidate.is_file():
                return candidate
        return None

    def _eda_summary_markdown(self) -> str:
        """Return the checked-in EDA summary for this task, or empty."""
        try:
            from lab.tasks.eda import load_summary
            summary = load_summary(self.settings, self.adapter.name)
        except Exception:  # noqa: BLE001
            return ""
        return summary.strip()

    def _available_checkpoints_markdown(self) -> str:
        """Render a short markdown list of checkpoints the LLM may reference."""
        rows: list[str] = []
        for entry in self.memory.entries:
            ckpt = entry.checkpoint_path if hasattr(entry, "checkpoint_path") else None
            if not ckpt:
                continue
            score = entry.primary_score
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
            arch = entry.architecture_name or "unknown"
            rows.append(
                f"- `{entry.experiment_id}` — [{arch}] score={score_str}"
                f" (set `init_from_experiment_id` to warm-start from here)"
            )
        if not rows:
            return "_no archived checkpoints yet_"
        return "\n".join(rows)

    def _resolve_prompt_paths(self) -> dict[str, str]:
        """Record which prompt file was actually used for each prompt task.

        Respects ``self.prompt_overrides`` so per-study A/B runs are
        attributable to the right version in prompt_scoring.
        """
        out: dict[str, str] = {}
        for task in ("propose_architecture", "generate_code", "analyze_result",
                     "recover_from_error", "executive_summary"):
            try:
                override = self.prompt_overrides.get(task)
                if override:
                    data = self.registry._load_pointer()[task]
                    rel = data["versions"][override]["path"]
                    out[task] = str(self.registry.root / rel)
                else:
                    out[task] = str(self.registry.resolve_active_path(task))
            except KeyError:
                continue
        return out

    def _training_env(self) -> dict[str, str]:
        prefix = self.settings.env_prefix
        t = self.settings.training
        env = {
            f"{prefix}_EPOCHS": str(self.settings.compute_budget.max_epochs_per_run),
            f"{prefix}_BATCH_SIZE": str(t.batch_size),
            f"{prefix}_NUM_WORKERS": str(t.num_workers),
            f"{prefix}_DEVICE": t.device if t.device != "auto" else "cpu",
            f"{prefix}_PERSISTENT_WORKERS": "1" if t.persistent_workers else "0",
            f"{prefix}_PREFETCH_FACTOR": str(t.prefetch_factor),
        }
        env.update(self.adapter.env_vars(self.settings))
        # Offline pretrained weights — turn `weights="DEFAULT"` into a
        # local file lookup. Only applied when the path exists; absent
        # directory is a silent no-op so first-run users aren't forced
        # to pre-populate the cache before running locally.
        weights_dir = self.settings.abspath(self.settings.paths.offline_weights)
        if weights_dir.is_dir():
            env.setdefault("TORCH_HOME", str(weights_dir))
            hf_cache = weights_dir / "huggingface"
            env.setdefault("HF_HOME", str(hf_cache))
            env.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
            env.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
            env.setdefault("TIMM_HOME", str(weights_dir))
        return env

    @staticmethod
    def _derive_final_status(experiments: list[Experiment], *, aborted: bool) -> StudyStatus:
        """Pick the right end-of-loop status for the study.

        - User abort wins everything (matches what they asked for).
        - At least one COMPLETED experiment → study COMPLETED.
        - Empty experiments list or all FAILED → study FAILED.

        Without this rule the orchestrator marks every loop that exits
        without an exception as COMPLETED, which is misleading when all
        N experiments crashed.
        """
        if aborted:
            return StudyStatus.ABORTED
        if any(e.status == ExperimentStatus.COMPLETED for e in experiments):
            return StudyStatus.COMPLETED
        return StudyStatus.FAILED

    def _begin_task(self, exp: Experiment, name: str) -> Task:
        """Append a Task in RUNNING state and persist immediately so the
        UI can show "currently: <name>" without waiting for the work
        to finish."""
        t = Task(name=name, status=TaskStatus.RUNNING, started_at=_now())
        exp.tasks.append(t)
        self._persist()
        return t

    def _finish_task(
        self,
        task: Task,
        *,
        status: TaskStatus,
        error: TaskError | None = None,
        output: dict | None = None,
        code: str | None = None,
    ) -> None:
        task.status = status
        if error is not None:
            task.error = error
        if output:
            task.output.update(output)
        if code is not None:
            task.code_used = code
        task.completed_at = _now()
        self._persist()

    def _persist(self) -> None:
        """Re-save the study mid-experiment so the UI sees live status.

        No-op unless ``_live_save`` is set (it's set only while an
        experiment is in flight). Any IO error here is swallowed —
        we'd rather press on with the run than crash.
        """
        save = self._live_save
        if save is None:
            return
        try:
            save()
        except Exception:  # noqa: BLE001
            logger.exception("live save failed")

    def _fire(self, hook, *args) -> None:
        if hook is None:
            return
        try:
            hook(*args)
        except Exception:  # noqa: BLE001
            logger.exception("Hook %s raised", hook.__name__)

    def _parse_proposal(self, raw: str) -> dict:
        text = _strip_fences(raw).strip()
        # Find first JSON object in the response (LLM may wrap in prose).
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "architecture_name": "parse_failed",
                "architecture_family": None,
                "description": raw[:200],
                "reasoning": "LLM returned non-JSON",
            }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


# Match ```python … ``` (or just ``` … ```) anywhere in the response.
# We grab the FIRST fenced block — when the LLM wraps its prose with
# code in the middle, we want the code, not the prose.
_FENCED_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_MODEL_BLOCK_RE = re.compile(
    r"(?s)(^### MODEL ###\s*\n)(.*?)(\n^### END MODEL ###)",
    re.MULTILINE,
)
_TORCH_HUB_LOAD_RE = re.compile(
    r"torch\.hub\.load\(\s*['\"][^'\"]+['\"]\s*,\s*['\"]([A-Za-z0-9_]+)['\"](?P<rest>\s*,[^)]*)?\)",
)
_TORCHVISION_MODEL_ALLOWLIST = frozenset({
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet_b0",
    "efficientnet_b1",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
})


def _strip_fences(text: str) -> str:
    """Extract Python code from an LLM response.

    Three cases:
      1. The response contains a fenced ```python``` block somewhere
         — we return its contents.
      2. The response IS already a code block (starts with ```) — we
         peel the outer fence.
      3. The response is bare code (or bare prose; the validator will
         reject the latter on the next round-trip).
    """
    # Case 1: any fenced block anywhere — return the first one.
    m = _FENCED_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # Case 2: outer fence only (no closing on its own line).
    stripped = text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl > 0:
            stripped = stripped[first_nl + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _extract_build_model_block(text: str, *, fn_name: str = "build_model") -> str | None:
    """Extract just the ``fn_name`` function from an LLM response."""
    code = _strip_fences(text).strip()
    if not code:
        return None

    # If the model returned a whole skeleton/script, prefer the MODEL block.
    m = _MODEL_BLOCK_RE.search(code)
    if m:
        block = m.group(2).strip()
        return block or None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            segment = ast.get_source_segment(code, node)
            if segment:
                return segment.strip()
            lines = code.splitlines()
            start = max(0, node.lineno - 1)
            end = node.end_lineno if node.end_lineno is not None else start + 1
            return "\n".join(lines[start:end]).strip()
    return None


def _inject_model_block(template_code: str, model_block: str) -> str | None:
    """Replace content between `### MODEL ###` markers."""
    m = _MODEL_BLOCK_RE.search(template_code)
    if not m:
        return None
    block = model_block.strip()
    if not block:
        return None
    return (
        template_code[:m.start(2)]
        + block
        + template_code[m.end(2):]
    )


def _compose_code_from_model_response(
    raw: str,
    *,
    template_code: str,
    fn_name: str = "build_model",
    first_arg_name: str | None = None,
) -> str:
    """Compose runnable script by injecting the model block into the skeleton.

    Contract: the LLM must return ONLY ``fn_name``. We (a) extract that one
    function from the response, (b) validate it via ``validate_model_block``,
    and (c) inject it between the skeleton's ``### MODEL ###`` markers. Any
    deviation (extra classes, top-level imports, wrong first arg) is rejected
    here so the orchestrator's validator gets a clean reason to trigger
    recovery instead of executing whatever the LLM tried to smuggle in.

    If extraction/validation fails, we fall back to the raw stripped response
    so the downstream validator produces a specific SyntaxError / missing-fn
    message.
    """
    model_block = _extract_build_model_block(raw, fn_name=fn_name)
    if model_block:
        block_check = validator_mod.validate_model_block(
            model_block,
            fn_name=fn_name,
            first_arg_name=first_arg_name,
        )
        if block_check.ok:
            injected = _inject_model_block(template_code, model_block)
            if injected is not None:
                return injected
        else:
            logger.warning(
                "Model block rejected (%s): %s",
                block_check.error_type,
                block_check.message,
            )
    return _strip_fences(raw)


def _insert_local_import_into_build_model(code: str, import_stmt: str) -> str | None:
    """Insert `import_stmt` at top of build_model body when missing."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    build_fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_model"),
        None,
    )
    if build_fn is None:
        return None
    start = build_fn.lineno - 1
    end = (build_fn.end_lineno or build_fn.lineno) - 1
    lines = code.splitlines()
    block_text = "\n".join(lines[start:end + 1])
    if import_stmt in block_text:
        return code

    indent = 4
    if build_fn.body:
        indent = max(4, build_fn.body[0].col_offset)
    line = (" " * indent) + import_stmt

    insert_at = build_fn.lineno
    if (
        build_fn.body
        and isinstance(build_fn.body[0], ast.Expr)
        and isinstance(build_fn.body[0].value, ast.Constant)
        and isinstance(build_fn.body[0].value.value, str)
    ):
        insert_at = build_fn.body[0].end_lineno or build_fn.body[0].lineno

    if insert_at < 0 or insert_at > len(lines):
        return None
    lines.insert(insert_at, line)
    out = "\n".join(lines)
    if code.endswith("\n"):
        out += "\n"
    return out


def _apply_validator_autofix(code: str, result: validator_mod.ValidationResult) -> str | None:
    """Deterministic fixes for common validator failures.

    Keep this conservative: only apply mechanical, low-risk rewrites that we
    can validate immediately on the next loop iteration.
    """
    msg = result.message or ""
    if result.error_type == "DryRunFailed":
        if "NameError: name 'torchvision' is not defined" in msg:
            return _insert_local_import_into_build_model(code, "import torchvision")
        if "NameError: name 'timm' is not defined" in msg:
            return _insert_local_import_into_build_model(code, "import timm")
    if result.error_type == "ForbiddenModelSource":
        rewritten = _rewrite_torch_hub_load_to_torchvision(code)
        if rewritten is not None and rewritten != code:
            rewritten = _insert_local_import_into_build_model(
                rewritten,
                "import torchvision",
            ) or rewritten
            return rewritten
    return None


def _rewrite_torch_hub_load_to_torchvision(code: str) -> str | None:
    """Rewrite simple `torch.hub.load(..., '<model>', ...)` calls.

    We only rewrite known torchvision model names to avoid creating
    unsupported constructors.
    """
    changed = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal changed
        model_name = match.group(1)
        if model_name not in _TORCHVISION_MODEL_ALLOWLIST:
            return match.group(0)
        changed = True
        rest = (match.group("rest") or "").strip()
        if rest.startswith(","):
            rest = rest[1:].strip()
        args = rest
        if args:
            return f"torchvision.models.{model_name}({args})"
        return f"torchvision.models.{model_name}()"

    out = _TORCH_HUB_LOAD_RE.sub(_replace, code)
    return out if changed else None


def _resolve_validator_smoke_input_shape(adapter: TaskAdapter) -> tuple[int, ...] | None:
    """Best-effort input shape for validator forward smoke checks.

    Keep this conservative: we currently only enable it for audio tensor tasks
    where a float `(batch, C, H, W)` dummy input is unambiguous.
    """
    if adapter.kind != "audio_multilabel":
        return None
    raw = adapter.task_cfg.get("model", {}).get("input_shape")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    try:
        dims = tuple(int(d) for d in raw)
    except (TypeError, ValueError):
        return None
    if any(d <= 0 for d in dims):
        return None
    return dims


def _resolve_validator_smoke_num_classes(profile) -> int | None:
    """Return class-count hint for build_model(num_classes) smoke calls."""
    if profile is None:
        return None
    n = getattr(profile, "num_classes", None)
    if isinstance(n, int) and n > 0:
        return n
    return None


def _extract_metric_from_dict(obj: object, metric: str) -> float | None:
    if not isinstance(obj, dict):
        return None
    v = obj.get(metric)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _extract_metric_from_history(history: object, metric: str) -> float | None:
    if not isinstance(history, list):
        return None
    best: float | None = None
    for row in history:
        if not isinstance(row, dict):
            continue
        v = row.get(metric)
        if not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if best is None or fv > best:
            best = fv
    return best


def _build_executor(
    *,
    backend: str,
    settings: Settings,
    sandbox_root: Path,
    training_env: dict[str, str],
) -> Executor:
    """Factory for the executor named in ``backend``.

    Unknown backends fall back to local with a warning so a typo in
    the UI dropdown never blocks a study from starting.
    """
    timeout = settings.compute_budget.max_experiment_seconds
    if backend == "kaggle":
        from lab.core.kaggle_executor import KaggleExecutor
        k = settings.executor.kaggle
        return KaggleExecutor(
            username=k.username,
            kernel_prefix=k.kernel_prefix,
            enable_gpu=k.enable_gpu,
            enable_internet=k.enable_internet,
            accelerator=k.accelerator,
            poll_interval_seconds=k.poll_interval_seconds,
            log_poll_interval_seconds=k.log_poll_interval_seconds,
            poll_timeout_seconds=min(k.poll_timeout_seconds, timeout * 10) or timeout,
            dataset_sources=list(k.dataset_sources),
            competition_sources=list(k.competition_sources),
            weights_dataset=k.weights_dataset,
            sandbox_root=sandbox_root,
            repo_root=settings.repo_root,
            training_env=training_env,
        )
    if backend == "modal":
        from lab.core.modal_executor import ModalExecutor
        m = settings.executor.modal
        return ModalExecutor(
            app_name=m.app_name,
            image_name=m.image_name,
            pip_packages=list(m.pip_packages),
            gpu=m.gpu,
            cpu=m.cpu,
            memory_mb=m.memory_mb,
            timeout_seconds=min(m.timeout_seconds, timeout * 10) or timeout,
            retries=max(0, int(m.retries)),
            output_volume_name=m.output_volume_name,
            output_mount_path=m.output_mount_path,
            s3_bucket_name=m.s3_bucket_name,
            s3_secret_name=m.s3_secret_name,
            s3_endpoint_url=m.s3_endpoint_url,
            s3_mount_path=m.s3_mount_path,
            s3_key_prefix=m.s3_key_prefix,
            set_processed_dir_from_s3=m.set_processed_dir_from_s3,
            processed_subpath=m.processed_subpath,
            sandbox_root=sandbox_root,
            repo_root=settings.repo_root,
            training_env=training_env,
            env_prefix=settings.env_prefix,
        )
    if backend != "local":
        logger.warning("Unknown executor backend %r — falling back to local", backend)
    return LocalExecutor(
        sandbox_root=sandbox_root,
        timeout_seconds=timeout,
        repo_root=settings.repo_root,
        training_env=training_env,
    )


_LR_MIN = 1e-5
_LR_MAX = 1e-1
_MAX_PROPOSAL_EPOCHS = 10
_VALID_LR_SCHEDULES = frozenset({"constant", "cosine", "onecycle"})


def _proposal_hyperparam_env(
    proposal: dict | None,
    *,
    env_prefix: str,
) -> dict[str, str]:
    """Translate the LLM's proposal hyperparameters into ``AGENT_*`` env vars.

    Silently drops invalid values — the skeleton has its own clamping, so
    the worst case is a run with the default LR. We don't want an out-of-
    range LR to fail the whole experiment.
    """
    if not isinstance(proposal, dict):
        return {}
    out: dict[str, str] = {}
    lr = proposal.get("lr")
    if isinstance(lr, (int, float)) and _LR_MIN <= float(lr) <= _LR_MAX:
        out[f"{env_prefix}_LR"] = repr(float(lr))
    sched = proposal.get("lr_schedule")
    if isinstance(sched, str) and sched.strip().lower() in _VALID_LR_SCHEDULES:
        out[f"{env_prefix}_LR_SCHEDULE"] = sched.strip().lower()
    epochs = proposal.get("epochs")
    if isinstance(epochs, int) and 1 <= epochs <= _MAX_PROPOSAL_EPOCHS:
        out[f"{env_prefix}_EPOCHS"] = str(epochs)
    init_exp = proposal.get("init_from_experiment_id")
    if isinstance(init_exp, str) and init_exp.strip():
        out[f"{env_prefix}_INIT_FROM_EXPERIMENT_ID"] = init_exp.strip()
    return out


__all__ = [
    "Orchestrator",
    "OrchestratorHooks",
    "_strip_fences",
    "_extract_build_model_block",
    "_inject_model_block",
    "_compose_code_from_model_response",
    "_proposal_hyperparam_env",
]
