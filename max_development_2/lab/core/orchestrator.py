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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from lab.config import Settings
from lab.core import validator as validator_mod
from lab.core.context import ContextBuilder, load_model_registry
from lab.core.executor import CodeExecutor, ExecutionResult
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
    ):
        self.settings = settings
        self.adapter = adapter
        self.llm = llm or LLMClient(
            base_url=settings.llm.base_url,
            model=settings.llm.default_model,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout_seconds=settings.llm.timeout_seconds,
            retry_attempts=settings.llm.retry_attempts,
            retry_backoff_seconds=settings.llm.retry_backoff_seconds,
        )
        self.hooks = hooks or OrchestratorHooks()
        self.predecessor = predecessor

        self.registry = PromptRegistry(settings.abspath(settings.paths.prompts_dir))
        self.engine = PromptEngine(self.registry)

        self.memory = Memory(score_metric=adapter.primary_metric)
        if predecessor is not None:
            self.memory.seed_from_predecessor(predecessor)

        skeleton_path = settings.abspath(adapter.code_skeleton_path())
        skeleton_content = skeleton_path.read_text() if skeleton_path.exists() else ""
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

        sandbox_root = settings.abspath(settings.paths.sandbox)
        self.executor = CodeExecutor(
            sandbox_root=sandbox_root,
            timeout_seconds=settings.compute_budget.max_experiment_seconds,
            repo_root=settings.repo_root,
            training_env=self._training_env(),
        )

        # Graceful SIGINT handling — set by run()
        self._abort_requested = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_study(self, name: str = "", tags: list[str] | None = None) -> Study:
        study = Study(
            name=name or f"{self.adapter.name}-{_now().strftime('%Y%m%d_%H%M%S')}",
            task_name=self.adapter.name,
            primary_metric=self.adapter.primary_metric,
            predecessor_id=self.predecessor.id if self.predecessor else None,
            prompt_template_paths=self._resolve_prompt_paths(),
            publish=self.settings.publishing.default_publish,
            tags=tags or list(self.settings.publishing.default_tags),
        )
        study.status = StudyStatus.RUNNING
        study.started_at = _now()
        experiments_dir = self.settings.abspath(self.settings.paths.experiments)
        study.save(experiments_dir)
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
                    primary_metric=self.adapter.primary_metric,
                )
                logger.info("── experiment %d/%d: %s", i + 1, budget.max_experiments, exp.id)
                self._fire(self.hooks.on_experiment_start, exp)
                self._run_experiment(exp)
                self.memory.add(exp)
                self.memory.save(experiments_dir / study.id / "memory.json")

                study.experiments.append(exp)
                self._update_best(study)
                study.save(experiments_dir)
                self._fire(self.hooks.on_experiment_end, exp)

            study.status = StudyStatus.ABORTED if self._abort_requested else StudyStatus.COMPLETED
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
        logger.warning("SIGINT received — finishing current experiment then stopping.")
        self._abort_requested = True

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def _run_experiment(self, exp: Experiment) -> None:
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = _now()

        # 1) Propose architecture
        propose_task = Task(name="propose_architecture")
        try:
            proposal_text = self._propose_architecture()
            proposal_data = self._parse_proposal(proposal_text)
            exp.architecture_name = proposal_data.get("architecture_name", "unknown")
            exp.architecture_family = proposal_data.get("architecture_family")
            exp.architecture_proposal = json.dumps(proposal_data, indent=2)
            propose_task.output = proposal_data
            propose_task.status = TaskStatus.COMPLETED
        except Exception as exc:  # noqa: BLE001
            logger.exception("propose_architecture failed")
            propose_task.status = TaskStatus.FAILED
            propose_task.error = TaskError(error_type="LLMError", message=str(exc))
            exp.tasks.append(propose_task)
            exp.status = ExperimentStatus.FAILED
            exp.error = propose_task.error
            exp.completed_at = _now()
            return
        exp.tasks.append(propose_task)

        # 2) Generate code with Layer-1 retry (codegen + validator)
        gen_task = Task(name="generate_code")
        try:
            code = self._generate_code(exp.architecture_proposal or "")
        except LLMError as exc:
            gen_task.status = TaskStatus.FAILED
            gen_task.error = TaskError(error_type="LLMError", message=str(exc))
            exp.tasks.append(gen_task)
            exp.status = ExperimentStatus.FAILED
            exp.error = gen_task.error
            exp.completed_at = _now()
            return

        val_ok = False
        for attempt in range(1, self.settings.compute_budget.max_codegen_retries + 1):
            result = validator_mod.validate(
                code,
                extra_spawn_triggers=self.adapter.spawn_triggering_calls(),
            )
            if result.ok:
                val_ok = True
                break
            logger.warning("Validator rejected code (attempt %d): %s", attempt, result.message)
            # Regenerate via recover_from_error prompt
            try:
                code = self._recover_code(code, result.error_type, result.message, traceback="")
            except LLMError as exc:
                gen_task.status = TaskStatus.FAILED
                gen_task.error = TaskError(error_type="LLMError", message=str(exc))
                break

        gen_task.code_used = code
        exp.code = code
        if not val_ok:
            gen_task.status = TaskStatus.FAILED
            if gen_task.error is None:
                gen_task.error = TaskError(error_type="ValidationFailed", message="Codegen retries exhausted")
            exp.tasks.append(gen_task)
            exp.status = ExperimentStatus.FAILED
            exp.error = gen_task.error
            exp.completed_at = _now()
            return
        gen_task.status = TaskStatus.COMPLETED
        exp.tasks.append(gen_task)

        # 3) Execute with Layer-2 retry (runtime error recovery)
        exec_result = self._execute_with_recovery(code, exp)
        run_task = Task(
            name="execute_training",
            status=TaskStatus.COMPLETED if exec_result.succeeded else TaskStatus.FAILED,
            error=exec_result.error,
            output={
                "duration_seconds": exec_result.duration_seconds,
                "exit_code": exec_result.exit_code,
                "timed_out": exec_result.timed_out,
            },
        )
        exp.tasks.append(run_task)
        exp.sandbox_path = str(exec_result.workdir)
        exp.duration_seconds = exec_result.duration_seconds

        if not exec_result.succeeded:
            exp.status = ExperimentStatus.FAILED
            exp.error = exec_result.error
            exp.completed_at = _now()
            return

        # 4) Capture metrics
        metrics_task = Task(name="capture_metrics")
        try:
            metrics = self._capture_metrics(exec_result)
            exp.metrics = metrics.get("metrics", {})
            exp.history = metrics.get("history", [])
            exp.primary_score = metrics.get("primary_score")
            extra_errors = self.adapter.validate_training_output(metrics.get("raw", {}))
            if extra_errors:
                metrics_task.status = TaskStatus.FAILED
                metrics_task.error = TaskError(
                    error_type="TaskValidationFailed",
                    message="; ".join(extra_errors),
                )
                exp.status = ExperimentStatus.FAILED
                exp.error = metrics_task.error
            else:
                metrics_task.status = TaskStatus.COMPLETED
                exp.status = ExperimentStatus.COMPLETED
        except Exception as exc:  # noqa: BLE001
            metrics_task.status = TaskStatus.FAILED
            metrics_task.error = TaskError(error_type="MetricsError", message=str(exc))
            exp.status = ExperimentStatus.FAILED
            exp.error = metrics_task.error
        exp.tasks.append(metrics_task)
        exp.completed_at = _now()

    def _execute_with_recovery(self, code: str, exp: Experiment) -> ExecutionResult:
        max_attempts = self.settings.compute_budget.max_recovery_attempts
        for attempt in range(1, max_attempts + 1):
            result = self.executor.run(code, experiment_id=exp.id)
            if result.succeeded:
                return result
            if result.error is None or result.error.error_type in {"Timeout", "OOM"}:
                # Hard failures — stop retrying.
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
                )
                exp.code = code
            except LLMError as exc:
                logger.error("recover_from_error LLM call failed: %s", exc)
                return result
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # LLM calls — thin wrappers around prompt engine + llm client
    # ------------------------------------------------------------------

    def _propose_architecture(self) -> str:
        slots = self.context.build()
        system, user = self.engine.render("propose_architecture", slots)
        return self.llm.chat(format_messages(system, user))

    def _generate_code(self, architecture_proposal: str) -> str:
        slots = self.context.build(architecture_proposal=architecture_proposal)
        system, user = self.engine.render("generate_code", slots)
        raw = self.llm.chat(format_messages(system, user))
        return _strip_fences(raw)

    def _recover_code(self, code: str, error_type: str, error_message: str, traceback: str) -> str:
        slots = self.context.build(
            error_type=error_type,
            error_message=error_message,
            error_traceback=traceback,
            broken_code=code,
        )
        system, user = self.engine.render("recover_from_error", slots)
        raw = self.llm.chat(format_messages(system, user))
        return _strip_fences(raw)

    # ------------------------------------------------------------------
    # Metrics + helpers
    # ------------------------------------------------------------------

    def _capture_metrics(self, exec_result: ExecutionResult) -> dict:
        assert exec_result.results_json_path is not None
        raw = json.loads(exec_result.results_json_path.read_text())
        primary_metric = raw.get("primary_metric", self.adapter.primary_metric)
        primary_score = raw.get("primary_score")
        history = raw.get("history", [])
        final = raw.get("final", history[-1] if history else {})
        metrics: dict[str, float] = {}
        for k, v in final.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
        if primary_score is None and primary_metric in metrics:
            primary_score = metrics[primary_metric]
        return {
            "metrics": metrics,
            "history": history,
            "primary_score": primary_score,
            "raw": raw,
        }

    def _update_best(self, study: Study) -> None:
        ok = [e for e in study.experiments if e.primary_score is not None]
        if not ok:
            return
        best = max(ok, key=lambda e: e.primary_score)  # type: ignore[arg-type]
        study.best_experiment_id = best.id
        study.best_score = best.primary_score

    def _resolve_prompt_paths(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for task in ("propose_architecture", "generate_code", "analyze_result",
                     "recover_from_error", "executive_summary"):
            try:
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
        return env

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


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*|\s*```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    """Remove common ``` markdown fences the LLM sprinkles around code."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl > 0:
            stripped = stripped[first_nl + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


__all__ = ["Orchestrator", "OrchestratorHooks"]
