"""Orchestrator — the main agent loop.

Drives the Propose → Generate → Validate → Execute → Capture → Analyze cycle
for each experiment in a Study. Respects compute budget + stopping criteria.

Responsibilities
----------------
1. Load the pipeline YAML referenced by the Study.
2. For each experiment slot (up to `compute_budget.max_experiments`):
    a. Create a fresh Experiment object (fresh ID, timestamps).
    b. Walk the pipeline steps in order, executing each as a Task:
        - LLM tasks: ContextHandler fills the template → LLMClient.chat()
        - Predefined tasks: dispatch to `agent.handlers.<module>.run()`
    c. Mutate the Experiment based on task outputs (architecture, code,
       results, submission path).
    d. Append to memory + write logs + check stopping criteria.
3. Update Study.best_experiment_id / best_score whenever we improve.
4. Persist the final Study state.

Design constraints
------------------
- The orchestrator does NOT know how the LLM or the sandbox work — it
  just calls the injected dependencies. This keeps it trivially testable
  with mocks.
- Every task goes through the logger so nothing is ever lost.
- Stopping criteria: max experiments, wallclock budget, explicit abort.
  Score-plateau detection is intentionally out of scope for v1 (can be
  added as a stopping criterion later).
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from agent.context_handler import ContextHandler
from agent.executor import CodeExecutor
from agent.llm_client import LLMClient, LLMError
from agent.logger import ExperimentLogger
from agent.memory import ExperimentMemory
from agent.models import (
    Experiment,
    ExperimentStatus,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    Study,
    StudyStatus,
    Task,
    TaskError,
    TaskStatus,
    TaskType,
    TrainingResults,
)
from agent.prompt_engine import PromptEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stop reason enumeration (for the return value of run())
# ---------------------------------------------------------------------------


class StopReason:
    MAX_EXPERIMENTS = "max_experiments"
    WALLCLOCK = "wallclock_exceeded"
    USER_ABORT = "user_abort"
    NO_PIPELINE_STEPS = "no_pipeline_steps"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class Orchestrator:
    """Main agent loop. All dependencies are injected for testability."""

    study: Study
    llm_client: LLMClient
    prompt_engine: PromptEngine
    context_handler: ContextHandler
    memory: ExperimentMemory
    experiment_logger: ExperimentLogger
    executor: CodeExecutor

    pipeline: PipelineDefinition = field(init=False)
    _handler_cache: dict[str, Callable[..., Task]] = field(default_factory=dict)
    _start_wallclock: float = field(default=0.0)
    _abort_requested: bool = field(default=False)

    # -- lifecycle ----------------------------------------------------------

    def __post_init__(self) -> None:
        self.pipeline = _load_pipeline(self.study.pipeline_config_path)

    def request_abort(self) -> None:
        """Ask the loop to stop after the current experiment finishes."""
        self._abort_requested = True

    # -- main loop ----------------------------------------------------------

    def run(self) -> tuple[Study, str]:
        """Run the loop until a stopping criterion fires.

        Returns `(study, stop_reason)`. The study is persisted on every
        experiment completion so a crash mid-loop doesn't lose work.
        """
        self._start_wallclock = time.monotonic()

        logger.info("=" * 66)
        logger.info("Starting study: %s", self.study.name)
        logger.info("  id:       %s", self.study.study_id)
        logger.info("  pipeline: %s", self.study.pipeline_config_path)
        logger.info("  model:    %s", self.llm_client.model)
        logger.info(
            "  budget:   %d experiments, %d min wallclock",
            self.study.compute_budget.max_experiments,
            self.study.compute_budget.max_wallclock_minutes,
        )
        logger.info("  steps:    %s", [s.task_name for s in self.pipeline.steps])
        logger.info("=" * 66)

        if not self.pipeline.steps:
            self.study.status = StudyStatus.ABORTED
            self._save_study()
            return self.study, StopReason.NO_PIPELINE_STEPS

        stop_reason: str = StopReason.MAX_EXPERIMENTS
        max_exp = self.study.compute_budget.max_experiments

        for i in range(max_exp):
            if self._abort_requested:
                stop_reason = StopReason.USER_ABORT
                break
            if self._wallclock_exceeded():
                stop_reason = StopReason.WALLCLOCK
                break

            logger.info("")
            logger.info("──── Experiment %d/%d ────", i + 1, max_exp)
            experiment = self._make_experiment()
            self._run_experiment(experiment)
            self._update_best(experiment)
            self._save_study()

        logger.info("")
        logger.info("=" * 66)
        logger.info("Study finished: %s", stop_reason)
        logger.info("  experiments run: %d", len(self.study.experiment_ids))
        if self.study.best_experiment_id:
            logger.info(
                "  best: %s @ %s=%.4f",
                self.study.best_experiment_id,
                self.memory.score_metric,
                self.study.best_score or 0.0,
            )
        else:
            logger.info("  best: (none — no successful experiments)")
        logger.info("=" * 66)

        if self._abort_requested:
            self.study.status = StudyStatus.ABORTED
        else:
            self.study.status = StudyStatus.COMPLETED
        self._save_study()
        return self.study, stop_reason

    # -- per-experiment -----------------------------------------------------

    def _make_experiment(self) -> Experiment:
        """Create a fresh Experiment and register it on the Study."""
        idx = len(self.study.experiment_ids) + 1
        experiment_id = f"exp_{idx:03d}"
        now = _now()
        experiment = Experiment(
            experiment_id=experiment_id,
            study_id=self.study.study_id,
            llm_model=self.llm_client.model,
            status=ExperimentStatus.RUNNING,
            created_at=now,
            started_at=now,
        )
        self.study.experiment_ids.append(experiment_id)
        self.study.updated_at = now
        return experiment

    def _run_experiment(self, experiment: Experiment) -> None:
        """Execute every pipeline step for one experiment."""
        # Full output history keyed by task_name. Unlike "previous_task_output"
        # (which only holds the *last* task's output), this lets downstream
        # handlers look up specific earlier tasks by name, e.g.
        # `outputs["generate_code"]["code"]` even when `validate_code`
        # runs in between.
        outputs_by_name: dict[str, dict[str, Any]] = {}
        previous_task_output: dict[str, Any] = {}
        task_counter = 0

        for step in self.pipeline.steps:
            task_counter += 1
            task = Task(
                task_id=f"{experiment.experiment_id}_task_{task_counter:02d}_{step.task_name}",
                experiment_id=experiment.experiment_id,
                task_type=step.task_type,
                task_name=step.task_name,
            )
            experiment.task_ids.append(task.task_id)

            logger.info(
                "  [%s] %s (%s) ...",
                experiment.experiment_id,
                step.task_name,
                step.task_type.value if hasattr(step.task_type, "value") else step.task_type,
            )
            task_start = time.monotonic()

            try:
                if step.task_type == TaskType.LLM.value or step.task_type == TaskType.LLM:
                    self._run_llm_task(task, step, experiment, previous_task_output)
                else:
                    self._run_predefined_task(
                        task,
                        step,
                        experiment,
                        previous_task_output,
                        outputs_by_name,
                    )
            except Exception as exc:  # noqa: BLE001 — task failures must not kill the loop
                logger.exception("Task %s crashed", task.task_id)
                task.status = TaskStatus.FAILED
                task.error = TaskError(
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                task.completed_at = _now()

            task_duration = time.monotonic() - task_start
            self._log_task_outcome(task, task_duration)

            self.experiment_logger.write_task(task)

            # Lift relevant outputs onto the Experiment
            self._apply_task_output_to_experiment(task, experiment)
            previous_task_output = dict(task.output)
            outputs_by_name[step.task_name] = dict(task.output)

            # Stop the pipeline for this experiment on first hard failure
            if task.status in (
                TaskStatus.FAILED.value,
                TaskStatus.TIMEOUT.value,
                TaskStatus.FAILED,
                TaskStatus.TIMEOUT,
            ):
                self._mark_experiment_failed(experiment, task)
                self.experiment_logger.write_experiment(experiment)
                self.memory.append(experiment)
                logger.info(
                    "  [%s] FAILED after %s",
                    experiment.experiment_id,
                    step.task_name,
                )
                return

        # All steps succeeded
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = _now()
        self.experiment_logger.write_experiment(experiment)
        self.memory.append(experiment)
        score = None
        if experiment.results:
            score = experiment.results.metrics.get(self.memory.score_metric)
        if score is not None:
            logger.info(
                "  [%s] COMPLETED | %s=%.4f",
                experiment.experiment_id,
                self.memory.score_metric,
                score,
            )
        else:
            logger.info("  [%s] COMPLETED (no score)", experiment.experiment_id)

    # -- task logging -------------------------------------------------------

    def _log_task_outcome(self, task: Task, duration: float) -> None:
        """Print a one-line summary of what a task just did."""
        status = task.status.value if hasattr(task.status, "value") else task.status

        if status == TaskStatus.COMPLETED.value:
            extra = self._task_success_extra(task)
            suffix = f" | {extra}" if extra else ""
            logger.info("    ✓ %s (%.1fs)%s", task.task_name, duration, suffix)
        else:
            err = task.error
            if err is not None:
                msg = err.message.splitlines()[0][:120]
                logger.info(
                    "    ✗ %s (%.1fs) | %s: %s",
                    task.task_name,
                    duration,
                    err.error_type,
                    msg,
                )
            else:
                logger.info(
                    "    ✗ %s (%.1fs) | status=%s",
                    task.task_name,
                    duration,
                    status,
                )

    def _task_success_extra(self, task: Task) -> str:
        """Return a short extra blurb describing what a successful task
        produced, for the progress log."""
        if task.task_name == "propose_architecture":
            cfg = task.output.get("architecture_proposal") or {}
            arch = cfg.get("architecture") if isinstance(cfg, dict) else None
            return f"arch={arch}" if arch else ""

        if task.task_name == "generate_code":
            code = task.output.get("code", "")
            if isinstance(code, str):
                return f"{len(code)} bytes of code"
            return ""

        if task.task_name == "execute_training":
            duration = task.output.get("duration_seconds")
            if isinstance(duration, (int, float)):
                return f"training took {duration:.1f}s"
            return ""

        if task.task_name == "capture_metrics":
            training = task.output.get("training_results")
            if isinstance(training, dict):
                metrics = training.get("metrics", {})
                if metrics:
                    bits = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
                    return ", ".join(bits[:3])
            return ""

        return ""

    # -- task runners -------------------------------------------------------

    def _run_llm_task(
        self,
        task: Task,
        step: PipelineStep,
        experiment: Experiment,
        previous_task_output: dict[str, Any],
    ) -> None:
        """Execute one LLM task: build prompt → call LLM → store response."""
        if step.prompt_template is None:
            raise ValueError(
                f"LLM task '{step.task_name}' is missing prompt_template in the pipeline YAML"
            )

        task.started_at = _now()
        task.status = TaskStatus.RUNNING

        template = self.prompt_engine.load(step.prompt_template)

        extra_slots = self._extra_slots_for_task(
            step.task_name, experiment, previous_task_output
        )
        system, user = self.context_handler.build(template, extra_slots=extra_slots)
        task.prompt_used = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"

        try:
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        except LLMError as exc:
            task.status = TaskStatus.FAILED
            task.error = TaskError(error_type="LLMError", message=str(exc))
            task.completed_at = _now()
            return

        task.llm_response = response
        task.output = self._parse_llm_response(step.task_name, response)
        task.status = TaskStatus.COMPLETED
        task.completed_at = _now()

    def _run_predefined_task(
        self,
        task: Task,
        step: PipelineStep,
        experiment: Experiment,
        previous_task_output: dict[str, Any],
        outputs_by_name: dict[str, dict[str, Any]],
    ) -> None:
        """Execute one predefined handler task."""
        if step.handler is None:
            raise ValueError(
                f"Predefined task '{step.task_name}' is missing handler in the pipeline YAML"
            )

        handler_fn = self._resolve_handler(step.handler)

        # Both validate_code and execute_training need the code produced by
        # the generate_code task. We look it up by name so intermediate
        # tasks (e.g. validate_code between generate_code and execute_training)
        # don't accidentally shadow the output.
        generated_code = _lookup_generated_code(outputs_by_name)

        if step.task_name == "execute_training":
            if not generated_code:
                task.status = TaskStatus.FAILED
                task.error = TaskError(
                    error_type="NoCode",
                    message=(
                        "execute_training could not find generated code from a "
                        "preceding generate_code task. Check the pipeline order."
                    ),
                )
                task.completed_at = _now()
                return
            task.code_used = generated_code
            handler_fn(
                task,
                config=step.config,
                executor=self.executor,
            )
            return

        if step.task_name == "validate_code":
            task.code_used = generated_code
            handler_fn(task, config=step.config)
            return

        if step.task_name == "capture_metrics":
            handler_fn(
                task,
                config=step.config,
                previous_task_output=previous_task_output,
            )
            return

        # Generic dispatch for future handlers
        kwargs = {"config": step.config}
        sig = inspect.signature(handler_fn)
        if "previous_task_output" in sig.parameters:
            kwargs["previous_task_output"] = previous_task_output
        if "executor" in sig.parameters:
            kwargs["executor"] = self.executor
        handler_fn(task, **kwargs)

    # -- task output → experiment --------------------------------------------

    def _apply_task_output_to_experiment(
        self, task: Task, experiment: Experiment
    ) -> None:
        """Lift structured outputs from a Task onto the parent Experiment."""
        if task.status not in (
            TaskStatus.COMPLETED.value,
            TaskStatus.COMPLETED,
        ):
            return

        if task.task_name == "propose_architecture":
            cfg_data = task.output.get("architecture_proposal") or task.output
            try:
                experiment.config = _parse_model_config(cfg_data)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to parse model config from propose_architecture: %s", exc
                )

        elif task.task_name == "capture_metrics":
            tr_data = task.output.get("training_results")
            if isinstance(tr_data, dict):
                try:
                    experiment.results = TrainingResults.model_validate(tr_data)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to parse training_results: %s", exc)

    def _mark_experiment_failed(self, experiment: Experiment, task: Task) -> None:
        """Set final experiment status based on the failed task."""
        if task.status in (TaskStatus.TIMEOUT.value, TaskStatus.TIMEOUT):
            experiment.status = ExperimentStatus.TIMEOUT
        else:
            experiment.status = ExperimentStatus.FAILED
        experiment.completed_at = _now()

    # -- slot / response helpers --------------------------------------------

    def _extra_slots_for_task(
        self,
        task_name: str,
        experiment: Experiment,
        previous_task_output: dict[str, Any],
    ) -> dict[str, Any]:
        """Per-task slots that aren't handled by the ContextHandler itself."""
        if task_name == "generate_code":
            proposal = previous_task_output.get("architecture_proposal") or (
                experiment.config.model_dump() if experiment.config else {}
            )
            return {"architecture_proposal": proposal}

        if task_name == "analyze_results":
            slots: dict[str, Any] = {
                "architecture": experiment.config.architecture
                if experiment.config
                else "(unknown)",
                "hyperparams": experiment.config.hyperparams
                if experiment.config
                else {},
                "augmentation": experiment.config.augmentation
                if experiment.config
                else {},
                "results": experiment.results.model_dump() if experiment.results else {},
            }
            best = self.memory.best()
            slots["previous_best"] = (
                best.model_dump()
                if best and best.experiment_id != experiment.experiment_id
                else "(none)"
            )
            return slots

        return {}

    def _parse_llm_response(self, task_name: str, response: str) -> dict[str, Any]:
        """Extract structured data from an LLM response.

        - propose_architecture → parse JSON object
        - generate_code → extract Python code (stripping markdown fences)
        - analyze_results → keep raw text in "analysis"
        """
        if task_name == "propose_architecture":
            try:
                return {"architecture_proposal": _parse_json_from_text(response)}
            except ValueError as exc:
                raise ValueError(
                    f"propose_architecture: LLM did not return valid JSON: {exc}"
                ) from exc

        if task_name == "generate_code":
            return {"code": _strip_code_fences(response)}

        if task_name == "analyze_results":
            return {"analysis": response.strip()}

        return {"raw": response}

    # -- pipeline wiring ----------------------------------------------------

    def _resolve_handler(self, dotted_path: str) -> Callable[..., Task]:
        """Resolve a dotted-path handler string to its `run` function."""
        if dotted_path in self._handler_cache:
            return self._handler_cache[dotted_path]

        module = importlib.import_module(dotted_path)
        if not hasattr(module, "run"):
            raise AttributeError(
                f"Handler module '{dotted_path}' must export a `run(task, ...)` function"
            )
        fn = module.run
        self._handler_cache[dotted_path] = fn
        return fn

    # -- study state --------------------------------------------------------

    def _save_study(self) -> None:
        self.study.updated_at = _now()
        self.experiment_logger.write_study(self.study)

    def _update_best(self, experiment: Experiment) -> None:
        if experiment.status != ExperimentStatus.COMPLETED.value and experiment.status != ExperimentStatus.COMPLETED:
            return
        if experiment.results is None:
            return
        score = experiment.results.metrics.get(self.memory.score_metric)
        if score is None:
            return
        if self.study.best_score is None or score > self.study.best_score:
            previous = self.study.best_score
            self.study.best_score = score
            self.study.best_experiment_id = experiment.experiment_id
            if previous is None:
                logger.info(
                    "  ⭐ new best: %s @ %s=%.4f",
                    experiment.experiment_id,
                    self.memory.score_metric,
                    score,
                )
            else:
                delta = score - previous
                logger.info(
                    "  ⭐ new best: %s @ %s=%.4f (+%.4f)",
                    experiment.experiment_id,
                    self.memory.score_metric,
                    score,
                    delta,
                )

    def _wallclock_exceeded(self) -> bool:
        elapsed_min = (time.monotonic() - self._start_wallclock) / 60.0
        return elapsed_min >= self.study.compute_budget.max_wallclock_minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_pipeline(path: Path) -> PipelineDefinition:
    raw = yaml.safe_load(Path(path).read_text())
    return PipelineDefinition.model_validate(raw)


def _lookup_generated_code(outputs_by_name: dict[str, dict[str, Any]]) -> str:
    """Return the Python code produced by the generate_code task, if any.

    We look up by task name so intermediate steps (validate_code between
    generate_code and execute_training) do not shadow the output.
    Returns an empty string if generate_code did not produce code.
    """
    gen = outputs_by_name.get("generate_code")
    if not gen:
        return ""
    code = gen.get("code", "")
    return code if isinstance(code, str) else ""


def _parse_model_config(data: Any) -> ModelConfig:
    """Be tolerant when parsing LLM output into a ModelConfig.

    The LLM may return extra fields or mislabeled types. We pick out the
    known fields and let Pydantic validate the rest.
    """
    if isinstance(data, ModelConfig):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict for ModelConfig, got {type(data).__name__}")
    known = {
        "architecture": data.get("architecture", "unknown"),
        "pretrained_model": data.get("pretrained_model"),
        "hyperparams": data.get("hyperparams", {}) or {},
        "augmentation": data.get("augmentation", {}) or {},
    }
    return ModelConfig.model_validate(known)


def _parse_json_from_text(text: str) -> dict[str, Any]:
    """Extract the first top-level JSON object from a string.

    Handles common LLM quirks: markdown fences, leading prose, trailing prose.
    """
    text = text.strip()
    # Strip ```json or ``` fences if present
    if text.startswith("```"):
        # Find matching end fence
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Locate the first '{' and the matching '}'
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise ValueError("Unbalanced braces in response")
    return json.loads(text[start : end + 1])


def _strip_code_fences(text: str) -> str:
    """Remove ```python / ``` fences the LLM may wrap code in."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first fence line
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip() + "\n"


def _now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["Orchestrator", "StopReason"]
