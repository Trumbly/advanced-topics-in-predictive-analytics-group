"""Single experiment iteration (ADR-005, ADR-018).

Stages:
1. propose -> Proposal (one retry on JSON parse fail)
2. generate -> code
3. validate + retry (auto-fix or LLM re-prompt; layer 1 of two-layer self-repair)
4. execute + retry on transient (layer 2)
5. capture + judge
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from lab.config import Settings
from lab.core import telemetry
from lab.core.executor import LocalExecutor
from lab.core.judge import Judge
from lab.core.llm import LLMClient
from lab.core.memory import Memory
from lab.core.models import (
    Experiment,
    ExecutionResult,
    Proposal,
    Task,
    TaskError,
    ValidationResult,
    Verdict,
)
from lab.core.parsing import ProposalParseError, parse_proposal
from lab.core.recovery import _HARD_FAILURE_ERROR_TYPES, Recovery
from lab.core.validator import Validator
from lab.prompts.engine import PromptEngine
from lab.tasks.base import TaskAdapter
from lab.tasks.skeleton import render_skeleton, splice_build_model


@dataclass
class Hooks:
    on_experiment_start: Callable[[Experiment], None] | None = None
    on_experiment_end: Callable[[Experiment], None] | None = None


@dataclass
class RunContext:
    settings: Settings
    adapter: TaskAdapter
    client: LLMClient
    engine: PromptEngine
    memory: Memory
    validator: Validator
    executor: LocalExecutor
    recovery: Recovery
    judge: Judge
    eda_summary: str = ""
    hooks: Hooks | None = None


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def run_experiment(exp: Experiment, ctx: RunContext) -> Experiment:
    telemetry.set_experiment(exp.id)
    telemetry.log_event("experiment_start", index=exp.index, id=exp.id)
    if ctx.hooks and ctx.hooks.on_experiment_start:
        ctx.hooks.on_experiment_start(exp)

    try:
        proposal = _propose(ctx, exp)
        exp.proposal = proposal
        exp.status = "GENERATING"
        code = _generate(ctx, exp, proposal)
        exp.status = "VALIDATING"
        validated_code = _validate_with_retry(ctx, exp, code)
        exp.code = validated_code
        exp.status = "EXECUTING"
        exec_result = _execute_with_retry(ctx, exp, validated_code, proposal)
        _capture(exp, exec_result)
        exp.status = "JUDGED"
        try:
            verdict = ctx.judge.judge_experiment(exp, ctx.memory)
            exp.verdict = verdict
        except Exception as exc:  # pragma: no cover - defensive
            telemetry.log_event("error", phase="judge", message=str(exc), level="error")
        ctx.memory.add(exp)
    except _HardFailure as hf:
        exp.status = "FAILED"
        exp.tasks.append(
            Task(
                name=hf.task_name,
                status="FAILED",
                error=hf.error,
            )
        )
        ctx.memory.add(exp)
        telemetry.log_event(
            "experiment_end", id=exp.id, succeeded=False, level="error"
        )
        if ctx.hooks and ctx.hooks.on_experiment_end:
            ctx.hooks.on_experiment_end(exp)
        return exp

    telemetry.log_event(
        "experiment_end", id=exp.id, succeeded=True, score=exp.primary_score
    )
    if ctx.hooks and ctx.hooks.on_experiment_end:
        ctx.hooks.on_experiment_end(exp)
    return exp


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


class _HardFailure(Exception):
    def __init__(self, task_name: str, error: TaskError):
        self.task_name = task_name
        self.error = error


def _propose(ctx: RunContext, exp: Experiment) -> Proposal:
    slots = _propose_slots(ctx)
    sys_prompt, user_prompt = ctx.engine.render("propose_architecture", slots)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = ctx.client.chat(messages)
    try:
        proposal = parse_proposal(
            raw, retry_client=ctx.client, retry_messages=messages + [{"role": "assistant", "content": raw}]
        )
    except ProposalParseError as exc:
        raise _HardFailure(
            task_name="propose",
            error=TaskError(error_type="Other", message=f"proposal parse: {exc}"),
        )
    proposal = _clamp_proposal(proposal, ctx.settings)
    exp.tasks.append(
        Task(name="propose", status="SUCCEEDED", input={}, output={"proposal": proposal.model_dump()})
    )
    return proposal


def _generate(ctx: RunContext, exp: Experiment, proposal: Proposal) -> str:
    slots = _propose_slots(ctx)
    slots["proposal"] = proposal.model_dump_json(indent=2)
    fn_name, arg_name = ctx.adapter.model_block_signature()
    slots["model_block_signature"] = f"def {fn_name}({arg_name}: int)"
    sys_prompt, user_prompt = ctx.engine.render("generate_code", slots)
    raw = ctx.client.chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    rendered_skeleton = render_skeleton(ctx.settings)
    return splice_build_model(rendered_skeleton, _extract_block(raw))


def _validate_with_retry(ctx: RunContext, exp: Experiment, code: str) -> str:
    fn, arg = ctx.adapter.model_block_signature()
    smoke_shape = tuple(ctx.settings.task.input_tensor_shape)
    smoke_classes = ctx.settings.task.expected_num_classes
    attempts = ctx.settings.compute_budget.max_codegen_retries
    current = code
    last: ValidationResult | None = None
    for attempt in range(attempts + 1):
        result = ctx.validator.validate(
            current,
            signature=(fn, arg),
            smoke_input_shape=smoke_shape,
            smoke_num_classes=smoke_classes,
        )
        last = result
        telemetry.log_event(
            "validate", attempt=attempt, ok=result.ok, error_type=result.error_type
        )
        # Persist a Task per attempt so the UI can replay the retry chain.
        exp.tasks.append(
            Task(
                name="validate",
                status="SUCCEEDED" if result.ok else "FAILED",
                input={"attempt": attempt},
                output={"ok": result.ok, "error_type": result.error_type},
                error=(
                    None
                    if result.ok
                    else TaskError(
                        error_type=result.error_type or "Other",
                        message=result.message or "",
                        autofix_hint=result.autofix_hint,
                    )
                ),
            )
        )
        if result.ok:
            return current
        autofixed = ctx.recovery.try_autofix(current, result)
        if autofixed is not None:
            exp.tasks.append(
                Task(
                    name="recover",
                    status="SUCCEEDED",
                    input={"attempt": attempt, "kind": "autofix"},
                    output={"hint": result.autofix_hint or ""},
                )
            )
            current = autofixed
            continue
        if attempt >= attempts:
            break
        slots = _propose_slots(ctx)
        slots["model_block_signature"] = f"def {fn}({arg}: int)"
        rendered = ctx.recovery.ask_llm(current, result, slots=slots)
        exp.tasks.append(
            Task(
                name="recover",
                status="SUCCEEDED",
                input={"attempt": attempt, "kind": "llm_reprompt"},
                output={},
            )
        )
        current = splice_build_model(render_skeleton(ctx.settings), _extract_block(rendered))

    raise _HardFailure(
        task_name="validate",
        error=TaskError(
            error_type=last.error_type or "Other" if last else "Other",
            message=(last.message if last else "no validator result"),
            autofix_hint=last.autofix_hint if last else None,
        ),
    )


def _execute_with_retry(
    ctx: RunContext, exp: Experiment, code: str, proposal: Proposal
) -> ExecutionResult:
    attempts = ctx.settings.compute_budget.max_recovery_attempts
    timeout = ctx.settings.compute_budget.max_experiment_seconds
    epochs_clamped = min(proposal.epochs, ctx.settings.compute_budget.max_epochs_per_run)

    # The subprocess runs with cwd=sandbox/<exp_id>, so every path env var
    # must be ABSOLUTE — otherwise relative paths from the config resolve
    # inside the sandbox dir and fail with FileNotFoundError.
    extra_env = {
        "AGENT_DEVICE": "cpu",
        "AGENT_BATCH_SIZE": "8",
        "AGENT_EPOCHS": str(epochs_clamped),
        "AGENT_PROCESSED_DIR": str(
            Path(ctx.settings.task.processed_data_dir).resolve()
        ),
        "AGENT_CHECKPOINT_OUT": str(
            (Path(ctx.settings.paths.sandbox) / exp.id / "checkpoint.pt").resolve()
        ),
        "AGENT_SEED": "42",
        "AGENT_LR": str(proposal.lr),
        "AGENT_LR_SCHEDULE": proposal.lr_schedule,
        "AGENT_WEIGHT_DECAY": str(proposal.weight_decay if proposal.weight_decay is not None else 0.0),
    }
    warm_start = _resolve_warm_start(ctx, proposal)
    if warm_start is not None:
        extra_env["AGENT_CHECKPOINT_IN"] = str(Path(warm_start).resolve())

    current = code
    last: ExecutionResult | None = None
    for attempt in range(attempts + 1):
        result = ctx.executor.run(
            current,
            experiment_id=exp.id,
            extra_env=extra_env,
            timeout_s=timeout,
        )
        last = result
        telemetry.log_event(
            "execute",
            attempt=attempt,
            succeeded=result.succeeded,
            error_type=(result.error.error_type if result.error else None),
            duration_s=result.duration_seconds,
        )
        # Persist a Task per attempt so the retry chain shows up in the UI.
        exp.tasks.append(
            Task(
                name="execute",
                status="SUCCEEDED" if result.succeeded else "FAILED",
                input={"attempt": attempt},
                output={
                    "succeeded": result.succeeded,
                    "duration_s": result.duration_seconds,
                    "primary_score": result.primary_score,
                },
                error=result.error,
            )
        )
        if result.succeeded:
            exp.sandbox_path = str(Path(ctx.settings.paths.sandbox) / exp.id)
            exp.checkpoint_path = extra_env["AGENT_CHECKPOINT_OUT"]
            return result

        if result.error and result.error.error_type in _HARD_FAILURE_ERROR_TYPES:
            raise _HardFailure(task_name="execute", error=result.error)

        if attempt >= attempts:
            raise _HardFailure(
                task_name="execute",
                error=result.error
                or TaskError(error_type="Other", message="execution failed"),
            )

        slots = _propose_slots(ctx)
        fn, arg = ctx.adapter.model_block_signature()
        slots["model_block_signature"] = f"def {fn}({arg}: int)"
        repaired = ctx.recovery.ask_llm(current, result.error, slots=slots) if result.error else None
        exp.tasks.append(
            Task(
                name="recover",
                status="SUCCEEDED" if repaired else "FAILED",
                input={"attempt": attempt, "kind": "llm_reprompt_after_execute"},
                output={},
            )
        )
        if repaired is None:
            raise _HardFailure(task_name="execute", error=result.error or TaskError(error_type="Other", message="no error"))
        current = splice_build_model(render_skeleton(ctx.settings), _extract_block(repaired))

    assert last is not None
    return last  # pragma: no cover


def _resolve_warm_start(ctx: RunContext, proposal: Proposal) -> str | None:
    """Return the AGENT_CHECKPOINT_IN value, gated by ADR-007 and predecessor memory."""
    if not ctx.settings.agent.memory_enabled:
        return None
    if not proposal.init_from_experiment_id:
        return None
    for w in ctx.memory.wins:
        if w.id == proposal.init_from_experiment_id and w.checkpoint_path:
            return str(w.checkpoint_path)
    telemetry.log_event(
        "recover",
        level="warn",
        message=f"warm-start checkpoint missing for {proposal.init_from_experiment_id}; running cold",
    )
    return None


def _capture(exp: Experiment, result: ExecutionResult) -> None:
    exp.primary_score = result.primary_score
    exp.metrics = dict(result.metrics)
    exp.history = list(result.history)
    exp.duration_seconds = result.duration_seconds
    # _execute_with_retry already appended a SUCCEEDED execute Task; nothing
    # else to record here.


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _propose_slots(ctx: RunContext) -> dict[str, str]:
    base = ctx.adapter.prompt_slots()
    base["personality"] = ctx.settings.agent.personality
    base["allow_transfer_learning"] = "true" if ctx.settings.agent.memory_enabled else "false"
    base["experiment_memory"] = ctx.memory.to_markdown()
    base["eda_summary"] = ctx.eda_summary or "(EDA not available)"
    return base


def _clamp_proposal(p: Proposal, settings: Settings) -> Proposal:
    cb = settings.compute_budget
    return p.model_copy(
        update={
            "epochs": max(1, min(p.epochs, cb.max_epochs_per_run)),
            "lr": min(max(p.lr, cb.lr_min), cb.lr_max),
        }
    )


_BUILD_START = "# --- AGENT_BUILD_MODEL_START ---"
_BUILD_END = "# --- AGENT_BUILD_MODEL_END ---"


def _extract_block(raw: str) -> str:
    """Pull the AGENT_BUILD_MODEL block out of an LLM reply."""
    if _BUILD_START in raw and _BUILD_END in raw:
        _, _, rest = raw.partition(_BUILD_START)
        block, _, _ = rest.partition(_BUILD_END)
        return block.strip()
    # Tolerant fallback: strip code fences and use whole body.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            if inner.startswith("python\n"):
                inner = inner[len("python\n") :]
            return inner.strip()
    return cleaned
