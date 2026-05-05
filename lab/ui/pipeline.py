"""Derive per-experiment pipeline state for the UI.

Each experiment moves through the same five phases::

    propose -> generate -> validate -> execute -> judge

with retries inside ``validate`` / ``execute``. The UI surfaces a fixed
five-pill row per experiment with each pill in one of these states:

- ``done``     -- phase completed successfully
- ``failed``   -- phase tried and the last attempt failed
- ``running``  -- phase currently in flight (only on the active experiment)
- ``pending``  -- phase not reached yet
"""

from __future__ import annotations

from typing import Iterable

from lab.core.models import Experiment, Study

PHASES: tuple[str, ...] = ("propose", "generate", "validate", "execute", "judge")

_TERMINAL_EXP_STATUSES = {"JUDGED", "COMPLETED", "FAILED", "ABORTED"}

# The orchestrator's exp.status -> the phase it is in the middle of.
_STATUS_TO_PHASE: dict[str, str] = {
    "PROPOSED": "propose",
    "GENERATING": "generate",
    "VALIDATING": "validate",
    "EXECUTING": "execute",
    "RECOVERING": "validate",
}


def derive_experiment_pipeline(
    exp: Experiment, *, is_active: bool = False
) -> list[tuple[str, str]]:
    """Return ``[(phase, state), ...]`` ordered by :data:`PHASES`.

    Past phases are read off the task list (last attempt's status); the
    *currently running* phase comes from the most recent task transition (so
    a validate FAILED followed by no recover yet routes back to ``generate``,
    matching the agent's actual behaviour where recover re-prompts the LLM
    for a new build_model block).
    """
    states: dict[str, str] = {p: "pending" for p in PHASES}

    if any(t.name == "propose" and t.status == "SUCCEEDED" for t in exp.tasks):
        states["propose"] = "done"

    if exp.code:
        states["generate"] = "done"

    val_tasks = [t for t in exp.tasks if t.name == "validate"]
    if val_tasks:
        last = val_tasks[-1]
        states["validate"] = "done" if last.status == "SUCCEEDED" else "failed"

    exec_tasks = [t for t in exp.tasks if t.name == "execute"]
    if exec_tasks:
        last = exec_tasks[-1]
        states["execute"] = "done" if last.status == "SUCCEEDED" else "failed"

    if exp.verdict is not None:
        states["judge"] = "done"

    if is_active and exp.status not in _TERMINAL_EXP_STATUSES:
        running = _running_phase(exp)
        if running is not None:
            states[running] = "running"

    return [(p, states[p]) for p in PHASES]


def _running_phase(exp: Experiment) -> str | None:
    """Pick the phase currently in flight.

    Order of evidence (strongest first):
      1. Most recent Task on the experiment — this captures retry transitions
         (a failed validate that has not yet been recovered routes the user
         back to ``generate``; an autofix routes back to ``validate``).
      2. ``exp.status`` set by the orchestrator at phase boundaries.
    """
    if exp.tasks:
        last = exp.tasks[-1]
        if last.name == "recover":
            kind = (last.input or {}).get("kind", "") if last.input else ""
            if "llm_reprompt" in kind:
                # Recover asked the LLM for a fresh build_model block: that
                # is generation work in progress.
                return "generate"
            # autofix is a deterministic in-place edit -> next thing is the
            # validator running again.
            return "validate"
        if last.name == "validate" and last.status != "SUCCEEDED":
            # Validate failed and the orchestrator is about to either autofix
            # or re-prompt the LLM. Either way, the next user-visible activity
            # is regenerating code.
            return "generate"
        if last.name == "execute" and last.status != "SUCCEEDED":
            # Execute failed; recover-after-execute will regenerate code.
            return "generate"
        if last.name == "propose" and last.status == "SUCCEEDED":
            return "generate"

    return _STATUS_TO_PHASE.get(exp.status)


def derive_study_pipelines(study: Study) -> dict[str, list[tuple[str, str]]]:
    """Return ``{experiment_id: pipeline_rows}`` for every experiment.

    The most-recent experiment on a non-terminal study is treated as the
    active one (gets the ``running`` highlight).
    """
    if not study.experiments:
        return {}
    last_idx = len(study.experiments) - 1
    study_active = study.status in {"RUNNING", "STARTED"}
    return {
        e.id: derive_experiment_pipeline(
            e, is_active=study_active and i == last_idx
        )
        for i, e in enumerate(study.experiments)
    }


def current_step(study: Study) -> tuple[str, str] | None:
    """Return ``(experiment_id, phase)`` for whatever is currently running."""
    if study.status not in {"RUNNING", "STARTED"} or not study.experiments:
        return None
    last = study.experiments[-1]
    pipeline = derive_experiment_pipeline(last, is_active=True)
    for phase, state in pipeline:
        if state == "running":
            return (last.id, phase)
    return None


__all__ = [
    "PHASES",
    "current_step",
    "derive_experiment_pipeline",
    "derive_study_pipelines",
]
