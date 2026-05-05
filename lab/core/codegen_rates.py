"""Per-experiment + per-model codegen success rates.

Measures how often the LLM produces code that survives the static + smoke
validator on the first try and how often it survives execution. Aggregated
per ``llm_model`` so we can compare which model writes the cleanest code.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from lab.core.loaders import list_studies, load_many
from lab.core.models import Experiment, Study


class ExperimentRates(BaseModel):
    """Per-experiment validate/execute pass rates."""

    experiment_id: str
    validate_attempts: int
    validate_succeeded: int
    validation_pass_rate: float           # succeeded / attempts (0.0 if no attempts)
    validate_first_try_ok: bool | None    # None when no validate Task exists
    execute_attempts: int
    execute_succeeded: int
    execution_pass_rate: float
    execute_first_try_ok: bool | None


class ModelCodegenStats(BaseModel):
    """Per-LLM-model aggregated codegen quality."""

    llm_model: str
    studies: int
    experiments: int
    validate_attempts: int
    validate_failed: int
    bad_code_rate: float                  # validate_failed / validate_attempts
    first_try_validate_pass_rate: float   # exps where first validate passed / total exps with any validate
    execute_attempts: int
    execute_failed: int
    execution_failure_rate: float


def experiment_rates(exp: Experiment) -> ExperimentRates:
    val = [t for t in exp.tasks if t.name == "validate"]
    ex = [t for t in exp.tasks if t.name == "execute"]

    val_ok = sum(1 for t in val if t.status == "SUCCEEDED")
    ex_ok = sum(1 for t in ex if t.status == "SUCCEEDED")

    return ExperimentRates(
        experiment_id=exp.id,
        validate_attempts=len(val),
        validate_succeeded=val_ok,
        validation_pass_rate=(val_ok / len(val)) if val else 0.0,
        validate_first_try_ok=(val[0].status == "SUCCEEDED") if val else None,
        execute_attempts=len(ex),
        execute_succeeded=ex_ok,
        execution_pass_rate=(ex_ok / len(ex)) if ex else 0.0,
        execute_first_try_ok=(ex[0].status == "SUCCEEDED") if ex else None,
    )


def study_rates(study: Study) -> list[ExperimentRates]:
    return [experiment_rates(e) for e in study.experiments]


def aggregate_model_codegen(
    experiments_dir: Path,
) -> dict[str, ModelCodegenStats]:
    """Group every experiment's validate/execute attempts by ``study.llm_model``."""
    bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "studies": set(),
            "experiments": 0,
            "v_attempts": 0,
            "v_failed": 0,
            "v_with_first_try": 0,
            "v_first_try_ok": 0,
            "e_attempts": 0,
            "e_failed": 0,
        }
    )

    studies = load_many(Path(experiments_dir), list_studies(experiments_dir))
    for s in studies:
        model = s.llm_model or "unknown"
        slot = bucket[model]
        slot["studies"].add(s.id)
        for exp in s.experiments:
            slot["experiments"] += 1
            val = [t for t in exp.tasks if t.name == "validate"]
            ex = [t for t in exp.tasks if t.name == "execute"]
            slot["v_attempts"] += len(val)
            slot["v_failed"] += sum(1 for t in val if t.status != "SUCCEEDED")
            if val:
                slot["v_with_first_try"] += 1
                if val[0].status == "SUCCEEDED":
                    slot["v_first_try_ok"] += 1
            slot["e_attempts"] += len(ex)
            slot["e_failed"] += sum(1 for t in ex if t.status != "SUCCEEDED")

    out: dict[str, ModelCodegenStats] = {}
    for model, slot in bucket.items():
        v_attempts = slot["v_attempts"]
        e_attempts = slot["e_attempts"]
        out[model] = ModelCodegenStats(
            llm_model=model,
            studies=len(slot["studies"]),
            experiments=slot["experiments"],
            validate_attempts=v_attempts,
            validate_failed=slot["v_failed"],
            bad_code_rate=(slot["v_failed"] / v_attempts) if v_attempts else 0.0,
            first_try_validate_pass_rate=(
                slot["v_first_try_ok"] / slot["v_with_first_try"]
                if slot["v_with_first_try"]
                else 0.0
            ),
            execute_attempts=e_attempts,
            execute_failed=slot["e_failed"],
            execution_failure_rate=(slot["e_failed"] / e_attempts) if e_attempts else 0.0,
        )
    return out
