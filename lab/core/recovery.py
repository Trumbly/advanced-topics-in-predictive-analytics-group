"""Two-layer self-repair: cheap deterministic autofix + LLM re-prompt (ADR-005).

Called by ``run_experiment`` (I-06) after a validator failure or an executor
failure. Tries deterministic fixes first; only falls back to the LLM when no
autofix applies.
"""

from __future__ import annotations

import re
from typing import Mapping

from lab.config import Settings
from lab.core.executor import _HARD_FAILURES as _EXECUTOR_HARD_FAILURES
from lab.core.llm import LLMClient
from lab.core.models import TaskError, ValidationResult
from lab.prompts.engine import PromptEngine

# Re-export so consumers (StudyRunner) need only import from recovery.
_HARD_FAILURE_ERROR_TYPES: frozenset[str] = _EXECUTOR_HARD_FAILURES


# Detect a deterministic ``nn.<bad>`` -> ``nn.<good>`` rename when the
# validator's autofix_hint named both layers.
_NN_RENAME_RE = re.compile(
    r"rename\s+nn\.(?P<bad>\w+)\s*(?:->|→)\s*nn\.(?P<good>\w+)"
)


class Recovery:
    def __init__(
        self,
        client: LLMClient,
        engine: PromptEngine,
        settings: Settings,
    ):
        self.client = client
        self.engine = engine
        self.settings = settings

    # ------------------------------------------------------------------
    # autofix
    # ------------------------------------------------------------------

    def try_autofix(self, code: str, finding: ValidationResult) -> str | None:
        """Return repaired code on a deterministic match, else None."""
        if finding.ok:
            return None

        if finding.error_type == "UnknownTorchNN" and finding.autofix_hint:
            m = _NN_RENAME_RE.search(finding.autofix_hint)
            if m:
                bad = m.group("bad")
                good = m.group("good")
                pattern = re.compile(rf"\bnn\.{re.escape(bad)}\b")
                repaired, n = pattern.subn(f"nn.{good}", code)
                if n > 0:
                    return repaired

        if finding.error_type == "BadSignature":
            repaired = self._inject_num_classes_arg(code)
            if repaired is not None:
                return repaired

        return None

    @staticmethod
    def _inject_num_classes_arg(code: str) -> str | None:
        """Add a `num_classes` parameter to a `def build_model()` if missing."""
        pattern = re.compile(r"def\s+build_model\s*\(\s*\)\s*:")
        if not pattern.search(code):
            return None
        return pattern.sub("def build_model(num_classes: int):", code, count=1)

    # ------------------------------------------------------------------
    # LLM re-prompt
    # ------------------------------------------------------------------

    def ask_llm(
        self,
        code: str,
        error: TaskError | ValidationResult,
        *,
        slots: Mapping[str, str],
    ) -> str:
        all_slots = dict(slots)
        all_slots["broken_code"] = code
        all_slots["error_type"], all_slots["error_message"], all_slots[
            "error_traceback"
        ] = _flatten_error(error)
        sys_prompt, user_prompt = self.engine.render(
            "recover_from_error", all_slots
        )
        return self.client.chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _flatten_error(error: TaskError | ValidationResult) -> tuple[str, str, str]:
    if isinstance(error, ValidationResult):
        return (
            error.error_type or "Unknown",
            error.message or "",
            "\n".join(error.findings or []),
        )
    return (
        error.error_type,
        error.message or "",
        error.traceback or "",
    )
