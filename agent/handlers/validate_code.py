"""Predefined handler: validate LLM-generated Python code before execution.

This is our first line of defense against malformed or unsafe code. It:

1. Compiles the code (catches SyntaxError before we spin up a subprocess).
2. Scans the AST for forbidden imports (anything that touches the network,
   the OS shell, or the parent process).
3. Scans the source for explicit patterns from the pipeline config's
   `reject_patterns` list (e.g. `os.system`, `subprocess.Popen`).
4. Optionally checks that the code imports at least one expected module
   from the fixed pipeline / framework allowlist.

If any check fails, the task status is set to FAILED with a TaskError
describing the problem. The orchestrator can then feed the error into
the error-recovery prompt.

The code is taken from `task.code_used` (populated by the preceding
`generate_code` LLM task).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from agent.models import Task, TaskError, TaskStatus


# ---------------------------------------------------------------------------
# Hardcoded safety list (always enforced, even if the pipeline config forgets)
# ---------------------------------------------------------------------------


# Top-level module names that are always forbidden, no matter what the
# pipeline YAML says. These are the ones with real blast radius.
FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "subprocess",
        "socket",
        "urllib",
        "http",
        "requests",
        "httpx",
        "ftplib",
        "telnetlib",
        "smtplib",
        "shutil",
    }
)


# Substring patterns that indicate dangerous string-based calls. The
# pipeline YAML can extend this list; these are the always-on baseline.
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "os.system",
    "os.popen",
    "os.execv",
    "os.execvp",
    "os.remove",
    "os.rmdir",
    "shutil.rmtree",
    "__import__",
    "eval(",
    "exec(",
    "open('/etc/",
    'open("/etc/',
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    ok: bool
    error_type: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run(task: Task, *, config: dict[str, Any] | None = None) -> Task:
    """Validate `task.code_used` and update the task in place.

    Args:
        task: the Task whose `code_used` field holds the code to validate.
        config: optional dict from the pipeline YAML's `config:` block.
            Supported keys:
              - `check_imports`: list of module names that SHOULD be present
                (validation fails if none are imported)
              - `reject_patterns`: extra substring patterns to reject,
                merged with the hardcoded FORBIDDEN_PATTERNS

    Returns:
        The same Task object, with `status`, `output`, and `error` updated.
    """
    task.started_at = _now()
    task.status = TaskStatus.RUNNING

    code = task.code_used or ""
    if not code.strip():
        return _fail(task, "NoCode", "Task.code_used is empty")

    config = config or {}
    extra_rejects = tuple(config.get("reject_patterns") or ())
    check_imports = tuple(config.get("check_imports") or ())

    result = validate(
        code,
        check_imports=check_imports,
        extra_reject_patterns=extra_rejects,
    )

    if not result.ok:
        return _fail(task, result.error_type, result.message)

    task.output = {"validation": "passed", "code_bytes": len(code.encode("utf-8"))}
    task.status = TaskStatus.COMPLETED
    task.completed_at = _now()
    return task


# ---------------------------------------------------------------------------
# Pure validation logic (no Task mutation — easy to unit-test)
# ---------------------------------------------------------------------------


def validate(
    code: str,
    *,
    check_imports: tuple[str, ...] = (),
    extra_reject_patterns: tuple[str, ...] = (),
) -> ValidationResult:
    """Validate a code string without any side effects."""
    # 1. Syntax check — compile raises SyntaxError with line info
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ValidationResult(
            ok=False,
            error_type="SyntaxError",
            message=f"Line {e.lineno}: {e.msg}",
        )

    # 2. Forbidden imports (AST-level, more reliable than regex)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                imported_names.add(top)
                if top in FORBIDDEN_IMPORTS:
                    return ValidationResult(
                        ok=False,
                        error_type="ForbiddenImport",
                        message=f"Import of '{alias.name}' is not allowed",
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            imported_names.add(top)
            if top in FORBIDDEN_IMPORTS:
                return ValidationResult(
                    ok=False,
                    error_type="ForbiddenImport",
                    message=f"Import from '{module}' is not allowed",
                )

    # 3. Forbidden string patterns (substring scan — catches tricks AST misses)
    all_patterns = FORBIDDEN_PATTERNS + tuple(extra_reject_patterns)
    for pattern in all_patterns:
        if pattern in code:
            return ValidationResult(
                ok=False,
                error_type="ForbiddenPattern",
                message=f"Forbidden pattern found: {pattern!r}",
            )

    # 4. Expected imports (at least one must be present)
    if check_imports and not (set(check_imports) & imported_names):
        return ValidationResult(
            ok=False,
            error_type="MissingExpectedImport",
            message=(
                f"None of the expected imports {list(check_imports)} "
                f"found. Got: {sorted(imported_names)}"
            ),
        )

    return ValidationResult(ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fail(task: Task, error_type: str, message: str) -> Task:
    task.status = TaskStatus.FAILED
    task.error = TaskError(error_type=error_type, message=message)
    task.output = {"validation": "failed"}
    task.completed_at = _now()
    return task


def _now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "run",
    "validate",
    "ValidationResult",
    "FORBIDDEN_IMPORTS",
    "FORBIDDEN_PATTERNS",
]
