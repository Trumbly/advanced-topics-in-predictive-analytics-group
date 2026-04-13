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
#
# NOTE: `eval(` and `exec(` are NOT in this list. They would false-positive on
# perfectly legitimate method calls like `model.eval()` (PyTorch's way of
# switching a model to inference mode) or `executor.exec(...)` (ThreadPool
# methods). Instead we reject BARE `eval()` / `exec()` calls at the AST level
# via `_FORBIDDEN_BARE_CALLS` below, which only fires on the builtin names.
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "os.system",
    "os.popen",
    "os.execv",
    "os.execvp",
    "os.remove",
    "os.rmdir",
    "shutil.rmtree",
    "__import__",
    "open('/etc/",
    'open("/etc/',
)


# Bare builtin names that must never be called as `name(...)`.
# Detected via AST Call nodes whose func is an ast.Name with one of these ids,
# so attribute calls like `model.eval()` are correctly ignored.
_FORBIDDEN_BARE_CALLS: frozenset[str] = frozenset({"eval", "exec"})


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
              - `max_epochs`: integer cap on EPOCHS = N. Rejects any code
                where `EPOCHS = <int>` at module scope exceeds this value.

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
    max_epochs = config.get("max_epochs")

    result = validate(
        code,
        check_imports=check_imports,
        extra_reject_patterns=extra_rejects,
        max_epochs=max_epochs,
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
    max_epochs: int | None = None,
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

    # 3. Bare builtin calls to eval() / exec() (AST-level, so method calls
    #    like `model.eval()` and `executor.exec()` are correctly allowed)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_BARE_CALLS:
                return ValidationResult(
                    ok=False,
                    error_type="ForbiddenBareCall",
                    message=(
                        f"Bare call to builtin '{node.func.id}()' is not allowed "
                        f"(line {node.lineno}). Method calls like "
                        f"'model.eval()' are fine."
                    ),
                )

    # 4. Forbidden string patterns (substring scan — catches tricks AST misses)
    all_patterns = FORBIDDEN_PATTERNS + tuple(extra_reject_patterns)
    for pattern in all_patterns:
        if pattern in code:
            return ValidationResult(
                ok=False,
                error_type="ForbiddenPattern",
                message=f"Forbidden pattern found: {pattern!r}",
            )

    # 5. Expected imports (at least one must be present)
    if check_imports and not (set(check_imports) & imported_names):
        return ValidationResult(
            ok=False,
            error_type="MissingExpectedImport",
            message=(
                f"None of the expected imports {list(check_imports)} "
                f"found. Got: {sorted(imported_names)}"
            ),
        )

    # 6. `__main__` guard enforcement.
    #    PyTorch's DataLoader with num_workers > 0 uses multiprocessing
    #    with the `spawn` start method on macOS. Spawn workers re-import
    #    the script, so any call that materializes the DataLoader (or
    #    iterates it) MUST live inside `if __name__ == "__main__":`.
    #    If it runs at module scope, each worker recursively tries to
    #    spawn its own workers and crashes with:
    #      RuntimeError: An attempt has been made to start a new process
    #      before the current process has finished its bootstrapping phase.
    #    We detect this by checking whether `load_precomputed_dataset(...)`
    #    or `DataLoader(...)` calls appear at top-level (outside any
    #    function def / class def / if-main block).
    guard_violation = _check_main_guard(tree)
    if guard_violation is not None:
        return ValidationResult(
            ok=False,
            error_type="MissingMainGuard",
            message=guard_violation,
        )

    # 7. Hallucinated `torch.nn.*` attributes. Small LLMs sometimes invent
    #    layer names like `nn.Conv2x2d`, `nn.LinearReLU`, `nn.SoftMax2D`.
    #    We reflect against the real `torch.nn` module and reject any
    #    attribute that does not exist. This catches typos and fabrications
    #    BEFORE the subprocess spends time loading data.
    nn_violation = _check_torch_nn_attributes(tree)
    if nn_violation is not None:
        return ValidationResult(
            ok=False,
            error_type="UnknownTorchNnAttribute",
            message=nn_violation,
        )

    # 8. EPOCHS cap enforcement. If the pipeline config set `max_epochs`,
    #    any top-level constant assignment `EPOCHS = <int>` with a value
    #    larger than max_epochs gets rejected.
    #
    #    The preferred pattern is:
    #        EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "15"))
    #    which is ACCEPTED as long as the default is within the cap.
    #    We only reject literal integer assignments that exceed the cap.
    if max_epochs is not None:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not (
                        isinstance(target, ast.Name) and target.id == "EPOCHS"
                    ):
                        continue
                    # Accept: EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "<n>"))
                    # as long as <n> itself is within the cap.
                    env_default = _epochs_env_default(node.value)
                    if env_default is not None:
                        if env_default > max_epochs:
                            return ValidationResult(
                                ok=False,
                                error_type="EpochsCapExceeded",
                                message=(
                                    f"BIRDCLEF_EPOCHS default {env_default} exceeds "
                                    f"the hard cap of {max_epochs}. Use default '15'."
                                ),
                            )
                        continue  # env-var pattern is fine
                    # Reject: EPOCHS = <literal int>  when literal > cap
                    if (
                        isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, int)
                        and node.value.value > max_epochs
                    ):
                        return ValidationResult(
                            ok=False,
                            error_type="EpochsCapExceeded",
                            message=(
                                f"EPOCHS = {node.value.value} exceeds the hard "
                                f"cap of {max_epochs} (fast-iteration mode). "
                                f"Use `EPOCHS = int(os.environ.get(\"BIRDCLEF_EPOCHS\", \"15\"))`."
                            ),
                        )

    # 9. Mock / stub detection. The LLM sometimes invents its own fake
    #    data loaders (MockLoader, MockDataset, mock_load_...) instead of
    #    using the real `load_precomputed_dataset` from `pipelines.data_loader`.
    #    This always leads to cascading failures because the mock data has
    #    wrong shapes, wrong class counts, and missing methods. Reject it
    #    early so the codegen retry can produce code that uses the real loader.
    _MOCK_CLASS_PREFIXES = ("Mock", "Fake", "Stub", "Dummy")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(node.name.startswith(p) for p in _MOCK_CLASS_PREFIXES):
                return ValidationResult(
                    ok=False,
                    error_type="MockDetected",
                    message=(
                        f"Class '{node.name}' looks like a mock/stub (line {node.lineno}). "
                        f"Do NOT create fake data loaders — use the real "
                        f"`load_precomputed_dataset` from `pipelines.data_loader`."
                    ),
                )
        if isinstance(node, ast.FunctionDef) and "mock" in node.name.lower():
            return ValidationResult(
                ok=False,
                error_type="MockDetected",
                message=(
                    f"Function '{node.name}' looks like a mock/stub (line {node.lineno}). "
                    f"Do NOT create fake data loaders — use the real "
                    f"`load_precomputed_dataset` from `pipelines.data_loader`."
                ),
            )

    return ValidationResult(ok=True)


def _epochs_env_default(node: ast.AST) -> int | None:
    """If `node` is `int(os.environ.get("BIRDCLEF_EPOCHS", "<n>"))`,
    return the int value of `<n>`. Otherwise return None.

    Also accepts the bare `os.environ.get("BIRDCLEF_EPOCHS", "<n>")` form
    as long as the default value is a string that parses to int.
    """
    # Unwrap `int(...)` if present.
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and len(node.args) == 1
    ):
        node = node.args[0]

    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute):
        return None
    # Match .get on something that looks like os.environ.
    if node.func.attr != "get":
        return None
    base = node.func.value
    # os.environ OR a variable named "environ" (unlikely but possible)
    is_env = (
        (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id == "os"
            and base.attr == "environ"
        )
        or (isinstance(base, ast.Name) and base.id == "environ")
    )
    if not is_env:
        return None
    if len(node.args) < 2:
        return None
    key_node, default_node = node.args[0], node.args[1]
    if not (
        isinstance(key_node, ast.Constant) and key_node.value == "BIRDCLEF_EPOCHS"
    ):
        return None
    if not isinstance(default_node, ast.Constant):
        return None
    try:
        return int(default_node.value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# __main__ guard detection
# ---------------------------------------------------------------------------

# Call expressions that trigger multiprocessing worker spawn and therefore
# MUST be inside `if __name__ == "__main__":`. We match on function-name
# or attribute-name suffix so both `load_precomputed_dataset(...)` and
# `pipelines.data_loader.load_precomputed_dataset(...)` are detected.
_SPAWN_TRIGGERING_CALLS: frozenset[str] = frozenset(
    {
        "load_precomputed_dataset",
        "DataLoader",
    }
)


def _is_if_name_main(node: ast.If) -> bool:
    """Return True iff `node` is the classic `if __name__ == "__main__":` guard."""
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left, right = test.left, test.comparators[0]

    def _is_name_main(n: ast.AST) -> bool:
        return isinstance(n, ast.Name) and n.id == "__name__"

    def _is_main_const(n: ast.AST) -> bool:
        return isinstance(n, ast.Constant) and n.value == "__main__"

    return (_is_name_main(left) and _is_main_const(right)) or (
        _is_name_main(right) and _is_main_const(left)
    )


def _call_trigger_name(call: ast.Call) -> str | None:
    """Return the triggering function name of a Call node, or None."""
    func = call.func
    if isinstance(func, ast.Name) and func.id in _SPAWN_TRIGGERING_CALLS:
        return func.id
    if isinstance(func, ast.Attribute) and func.attr in _SPAWN_TRIGGERING_CALLS:
        return func.attr
    return None


def _find_unguarded_spawn_calls(
    nodes: list[ast.stmt],
) -> list[tuple[str, int]]:
    """Walk top-level statements and collect (trigger_name, lineno) pairs
    for spawn-triggering calls that are NOT inside a function, class, or
    `if __name__ == "__main__":` block."""
    hits: list[tuple[str, int]] = []
    for stmt in nodes:
        # Skip bodies that Python would not re-execute recursively during
        # a spawn worker's module import: function/class defs.
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Skip the __main__ guard — its body is exactly where these calls
        # SHOULD live.
        if isinstance(stmt, ast.If) and _is_if_name_main(stmt):
            continue
        # Everything else is genuinely at top-level module scope.
        # Walk the statement and report any triggering call we find.
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                name = _call_trigger_name(node)
                if name is not None:
                    hits.append((name, node.lineno))
    return hits


def _check_main_guard(tree: ast.Module) -> str | None:
    """Return a violation message if any spawn-triggering call is at
    module scope, otherwise None."""
    unguarded = _find_unguarded_spawn_calls(list(tree.body))
    if not unguarded:
        return None
    first_name, first_line = unguarded[0]
    return (
        f"`{first_name}(...)` called at module scope (line {first_line}) "
        f"without an `if __name__ == \"__main__\":` guard. PyTorch DataLoader "
        f"with num_workers > 0 uses spawn workers that re-import the script; "
        f"without the guard each worker recursively spawns more workers and "
        f"Python raises a bootstrapping RuntimeError. Move the training block "
        f"(everything that actually RUNS, not class definitions) inside "
        f"`if __name__ == \"__main__\":`."
    )


# ---------------------------------------------------------------------------
# torch.nn attribute hallucination detection
# ---------------------------------------------------------------------------
#
# Small LLMs (nemotron-3-nano:4b in our runs) occasionally invent layer
# names that look plausible but don't actually exist in PyTorch. Real
# examples from our failed experiments:
#   - nn.Conv2x2d        (meant Conv2d, added an "x2")
#   - nn.LinearReLU      (no such thing)
#   - nn.BatchNormal2d   (meant BatchNorm2d)
#
# We detect these by reflecting against the actual `torch.nn` module at
# validation time. If a node of the form `nn.<Attr>` appears in an AST
# Call or Attribute expression and `<Attr>` is not in `dir(torch.nn)`,
# we reject. This is a pure static analysis check — no code gets run.
#
# `nn` is recognized as any alias that maps to `torch.nn`:
#   import torch.nn as nn         → alias = "nn"
#   from torch import nn          → alias = "nn"
#   import torch.nn as neural_net → alias = "neural_net"


def _find_torch_nn_aliases(tree: ast.Module) -> set[str]:
    """Return the set of local names that refer to the `torch.nn` module."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch.nn":
                    aliases.add(alias.asname or "torch.nn")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torch":
                for alias in node.names:
                    if alias.name == "nn":
                        aliases.add(alias.asname or "nn")
            elif node.module == "torch.nn":
                # `from torch.nn import Conv2d as C` — C is a layer, not nn.
                # We do NOT treat these as aliases of the module itself.
                pass
    return aliases


def _load_torch_nn_attribute_set() -> frozenset[str] | None:
    """Return the set of valid names in `torch.nn`, or None if torch is
    not importable in the validator process (e.g. during unit tests that
    have mocked torch out). When None, the check is skipped."""
    try:
        import torch.nn as _nn  # noqa: PLC0415

        return frozenset(
            name for name in dir(_nn) if not name.startswith("_")
        )
    except Exception:  # noqa: BLE001
        return None


def _check_torch_nn_attributes(tree: ast.Module) -> str | None:
    """Walk the AST and reject any `nn.<Attr>` reference that is not a
    real attribute of `torch.nn`. Returns an error message on violation,
    or None when the code is clean."""
    aliases = _find_torch_nn_aliases(tree)
    if not aliases:
        return None  # nothing imported nn — nothing to check

    valid = _load_torch_nn_attribute_set()
    if valid is None:
        return None  # torch unavailable in the validator env — skip

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if not isinstance(base, ast.Name):
            continue
        if base.id not in aliases:
            continue
        attr = node.attr
        # Dunder / private attributes we ignore (they are not real layer calls).
        if attr.startswith("_"):
            continue
        if attr in valid:
            continue
        # Close-match hint: what is the LLM probably trying to say?
        hint = _closest_nn_attribute(attr, valid)
        suffix = f" Did you mean `nn.{hint}`?" if hint else ""
        return (
            f"`{base.id}.{attr}` does not exist in torch.nn (line {node.lineno})."
            f"{suffix} Fix by using a real torch.nn layer name."
        )

    return None


def _closest_nn_attribute(
    name: str, valid: frozenset[str], *, max_distance: int = 3
) -> str | None:
    """Return the closest valid torch.nn name by Levenshtein distance,
    or None if nothing is within `max_distance`."""
    import difflib  # noqa: PLC0415

    matches = difflib.get_close_matches(name, list(valid), n=1, cutoff=0.6)
    if matches:
        return matches[0]
    # Fallback: scan for close prefixes.
    lname = name.lower()
    for v in valid:
        if lname.startswith(v.lower()[: max(4, len(v) - 1)]):
            return v
    return None


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
