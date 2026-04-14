"""Static validation of LLM-generated training code.

Preserves the two high-value AST checks from max_development:

  * ``_check_main_guard``       — the ``__main__``-guard enforcement that
    prevents PyTorch DataLoader spawn-storms on macOS.
  * ``_check_torch_nn_attributes`` — reflects against ``dir(torch.nn)`` to
    reject hallucinated layer names (``nn.Conv2x2d`` and friends).

Forbidden imports / bare ``eval``+``exec`` are always-on safety baseline.

This module is fully task-agnostic. The list of spawn-triggering call names
is configurable by the caller so each task adapter can declare its own
data-loader symbol.
"""
from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# Always-on safety baseline
# ---------------------------------------------------------------------------

FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "subprocess", "socket", "urllib", "http", "requests", "httpx",
        "ftplib", "telnetlib", "smtplib", "shutil",
    }
)

FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "os.system", "os.popen", "os.execv", "os.execvp",
    "os.remove", "os.rmdir", "shutil.rmtree",
    "__import__", "open('/etc/", 'open("/etc/',
)

# Bare builtins that must never be called as `name(...)`. Attribute forms
# like `model.eval()` are correctly ignored by the AST check.
_FORBIDDEN_BARE_CALLS: frozenset[str] = frozenset({"eval", "exec"})

# Default data-loader symbol names that, when called at module scope,
# must be inside an `if __name__ == "__main__":` guard. Task adapters can
# extend this via `validate(extra_spawn_triggers=...)`.
DEFAULT_SPAWN_TRIGGERING_CALLS: frozenset[str] = frozenset(
    {"DataLoader", "load_audio_dataset", "load_text_dataset"}
)


@dataclass
class ValidationResult:
    ok: bool
    error_type: str = ""
    message: str = ""
    findings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def validate(
    code: str,
    *,
    extra_spawn_triggers: Iterable[str] = (),
    extra_forbidden_patterns: Iterable[str] = (),
) -> ValidationResult:
    """Validate a Python source string.

    Returns a ``ValidationResult``. ``ok`` is ``True`` only if every check
    passed; otherwise ``error_type`` and ``message`` describe the first
    failure the caller should surface to the LLM.
    """
    # 1) Syntax
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationResult(
            ok=False,
            error_type="SyntaxError",
            message=f"SyntaxError: {exc.msg} at line {exc.lineno}",
        )

    # 2) Forbidden imports
    bad_import = _check_forbidden_imports(tree)
    if bad_import:
        return ValidationResult(ok=False, error_type="ForbiddenImport", message=bad_import)

    # 3) Forbidden substrings in source
    for pat in tuple(FORBIDDEN_PATTERNS) + tuple(extra_forbidden_patterns):
        if pat in code:
            return ValidationResult(
                ok=False,
                error_type="ForbiddenPattern",
                message=f"Forbidden pattern in source: {pat!r}",
            )

    # 4) Forbidden bare calls
    bare = _check_forbidden_bare_calls(tree)
    if bare:
        return ValidationResult(ok=False, error_type="ForbiddenCall", message=bare)

    # 5) __main__ guard (preserved from max_development)
    spawn_calls = frozenset(DEFAULT_SPAWN_TRIGGERING_CALLS) | frozenset(extra_spawn_triggers)
    guard = _check_main_guard(tree, spawn_calls)
    if guard:
        return ValidationResult(ok=False, error_type="MissingMainGuard", message=guard)

    # 6) torch.nn reflection (preserved from max_development)
    nn = _check_torch_nn_attributes(tree)
    if nn:
        return ValidationResult(ok=False, error_type="HallucinatedLayer", message=nn)

    return ValidationResult(ok=True)


# ---------------------------------------------------------------------------
# Forbidden imports
# ---------------------------------------------------------------------------


def _check_forbidden_imports(tree: ast.Module) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in FORBIDDEN_IMPORTS:
                    return f"Forbidden import: {a.name} (line {node.lineno})"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in FORBIDDEN_IMPORTS:
                return f"Forbidden import from: {node.module} (line {node.lineno})"
    return None


def _check_forbidden_bare_calls(tree: ast.Module) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FORBIDDEN_BARE_CALLS:
            return (
                f"`{func.id}(...)` is forbidden (line {node.lineno}). "
                f"Only attribute calls like `model.eval()` are allowed."
            )
    return None


# ---------------------------------------------------------------------------
# __main__ guard (preserved verbatim)
# ---------------------------------------------------------------------------


def _is_if_name_main(node: ast.If) -> bool:
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


def _call_trigger_name(call: ast.Call, spawn_calls: frozenset[str]) -> str | None:
    func = call.func
    if isinstance(func, ast.Name) and func.id in spawn_calls:
        return func.id
    if isinstance(func, ast.Attribute) and func.attr in spawn_calls:
        return func.attr
    return None


def _find_unguarded_spawn_calls(
    nodes: list[ast.stmt], spawn_calls: frozenset[str]
) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for stmt in nodes:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, ast.If) and _is_if_name_main(stmt):
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                name = _call_trigger_name(node, spawn_calls)
                if name is not None:
                    hits.append((name, node.lineno))
    return hits


def _check_main_guard(tree: ast.Module, spawn_calls: frozenset[str]) -> str | None:
    """Return a violation message if any spawn-triggering call is at module
    scope, otherwise None."""
    unguarded = _find_unguarded_spawn_calls(list(tree.body), spawn_calls)
    if not unguarded:
        return None
    first_name, first_line = unguarded[0]
    return (
        f"`{first_name}(...)` called at module scope (line {first_line}) "
        f'without an `if __name__ == "__main__":` guard. PyTorch DataLoader '
        f"with num_workers > 0 uses spawn workers that re-import the script; "
        f"without the guard each worker recursively spawns more workers and "
        f"Python raises a bootstrapping RuntimeError. Move the training block "
        f"(everything that actually RUNS, not class definitions) inside "
        f'`if __name__ == "__main__":`.'
    )


# ---------------------------------------------------------------------------
# torch.nn reflection (preserved verbatim)
# ---------------------------------------------------------------------------


def _find_torch_nn_aliases(tree: ast.Module) -> set[str]:
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
    return aliases


def _load_torch_nn_attribute_set() -> frozenset[str] | None:
    try:
        import torch.nn as _nn  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return frozenset(name for name in dir(_nn) if not name.startswith("_"))


def _closest_nn_attribute(name: str, valid: frozenset[str]) -> str | None:
    matches = difflib.get_close_matches(name, list(valid), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _check_torch_nn_attributes(tree: ast.Module) -> str | None:
    aliases = _find_torch_nn_aliases(tree)
    if not aliases:
        return None
    valid = _load_torch_nn_attribute_set()
    if valid is None:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if not isinstance(base, ast.Name):
            continue
        if base.id not in aliases:
            continue
        attr = node.attr
        if attr.startswith("_"):
            continue
        if attr in valid:
            continue
        hint = _closest_nn_attribute(attr, valid)
        suffix = f" Did you mean `nn.{hint}`?" if hint else ""
        return (
            f"`{base.id}.{attr}` does not exist in torch.nn "
            f"(line {node.lineno}).{suffix} Fix by using a real torch.nn layer name."
        )
    return None


__all__ = [
    "ValidationResult",
    "validate",
    "FORBIDDEN_IMPORTS",
    "FORBIDDEN_PATTERNS",
    "DEFAULT_SPAWN_TRIGGERING_CALLS",
]
