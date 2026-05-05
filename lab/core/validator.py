"""Static + smoke validator for LLM-generated training code (ADR-005).

Three static checks (syntax + forbidden imports + signature + ``nn.*`` reflection)
followed by a smoke forward pass on CPU. Trim of the prototype's 1024-LOC
behemoth: stays under ~300 LOC.
"""

from __future__ import annotations

import ast
import difflib
import traceback as _traceback
from typing import Iterable

from lab.config import Settings
from lab.core.models import ValidationResult

# Modules forbidden in every mode. Submission mode adds the network/install set.
_BASE_FORBIDDEN: frozenset[str] = frozenset(
    {"subprocess", "socket", "pip", "ctypes", "multiprocessing"}
)
_SUBMISSION_EXTRA_FORBIDDEN: frozenset[str] = frozenset(
    {"urllib", "requests", "http", "ftplib", "wget", "smtplib", "telnetlib"}
)


class Validator:
    """Validate generated build_model code against static + smoke checks."""

    def __init__(self, settings: Settings):
        self.settings = settings

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    def validate(
        self,
        code: str,
        *,
        signature: tuple[str, str],
        smoke_input_shape: tuple[int, ...],
        smoke_num_classes: int,
        submission_mode: bool = False,
    ) -> ValidationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return ValidationResult(
                ok=False,
                error_type="Syntax",
                message=f"{exc.msg} (line {exc.lineno})",
            )

        forbidden = self._forbidden(submission_mode)

        bad_import = _scan_imports(tree, forbidden)
        if bad_import:
            return ValidationResult(
                ok=False,
                error_type="ForbiddenImport",
                message=f"forbidden import: {bad_import}",
                findings=[bad_import],
            )

        if submission_mode and _has_shell_install(code):
            return ValidationResult(
                ok=False,
                error_type="ForbiddenImport",
                message="shell-install pattern (e.g. !pip install) is forbidden in submission mode",
            )

        fn_name, expected_arg = signature
        sig_check = _check_signature(tree, fn_name, expected_arg)
        if sig_check is not None:
            return sig_check

        nn_check = _check_torch_nn(tree)
        if nn_check is not None:
            return nn_check

        smoke = _smoke(code, fn_name, smoke_input_shape, smoke_num_classes)
        if smoke is not None:
            return smoke

        return ValidationResult(ok=True)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _forbidden(self, submission_mode: bool) -> frozenset[str]:
        if submission_mode:
            return _BASE_FORBIDDEN | _SUBMISSION_EXTRA_FORBIDDEN
        return _BASE_FORBIDDEN


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _scan_imports(tree: ast.AST, forbidden: frozenset[str]) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root(alias.name) in forbidden:
                    return alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _root(module) in forbidden:
                return module
    return None


def _root(module_path: str) -> str:
    return module_path.split(".")[0] if module_path else ""


def _check_signature(tree: ast.AST, fn_name: str, expected_arg: str) -> ValidationResult | None:
    found: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            found = node
            break
    if found is None:
        return ValidationResult(
            ok=False,
            error_type="BadSignature",
            message=f"function `{fn_name}` not defined",
            autofix_hint=f"add `def {fn_name}({expected_arg}: int)` returning a torch.nn.Module",
        )
    arg_names = [a.arg for a in found.args.args]
    if expected_arg not in arg_names:
        return ValidationResult(
            ok=False,
            error_type="BadSignature",
            message=f"`{fn_name}` is missing required argument `{expected_arg}` (got {arg_names})",
            autofix_hint=f"add `{expected_arg}` parameter to `{fn_name}`",
        )
    return None


def _check_torch_nn(tree: ast.AST) -> ValidationResult | None:
    """Reject hallucinated ``nn.Foo`` / ``torch.nn.Foo`` references."""
    try:
        import torch.nn as torch_nn
    except ImportError:  # pragma: no cover - torch is required for smoke too
        return None

    valid_attrs = {name for name in dir(torch_nn) if not name.startswith("_")}

    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        attr_chain = _attribute_chain(node)
        if attr_chain is None:
            continue
        # Match ``nn.<X>`` or ``torch.nn.<X>``
        if attr_chain[:1] == ["nn"] and len(attr_chain) >= 2:
            tail = attr_chain[1]
        elif attr_chain[:2] == ["torch", "nn"] and len(attr_chain) >= 3:
            tail = attr_chain[2]
        else:
            continue
        if tail not in valid_attrs:
            bad.append(tail)

    if bad:
        attr = bad[0]
        suggestion = _closest_attr(attr, valid_attrs)
        hint = (
            f"rename nn.{attr} -> nn.{suggestion}"
            if suggestion
            else f"`nn.{attr}` does not exist"
        )
        return ValidationResult(
            ok=False,
            error_type="UnknownTorchNN",
            message=f"unknown torch.nn attribute: nn.{attr}",
            findings=[f"nn.{a}" for a in bad],
            autofix_hint=hint,
        )
    return None


def _attribute_chain(node: ast.Attribute) -> list[str] | None:
    """Return the dotted name leading to and including this attribute."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return list(reversed(parts))
    return None


def _closest_attr(name: str, candidates: Iterable[str]) -> str | None:
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.7)
    return matches[0] if matches else None


def _has_shell_install(code: str) -> bool:
    lower = code.lower()
    return any(needle in lower for needle in ("!pip install", "!pip3 install", "%pip install"))


# ---------------------------------------------------------------------------
# Smoke forward pass
# ---------------------------------------------------------------------------


def _smoke(
    code: str,
    fn_name: str,
    smoke_input_shape: tuple[int, ...],
    smoke_num_classes: int,
) -> ValidationResult | None:
    try:
        import torch  # noqa: F401 — required for smoke
    except ImportError as exc:
        return ValidationResult(
            ok=False,
            error_type="SmokeFailed",
            message=f"torch import failed: {exc}",
        )

    namespace: dict[str, object] = {"__name__": "__sandbox__"}
    try:
        exec(compile(code, "<validator-sandbox>", "exec"), namespace)  # noqa: S102 — sandboxed; imports gated above
    except Exception as exc:
        return ValidationResult(
            ok=False,
            error_type="SmokeFailed",
            message=f"module-level execution failed: {exc!r}",
            findings=[_traceback.format_exc()],
        )

    fn = namespace.get(fn_name)
    if fn is None or not callable(fn):
        return ValidationResult(
            ok=False,
            error_type="SmokeFailed",
            message=f"`{fn_name}` not defined or not callable after exec",
        )

    try:
        model = fn(num_classes=smoke_num_classes)
    except Exception as exc:
        return ValidationResult(
            ok=False,
            error_type="SmokeFailed",
            message=f"{fn_name}({smoke_num_classes}) raised: {exc!r}",
            findings=[_traceback.format_exc()],
        )

    import torch  # local import (already guarded above)

    try:
        model.eval()
        x = torch.randn(2, *smoke_input_shape)
        with torch.no_grad():
            out = model(x)
    except Exception as exc:
        return ValidationResult(
            ok=False,
            error_type="SmokeFailed",
            message=f"forward pass raised: {exc!r}",
            findings=[_traceback.format_exc()],
        )

    expected = (2, smoke_num_classes)
    if tuple(out.shape) != expected:
        return ValidationResult(
            ok=False,
            error_type="SmokeFailed",
            message=f"output shape {tuple(out.shape)} != expected {expected}",
        )
    return None
