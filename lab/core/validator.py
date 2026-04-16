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
import copy
import difflib
import sys
import types
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Always-on safety baseline
# ---------------------------------------------------------------------------

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

# Bare builtins that must never be called as `name(...)`. Attribute forms
# like `model.eval()` are correctly ignored by the AST check.
_FORBIDDEN_BARE_CALLS: frozenset[str] = frozenset({"eval", "exec"})

# Default data-loader symbol names that, when called at module scope,
# must be inside an `if __name__ == "__main__":` guard. Task adapters can
# extend this via `validate(extra_spawn_triggers=...)`.
DEFAULT_SPAWN_TRIGGERING_CALLS: frozenset[str] = frozenset(
    {"DataLoader", "load_audio_dataset", "load_text_dataset"}
)
_OPTIONAL_MODEL_LIBS: frozenset[str] = frozenset({"torchvision", "timm"})


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
    require_build_model: bool = False,
    require_training_contract: bool = False,
    smoke_input_shape: tuple[int, ...] | None = None,
    smoke_num_classes: int | None = None,
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
        return ValidationResult(
            ok=False, error_type="ForbiddenImport", message=bad_import
        )

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

    # 4b) Disallow interactive/remote model loading patterns that are brittle
    # in non-interactive sandboxes (e.g. torch.hub trust prompts).
    remote = _check_disallowed_remote_model_loads(tree)
    if remote:
        return ValidationResult(
            ok=False,
            error_type="ForbiddenModelSource",
            message=remote,
        )

    # 5) Missing `nn` alias import (common runtime NameError in generated code)
    nn_alias = _check_missing_nn_alias(tree)
    if nn_alias:
        return ValidationResult(
            ok=False, error_type="MissingNNImport", message=nn_alias
        )

    # 5b) `.numpy()` on tensors that are likely still attached to autograd graph.
    # Common runtime crash: "Can't call numpy() on Tensor that requires grad."
    numpy_conv = _check_unsafe_numpy_calls(tree)
    if numpy_conv:
        return ValidationResult(
            ok=False,
            error_type="UnsafeNumpyConversion",
            message=numpy_conv,
        )

    # 6) __main__ guard (preserved from max_development)
    spawn_calls = frozenset(DEFAULT_SPAWN_TRIGGERING_CALLS) | frozenset(
        extra_spawn_triggers
    )
    guard = _check_main_guard(tree, spawn_calls)
    if guard:
        return ValidationResult(ok=False, error_type="MissingMainGuard", message=guard)

    # 7) torch.nn reflection (preserved from max_development)
    nn = _check_torch_nn_attributes(tree)
    if nn:
        return ValidationResult(ok=False, error_type="HallucinatedLayer", message=nn)

    # 8) Lightweight dry-run of model contract (`build_model`) without
    # executing training. Catches name/symbol issues that pass pure AST checks.
    smoke = _check_build_model_smoke(
        tree,
        require_build_model=require_build_model,
        smoke_input_shape=smoke_input_shape,
        smoke_num_classes=smoke_num_classes,
    )
    if smoke:
        return ValidationResult(ok=False, error_type="DryRunFailed", message=smoke)

    # 9) Optional training-script contract:
    # enforce that generated code still looks like a trainable end-to-end run.
    if require_training_contract:
        contract = _check_training_contract(tree, code, spawn_calls=spawn_calls)
        if contract:
            return ValidationResult(
                ok=False,
                error_type="MissingTrainingContract",
                message=contract,
            )

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


def _find_torch_hub_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return (`hub`-module aliases, `load`-function aliases)."""
    hub_aliases: set[str] = set()
    load_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch.hub":
                    hub_aliases.add(alias.asname or "torch.hub")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torch":
                for alias in node.names:
                    if alias.name == "hub":
                        hub_aliases.add(alias.asname or "hub")
            elif node.module == "torch.hub":
                for alias in node.names:
                    if alias.name == "load":
                        load_aliases.add(alias.asname or "load")
    return hub_aliases, load_aliases


def _is_torch_hub_load_call(
    call: ast.Call,
    *,
    hub_aliases: set[str],
    load_aliases: set[str],
) -> bool:
    fn = call.func
    # torch.hub.load(...)
    if isinstance(fn, ast.Attribute) and fn.attr == "load":
        val = fn.value
        if isinstance(val, ast.Attribute):
            if isinstance(val.value, ast.Name) and val.value.id == "torch" and val.attr == "hub":
                return True
        if isinstance(val, ast.Name) and val.id in hub_aliases:
            return True
    # from torch.hub import load; load(...)
    if isinstance(fn, ast.Name) and fn.id in load_aliases:
        return True
    return False


def _check_disallowed_remote_model_loads(tree: ast.Module) -> str | None:
    """Block model-loading APIs that trigger interactive trust/network prompts."""
    hub_aliases, load_aliases = _find_torch_hub_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_torch_hub_load_call(
            node, hub_aliases=hub_aliases, load_aliases=load_aliases,
        ):
            return (
                f"`torch.hub.load(...)` is disallowed (line {node.lineno}). "
                "It may trigger interactive trust prompts or remote downloads. "
                "Use local torchvision/torch model constructors instead."
            )
    return None


def _build_model_uses_optional_model_libs(build_model_fn: ast.FunctionDef) -> bool:
    """Return True when build_model imports optional backbone libraries."""
    for node in ast.walk(build_model_fn):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in _OPTIONAL_MODEL_LIBS:
                    return True
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".", 1)[0]
            if top in _OPTIONAL_MODEL_LIBS:
                return True
    return False


def _make_optional_model_stubs(real_torch) -> dict[str, object]:
    """Return lightweight module stubs for optional model libraries.

    The validator runtime is intentionally minimal and may not have torchvision
    or timm installed. We inject deterministic stubs so build_model can still
    be imported/constructed without network/download side effects.
    """
    if real_torch is None:
        class _LinearLike:  # noqa: D106
            def __init__(self, in_features: int = 16):
                self.in_features = in_features

            def __call__(self, x):
                return x

        class _Indexable:  # noqa: D106
            def __init__(self, items: list[object] | None = None, *, in_features: int = 16):
                self.in_features = in_features
                self._items = list(items or [_LinearLike(in_features) for _ in range(4)])

            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    return self._items[idx]
                n = int(idx)
                if n < 0:
                    n += len(self._items)
                if n < 0:
                    n = 0
                while n >= len(self._items):
                    self._items.append(_LinearLike(self.in_features))
                return self._items[n]

            def __setitem__(self, idx, value):
                n = int(idx)
                if n < 0:
                    n += len(self._items)
                if n < 0:
                    n = 0
                while n >= len(self._items):
                    self._items.append(_LinearLike(self.in_features))
                self._items[n] = value

            def __call__(self, x):
                return x

        class _BackboneStub:  # noqa: D106
            def __init__(self):
                hidden = 16
                self.last_channel = hidden
                self.features = _Indexable([_Indexable(in_features=hidden), _Indexable(in_features=hidden)], in_features=hidden)
                self.avgpool = _Indexable(in_features=hidden)
                self.classifier = _Indexable(in_features=hidden)
                self.fc = _LinearLike(hidden)

            def children(self):
                return [self.features, self.avgpool, self.classifier, self.fc]

            def __call__(self, x):
                return x
    else:
        class _LinearLike(real_torch.nn.Module):  # noqa: D106
            def __init__(self, in_features: int = 16):
                super().__init__()
                self.in_features = in_features

            def forward(self, x):
                return x

        class _Indexable(real_torch.nn.Module):  # noqa: D106
            def __init__(self, items: list[real_torch.nn.Module] | None = None, *, in_features: int = 16):
                super().__init__()
                self.in_features = in_features
                base = items or [_LinearLike(in_features) for _ in range(4)]
                self._items = real_torch.nn.ModuleList(base)

            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    return list(self._items[idx])
                n = int(idx)
                if n < 0:
                    n += len(self._items)
                if n < 0:
                    n = 0
                while n >= len(self._items):
                    self._items.append(_LinearLike(self.in_features))
                return self._items[n]

            def __setitem__(self, idx, value):
                n = int(idx)
                if n < 0:
                    n += len(self._items)
                if n < 0:
                    n = 0
                while n >= len(self._items):
                    self._items.append(_LinearLike(self.in_features))
                self._items[n] = value

            def forward(self, x):
                return x

        class _BackboneStub(real_torch.nn.Module):  # noqa: D106
            def __init__(self):
                super().__init__()
                hidden = 16
                self.last_channel = hidden
                self.features = _Indexable(
                    [_Indexable(in_features=hidden), _Indexable(in_features=hidden)],
                    in_features=hidden,
                )
                self.avgpool = real_torch.nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = _Indexable(in_features=hidden)
                self.fc = _LinearLike(hidden)

            def children(self):
                return [self.features, self.avgpool, self.classifier, self.fc]

            def forward(self, x):
                if isinstance(x, real_torch.Tensor) and x.ndim >= 4:
                    x = self.avgpool(x)
                    x = real_torch.flatten(x, 1)
                return self.fc(x)

    class _WeightsStub:  # noqa: D106
        DEFAULT = None
        IMAGENET1K_V1 = None
        IMAGENET1K_V2 = None

    def _model_builder(*_args, **_kwargs):
        return _BackboneStub()

    tv_models = types.ModuleType("torchvision.models")
    tv_models.resnet18 = _model_builder
    tv_models.resnet34 = _model_builder
    tv_models.resnet50 = _model_builder
    tv_models.efficientnet_b0 = _model_builder
    tv_models.efficientnet_b1 = _model_builder
    tv_models.mobilenet_v3_small = _model_builder
    tv_models.mobilenet_v3_large = _model_builder

    def _models_getattr(name: str):
        if name.endswith("_Weights"):
            return _WeightsStub
        return _model_builder

    tv_models.__getattr__ = _models_getattr  # type: ignore[attr-defined]

    tv = types.ModuleType("torchvision")
    tv.models = tv_models

    timm = types.ModuleType("timm")
    timm.create_model = _model_builder

    return {
        "torchvision": tv,
        "torchvision.models": tv_models,
        "timm": timm,
    }


def _install_module_stubs(stubs: dict[str, object]) -> dict[str, object | None]:
    previous: dict[str, object | None] = {}
    for name, stub in stubs.items():
        previous[name] = sys.modules.get(name)
        sys.modules[name] = stub  # type: ignore[assignment]
    return previous


def _restore_module_stubs(previous: dict[str, object | None]) -> None:
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old  # type: ignore[assignment]


def _check_missing_nn_alias(tree: ast.Module) -> str | None:
    """Catch `nn.*` usage when no `nn` alias is defined/imported."""
    first_nn_attr_line: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "nn":
                first_nn_attr_line = node.lineno
                break
    if first_nn_attr_line is None:
        return None

    # Accept proper imports (`from torch import nn` or `import torch.nn as nn`).
    imported_aliases = _find_torch_nn_aliases(tree)
    if "nn" in imported_aliases:
        return None

    # Accept explicit manual aliasing (`nn = torch.nn`) if present.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "nn":
                    return None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "nn":
                return None

    return (
        f"`nn.*` used at line {first_nn_attr_line}, but `nn` is not defined. "
        "Add `import torch.nn as nn` or `from torch import nn`."
    )


def _expr_contains_detach(expr: ast.AST) -> bool:
    """Return True if expression chain includes a `.detach()` call."""
    if isinstance(expr, ast.Call):
        f = expr.func
        if isinstance(f, ast.Attribute) and f.attr == "detach":
            return True
        # Recurse into method-call receiver chain, e.g. x.detach().cpu()
        if isinstance(f, ast.Attribute):
            return _expr_contains_detach(f.value)
        if isinstance(f, ast.Name):
            return False
    if isinstance(expr, ast.Attribute):
        return _expr_contains_detach(expr.value)
    return False


def _is_no_grad_expr(expr: ast.AST) -> bool:
    """Return True for torch.no_grad()/no_grad()/torch.inference_mode()."""
    # with torch.no_grad():
    if isinstance(expr, ast.Call):
        fn = expr.func
    else:
        fn = expr
    if isinstance(fn, ast.Name):
        return fn.id in {"no_grad", "inference_mode"}
    if isinstance(fn, ast.Attribute):
        return fn.attr in {"no_grad", "inference_mode"}
    return False


def _is_nn_module_ctor_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Attribute):
        # nn.Module(...)
        if isinstance(fn.value, ast.Name) and fn.value.id == "nn" and fn.attr == "Module":
            return True
        # torch.nn.Module(...)
        if (
            isinstance(fn.value, ast.Attribute)
            and isinstance(fn.value.value, ast.Name)
            and fn.value.value.id == "torch"
            and fn.value.attr == "nn"
            and fn.attr == "Module"
        ):
            return True
    return False


def _find_returned_bare_module_line(build_model_fn: ast.FunctionDef) -> int | None:
    for node in ast.walk(build_model_fn):
        if isinstance(node, ast.Return) and node.value is not None:
            if _is_nn_module_ctor_call(node.value):
                return node.lineno
    return None


def _find_returned_conv2d_in_channels_line(
    build_model_fn: ast.FunctionDef,
    expected_in_channels: int,
) -> tuple[int, int] | None:
    """Catch direct `return nn.Conv2d(<bad in_channels>, ...)` patterns."""
    for node in ast.walk(build_model_fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        is_conv = (
            isinstance(fn, ast.Attribute)
            and fn.attr == "Conv2d"
            and (
                (isinstance(fn.value, ast.Name) and fn.value.id == "nn")
                or (
                    isinstance(fn.value, ast.Attribute)
                    and isinstance(fn.value.value, ast.Name)
                    and fn.value.value.id == "torch"
                    and fn.value.attr == "nn"
                )
            )
        )
        if not is_conv:
            continue
        in_channels: int | None = None
        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, int):
            in_channels = int(call.args[0].value)
        if in_channels is None:
            for kw in call.keywords:
                if kw.arg == "in_channels" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                    in_channels = int(kw.value.value)
                    break
        if in_channels is None:
            continue
        if in_channels != expected_in_channels:
            return in_channels, node.lineno
    return None


def _check_unsafe_numpy_calls(tree: ast.Module) -> str | None:
    """Reject tensor `.numpy()` conversions that don't detach first."""
    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.no_grad_depth = 0
            self.error: str | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: D401
            enters = any(_is_no_grad_expr(d) for d in node.decorator_list)
            if enters:
                self.no_grad_depth += 1
            self.generic_visit(node)
            if enters:
                self.no_grad_depth -= 1

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: D401
            enters = any(_is_no_grad_expr(d) for d in node.decorator_list)
            if enters:
                self.no_grad_depth += 1
            self.generic_visit(node)
            if enters:
                self.no_grad_depth -= 1

        def visit_With(self, node: ast.With) -> None:  # noqa: D401
            enters = any(_is_no_grad_expr(i.context_expr) for i in node.items)
            if enters:
                self.no_grad_depth += 1
            self.generic_visit(node)
            if enters:
                self.no_grad_depth -= 1

        def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
            if self.error is not None:
                return
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "numpy":
                receiver = func.value
                if self.no_grad_depth <= 0 and not _expr_contains_detach(receiver):
                    self.error = (
                        f"Potential unsafe `.numpy()` conversion at line {node.lineno}. "
                        "Use `.detach().cpu().numpy()` (or `.detach().numpy()` on CPU tensors)."
                    )
                    return
            self.generic_visit(node)

    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.error


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


def _check_build_model_smoke(
    tree: ast.Module,
    *,
    require_build_model: bool,
    smoke_input_shape: tuple[int, ...] | None = None,
    smoke_num_classes: int | None = None,
) -> str | None:
    """Execute a safe subset of the module and call `build_model`.

    Goal: catch obvious runtime name/symbol errors before remote execution,
    while avoiding side effects like dataset loading/training.
    """
    build_model_fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_model"),
        None,
    )
    if build_model_fn is None:
        if require_build_model:
            return "Missing required function `build_model(num_classes)`."
        return None
    uses_optional_model_libs = _build_model_uses_optional_model_libs(build_model_fn)

    # Static guardrail that works even when torch isn't installed in the
    # validator runtime.
    bad_module_line = _find_returned_bare_module_line(build_model_fn)
    if bad_module_line is not None:
        return (
            "Dry-run build_model failed: `build_model` returns `nn.Module()` "
            f"at line {bad_module_line}. Return a concrete module with forward()."
        )

    if smoke_input_shape:
        try:
            expected_c = int(smoke_input_shape[0])
        except Exception:  # noqa: BLE001
            expected_c = 0
        if expected_c > 0:
            bad_conv = _find_returned_conv2d_in_channels_line(build_model_fn, expected_c)
            if bad_conv is not None:
                got_c, line = bad_conv
                return (
                    "Dry-run build_model likely channel mismatch: "
                    f"input has {expected_c} channel(s) but returned Conv2d uses "
                    f"in_channels={got_c} (line {line})."
                )

    safe_body: list[ast.stmt] = []
    for stmt in tree.body:
        # Never execute main block in dry-run.
        if isinstance(stmt, ast.If) and _is_if_name_main(stmt):
            continue
        # Imports are skipped to avoid local-env dependency drift.
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            # Decorators (e.g., @torch.no_grad()) are evaluated at definition time.
            # In dry-run mode we intentionally avoid executing runtime-heavy or
            # environment-dependent callables, so strip decorators here.
            clone = copy.deepcopy(stmt)
            clone.decorator_list = []
            safe_body.append(clone)
            continue
        # Keep simple constant assignments only.
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, ast.Constant):
                safe_body.append(stmt)
            continue
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is None or isinstance(stmt.value, ast.Constant):
                safe_body.append(stmt)
            continue
        # Keep module docstring expression only.
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            safe_body.append(stmt)

    module = ast.Module(body=safe_body, type_ignores=[])
    ast.fix_missing_locations(module)

    class _NNStub:
        class Module:  # noqa: D106
            pass

        def __getattr__(self, _name):
            def _ctor(*_args, **_kwargs):
                return object()

            return _ctor

    class _TorchStub:
        Tensor = object
        nn = _NNStub()

        def __getattr__(self, _name):
            def _fn(*_args, **_kwargs):
                return object()

            return _fn

    real_torch = None
    real_nn = None
    if smoke_input_shape is not None:
        try:
            import torch as _real_torch  # local import: optional runtime dependency
            real_torch = _real_torch
            real_nn = _real_torch.nn
        except Exception:  # noqa: BLE001
            real_torch = None
            real_nn = None

    ns: dict[str, object] = {
        "__name__": "__validator_dry_run__",
        "nn": real_nn if real_nn is not None else _NNStub(),
        "torch": real_torch if real_torch is not None else _TorchStub(),
    }
    stub_state: dict[str, object | None] = {}
    if uses_optional_model_libs:
        stub_state = _install_module_stubs(_make_optional_model_stubs(real_torch))
    try:
        try:
            code_obj = compile(module, "<validator_dry_run>", "exec")
            exec(code_obj, ns, ns)
        except Exception as exc:  # noqa: BLE001
            return f"Dry-run import/definition failed: {type(exc).__name__}: {exc}"

        fn = ns.get("build_model")
        if not callable(fn):
            return "Dry-run: `build_model` is not callable."

        n_classes = int(smoke_num_classes or 2)
        if n_classes <= 0:
            n_classes = 2

        try:
            try:
                model = fn(n_classes)  # type: ignore[misc]
            except TypeError:
                model = fn(num_classes=n_classes)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            return f"Dry-run build_model failed: {type(exc).__name__}: {exc}"

        if model is None:
            return "Dry-run: `build_model` returned None."

        # Optional forward-pass smoke test for vision/audio tensor models.
        # This catches common runtime failures *before* remote execution:
        # missing forward(), channel mismatches (1ch vs 3ch), etc.
        #
        # For build_model variants that import optional backbone libraries
        # (torchvision/timm), we intentionally skip forward smoke here: the
        # validator may be running in a constrained local env where those
        # libs are stubbed for import determinism.
        if (
            real_torch is not None
            and smoke_input_shape is not None
            and not uses_optional_model_libs
        ):
            try:
                dims = tuple(int(d) for d in smoke_input_shape)
                if any(d <= 0 for d in dims):
                    raise ValueError(f"non-positive shape component in {dims}")
            except Exception as exc:  # noqa: BLE001
                return f"Invalid smoke_input_shape {smoke_input_shape!r}: {exc}"
            try:
                if not isinstance(model, real_torch.nn.Module):
                    return (
                        "Dry-run build_model returned non-torch module "
                        f"({type(model).__name__})."
                    )
                model.eval()
                x = real_torch.zeros((2, *dims), dtype=real_torch.float32)
                with real_torch.no_grad():
                    out = model(x)
                if isinstance(out, (list, tuple)):
                    out = out[0] if out else None
                if out is None or not isinstance(out, real_torch.Tensor):
                    return "Dry-run forward produced non-tensor output."
            except Exception as exc:  # noqa: BLE001
                return (
                    "Dry-run forward failed for input "
                    f"(2, {', '.join(str(d) for d in dims)}): "
                    f"{type(exc).__name__}: {exc}"
                )
        return None
    finally:
        if stub_state:
            _restore_module_stubs(stub_state)


def _check_training_contract(
    tree: ast.Module,
    code: str,
    *,
    spawn_calls: frozenset[str],
) -> str | None:
    """Reject degenerate scripts that don't actually execute training."""
    has_main_fn = any(
        isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body
    )
    if not has_main_fn:
        return "Missing required `main()` training entrypoint."

    main_guards = [
        n for n in tree.body if isinstance(n, ast.If) and _is_if_name_main(n)
    ]
    if not main_guards:
        return "Missing `if __name__ == '__main__': main()` guard."

    calls_main = False
    for guard in main_guards:
        for node in ast.walk(guard):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "main":
                    calls_main = True
                    break
        if calls_main:
            break
    if not calls_main:
        return "Main guard exists but does not call `main()`."

    if "results.json" not in code:
        return "Training script must write `results.json`."

    has_data_loading = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_trigger_name(node, spawn_calls):
            has_data_loading = True
            break
    if not has_data_loading:
        expected = ", ".join(sorted(spawn_calls))
        return f"No data-loading call found (expected one of: {expected})."
    return None


# ---------------------------------------------------------------------------
# __main__ guard (preserved verbatim)
# ---------------------------------------------------------------------------


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
