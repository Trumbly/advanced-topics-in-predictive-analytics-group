# I-07 — Validator (static + smoke)

**Labels:** `track-loop`, `p0`
**Milestone:** week-2-loop-and-task
**Owner:** Dev B

## Context
ADR-005. Prototype validator ballooned to 1024 LOC. Trim to 3 static checks + smoke forward pass.

## Scope
Keep:
1. Forbidden imports (`subprocess`, `socket`, `urllib`, `requests`, `pip`, etc.)
2. `build_model(num_classes)` signature present and matches `adapter.model_block_signature()`
3. `torch.nn` attribute reflection (reject hallucinated layers like `nn.Conv2x2d`)

Plus:
4. Smoke forward pass: instantiate model, run forward on dummy tensor with `smoke_input_shape`, verify output shape is `(batch, smoke_num_classes)`.

## Interface
```python
@dataclass
class ValidationResult:
    ok: bool
    error_type: str | None    # "ForbiddenImport"|"BadSignature"|"UnknownTorchNN"|"SmokeFailed"|"Syntax"
    message: str | None
    findings: list[str] = field(default_factory=list)
    autofix_hint: str | None = None   # e.g. "rename nn.Conv2x2d → nn.Conv2d"

class Validator:
    def __init__(self, settings: Settings): ...
    def validate(self, code: str, *,
                 signature: tuple[str, str],          # ("build_model", "num_classes")
                 smoke_input_shape: tuple[int, ...],
                 smoke_num_classes: int,
                 submission_mode: bool = False) -> ValidationResult: ...
```

`submission_mode=True`: stricter — also forbid `requests`, `urllib`, `pip`, `wget`, network I/O, `!pip install`.

## Files
- Rewrite `lab/core/validator.py` (≤400 LOC target)
- Create `tests/test_validator.py`
- Create `tests/fixtures/code/` with: `ok.py`, `forbidden_import.py`, `bad_signature.py`, `hallucinated_nn.py`, `shape_smoke_fail.py`

## Acceptance
- All fixtures classified correctly.
- `autofix_hint` set for `UnknownTorchNN` and `BadSignature`.
- Submission-mode test rejects `urllib.request` even if loop-mode accepts it.
- Smoke test runs model forward on CPU in <5 s.

## Depends on
I-01.
