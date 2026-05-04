"""Track B Kaggle submission builder (ADR-011).

Validates code in submission mode (stricter than the loop-time validator) and
renders a CPU-only notebook from a Jinja template. The notebook contains:
- env cell setting AGENT_DEVICE / offline weight env vars
- the LLM-authored build_model block only
- the skeleton's inference helper code
- a final cell that loads test windows and writes submission.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from lab.config import Settings
from lab.core.models import Study, ValidationResult
from lab.core.validator import Validator
from lab.tasks import get_task_adapter
from lab.tasks.skeleton import (
    _BUILD_END,
    _BUILD_START,
    render_skeleton,
)


class SubmissionValidationError(Exception):
    def __init__(self, message: str, remediation: list[str]):
        super().__init__(message)
        self.remediation = remediation


_TEMPLATE_DIR = Path(__file__).parent


def build_submission_for_study(study: Study, settings: Settings) -> Path:
    """Build the Kaggle notebook for the study's best experiment."""
    if not study.best_experiment_id:
        raise SubmissionValidationError(
            "study has no best_experiment_id; nothing to submit",
            remediation=[
                "Run at least one successful experiment before building a submission."
            ],
        )

    best = next(
        (e for e in study.experiments if e.id == study.best_experiment_id),
        None,
    )
    if best is None or best.code is None:
        raise SubmissionValidationError(
            "best experiment is missing code on disk",
            remediation=[
                "Re-run the study or ensure best_experiment_id points at a stored experiment."
            ],
        )

    adapter = get_task_adapter(settings)
    _validate_for_submission(best.code, adapter, settings)

    out_dir = Path(settings.paths.experiments_dir) / study.id
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered_skeleton = render_skeleton(settings)
    build_block = _extract_build_block(best.code)
    inference_body = _extract_inference_body(rendered_skeleton)

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape([]),
    )
    template = env.get_template("notebook_template.ipynb.j2")
    notebook = template.render(
        env_cell=_env_cell(settings, study),
        build_model_cell=build_block,
        inference_cell=inference_body,
        submission_csv_cell=_csv_cell(settings),
    )

    notebook_path = out_dir / "submission.ipynb"
    notebook_path.write_text(notebook, encoding="utf-8")

    # Validate the rendered file is parseable as a notebook.
    try:
        import nbformat  # type: ignore[import-not-found]

        nbformat.read(str(notebook_path), as_version=4)
    except ImportError:
        json.loads(notebook_path.read_text())  # at least valid JSON
    except Exception as exc:  # pragma: no cover - nbformat quirks
        raise SubmissionValidationError(
            f"rendered notebook failed nbformat validation: {exc}",
            remediation=["Inspect the rendered submission.ipynb manually."],
        ) from exc

    return notebook_path


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _validate_for_submission(code: str, adapter, settings: Settings) -> None:
    validator = Validator(settings)
    result: ValidationResult = validator.validate(
        code,
        signature=adapter.model_block_signature(),
        smoke_input_shape=tuple(settings.task.input_tensor_shape),
        smoke_num_classes=settings.task.expected_num_classes,
        submission_mode=True,
    )
    if not result.ok:
        raise SubmissionValidationError(
            f"submission code rejected: {result.error_type}: {result.message}",
            remediation=[
                f"Fix the {result.error_type} reported by the validator before retrying."
            ]
            + ([result.autofix_hint] if result.autofix_hint else []),
        )


# ---------------------------------------------------------------------------
# notebook cells
# ---------------------------------------------------------------------------


def _env_cell(settings: Settings, study: Study) -> str:
    offline = settings.paths.offline_weights
    lines = [
        "import os",
        "os.environ['AGENT_DEVICE'] = 'cpu'",
        f"os.environ['AGENT_BATCH_SIZE'] = '8'",
        f"os.environ['AGENT_PROCESSED_DIR'] = '/kaggle/input/{settings.task.name}/processed'",
        f"os.environ['TORCH_HOME'] = {offline!r}",
        f"os.environ['HF_HOME'] = {offline!r}",
        f"os.environ['TIMM_HOME'] = {offline!r}",
        f"# Study: {study.id}, best experiment: {study.best_experiment_id}",
    ]
    return "\n".join(lines) + "\n"


def _csv_cell(settings: Settings) -> str:
    return (
        "import pandas as pd\n"
        "import torch\n"
        "from pathlib import Path\n\n"
        "model.eval()\n"
        "rows = []\n"
        "test_dir = Path('/kaggle/input/' + os.environ.get('AGENT_TASK_TESTDIR', 'test'))\n"
        "for path in sorted(test_dir.glob('*.pt')):\n"
        "    blob = torch.load(path, map_location='cpu', weights_only=False)\n"
        "    x = blob['x'].float()\n"
        "    with torch.no_grad():\n"
        "        probs = torch.sigmoid(model(x)).numpy()\n"
        "    for i, p in enumerate(probs):\n"
        "        for cls, prob in enumerate(p):\n"
        "            rows.append({'row_id': f\"{path.stem}_{i}\", 'species_id': cls, 'probability': float(prob)})\n"
        "pd.DataFrame(rows).to_csv('submission.csv', index=False)\n"
        "print('wrote submission.csv with', len(rows), 'rows')\n"
    )


# ---------------------------------------------------------------------------
# code splicing helpers
# ---------------------------------------------------------------------------


def _extract_build_block(code: str) -> str:
    if _BUILD_START not in code or _BUILD_END not in code:
        raise SubmissionValidationError(
            "experiment code is missing AGENT_BUILD_MODEL markers",
            remediation=["Re-render the skeleton and splice the LLM block."],
        )
    _, _, rest = code.partition(_BUILD_START)
    block, _, _ = rest.partition(_BUILD_END)
    return block.strip() + "\n"


def _extract_inference_body(rendered_skeleton: str) -> str:
    """Return the skeleton minus the training loop, leaving inference helpers."""
    _, _, rest = rendered_skeleton.partition(_BUILD_START)
    _, _, after_build = rest.partition(_BUILD_END)
    # Drop the `if __name__ == "__main__":` block so the notebook controls
    # execution explicitly.
    cutoff = after_build.find("if __name__")
    helpers = after_build[:cutoff] if cutoff != -1 else after_build
    return helpers.strip() + "\n\nmodel = build_model(num_classes=NUM_CLASSES)\n"
