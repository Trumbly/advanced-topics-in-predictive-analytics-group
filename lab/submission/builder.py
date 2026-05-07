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
    prologue = _extract_skeleton_prologue(rendered_skeleton)
    helpers = _extract_post_build_helpers(rendered_skeleton)

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape([]),
    )
    template = env.get_template("notebook_template.ipynb.j2")
    notebook = template.render(
        env_cell=_env_cell(settings, study),
        prologue_cell=prologue,
        build_model_cell=build_block,
        inference_cell=helpers + _model_init_cell(),
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
    """Notebook environment cell: GPU when available, larger batch for
    inference (no gradients), and an `AGENT_WEIGHTS_PATH` knob the user
    can point at the Kaggle Dataset where they uploaded the trained
    checkpoint -- see header comments below."""
    offline = settings.paths.offline_weights
    lines = [
        f"# Submission notebook for study {study.id}",
        f"# Best experiment: {study.best_experiment_id}",
        "#",
        "# WORKFLOW (BirdCLEF+ 2026 is a code competition — INFERENCE-ONLY):",
        "#  1. Train the model locally (no time limit).",
        "#  2. Upload the trained checkpoint as a Kaggle Dataset (Settings ->",
        "#     'Add Data'). Default path searched: /kaggle/input/**/*.pt",
        "#  3. Point AGENT_WEIGHTS_PATH at the file if it has a non-standard",
        "#     name, OR rename your upload to checkpoint.pt to use the default.",
        "#  4. The notebook only does inference -- runtime budget per Kaggle",
        "#     code-comp rules covers prediction over test_soundscapes only.",
        "import os",
        "import torch",
        "os.environ.setdefault('AGENT_DEVICE',",
        "                       'cuda' if torch.cuda.is_available() else 'cpu')",
        "os.environ.setdefault('AGENT_BATCH_SIZE', '64')",
        "os.environ.setdefault('AGENT_NUM_WORKERS', '2')",
        f"os.environ.setdefault('AGENT_PROCESSED_DIR',",
        f"                       '/kaggle/input/{settings.task.name}/processed')",
        f"os.environ.setdefault('TORCH_HOME', {offline!r})",
        f"os.environ.setdefault('HF_HOME', {offline!r})",
        f"os.environ.setdefault('TIMM_HOME', {offline!r})",
    ]
    return "\n".join(lines) + "\n"


def _csv_cell(settings: Settings) -> str:
    """Wide-format submission CSV generator.

    The Kaggle BirdCLEF+ 2026 submission is one row per 5-second window
    with one column per species (header taken verbatim from
    ``sample_submission.csv``). Each value is the predicted probability
    that the species is present in that window.

    Test files live at ``/kaggle/input/birdclef-2026/test_soundscapes/*.ogg``
    (60 s @ 32 kHz). We slice each into twelve 5 s windows, mel-spec each
    with the same ``MelParams`` the training cache used (sr=32k, n_fft=2048,
    hop=512, n_mels=128, dB scaling), feed through the model, and emit
    ``BC2026_Test_NNNN_..._{end_seconds}`` row IDs matching the
    sample_submission convention.
    """
    return (
        "import os\n"
        "import csv\n"
        "import numpy as np\n"
        "import torch\n"
        "import librosa\n"
        "from pathlib import Path\n\n"
        "BASE = Path(os.environ.get('AGENT_KAGGLE_BASE', '/kaggle/input/birdclef-2026'))\n"
        "TEST_DIR = BASE / 'test_soundscapes'\n"
        "SUBMISSION_TEMPLATE = BASE / 'sample_submission.csv'\n\n"
        "with open(SUBMISSION_TEMPLATE) as fh:\n"
        "    header = next(csv.reader(fh))\n"
        "species_columns = header[1:]   # canonical 234-class order from sample_submission\n"
        "n_species = len(species_columns)\n\n"
        "SR = 32000\n"
        "WINDOW_SECONDS = 5\n"
        "WINDOW_SAMPLES = SR * WINDOW_SECONDS\n"
        "N_MELS = 128\n"
        "N_FFT = 2048\n"
        "HOP_LENGTH = 512\n"
        "FMIN = 20\n"
        "FMAX = 16000\n\n"
        "def _mel(chunk):\n"
        "    mel = librosa.feature.melspectrogram(\n"
        "        y=chunk, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,\n"
        "        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,\n"
        "    )\n"
        "    return librosa.power_to_db(mel, ref=np.max).astype('float32')\n\n"
        "model.eval()\n"
        "rows = []\n"
        "for path in sorted(TEST_DIR.glob('*.ogg')):\n"
        "    base = path.stem\n"
        "    audio, _ = librosa.load(path, sr=SR, mono=True)\n"
        "    n_windows = len(audio) // WINDOW_SAMPLES\n"
        "    if n_windows == 0:\n"
        "        continue\n"
        "    mels = np.stack([\n"
        "        _mel(audio[i * WINDOW_SAMPLES:(i + 1) * WINDOW_SAMPLES])\n"
        "        for i in range(n_windows)\n"
        "    ])[:, None, :, :]   # (n_windows, 1, n_mels, n_frames)\n"
        "    x = torch.from_numpy(mels)\n"
        "    with torch.no_grad():\n"
        "        probs = torch.sigmoid(model(x)).numpy()\n"
        "    if probs.shape[1] < n_species:\n"
        "        # Pad missing species with the dataset prior (0.5 = uniform).\n"
        "        pad = np.full((probs.shape[0], n_species - probs.shape[1]), 0.5,\n"
        "                      dtype='float32')\n"
        "        probs = np.concatenate([probs, pad], axis=1)\n"
        "    elif probs.shape[1] > n_species:\n"
        "        probs = probs[:, :n_species]\n"
        "    for i, p in enumerate(probs):\n"
        "        end_seconds = (i + 1) * WINDOW_SECONDS\n"
        "        row_id = f\"{base}_{end_seconds}\"\n"
        "        row = {'row_id': row_id}\n"
        "        for sp, val in zip(species_columns, p):\n"
        "            row[sp] = float(val)\n"
        "        rows.append(row)\n\n"
        "with open('submission.csv', 'w', newline='') as fh:\n"
        "    writer = csv.DictWriter(fh, fieldnames=header)\n"
        "    writer.writeheader()\n"
        "    writer.writerows(rows)\n"
        "print('wrote submission.csv with', len(rows), 'rows ×', len(header), 'cols')\n"
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
    """Back-compat wrapper; new builds use prologue + helpers separately."""
    return _extract_post_build_helpers(rendered_skeleton) + _model_init_cell()


def _extract_skeleton_prologue(rendered_skeleton: str) -> str:
    """Everything BEFORE the build_model markers: imports + constants
    (NUM_CLASSES, INPUT_SHAPE, env-knob globals) + LazyMelDataset and
    related dataset helpers. Required so the notebook's later cells can
    reference these names — the previous build only kept post-build
    helpers, which made `model = build_model(num_classes=NUM_CLASSES)`
    crash with NameError on Kaggle."""
    pre, _, _ = rendered_skeleton.partition(_BUILD_START)
    return pre.rstrip() + "\n"


def _extract_post_build_helpers(rendered_skeleton: str) -> str:
    """Skeleton block AFTER the build_model marker, minus the training-only
    main() invocation. Keeps metric helpers + ``_ensure_channel_compat`` +
    the inference machinery. Drops the train loop because the notebook
    only does inference on Kaggle."""
    _, _, rest = rendered_skeleton.partition(_BUILD_START)
    _, _, after_build = rest.partition(_BUILD_END)
    cutoff = after_build.find("if __name__")
    helpers = after_build[:cutoff] if cutoff != -1 else after_build
    return helpers.strip() + "\n"


def _model_init_cell() -> str:
    """Build the model + apply the channel adapter + load the trained
    weights from the Kaggle dataset path. Runs after the prologue + the
    LLM-authored build_model + the post-build helpers are all in scope."""
    return (
        "\n# ---------- model init ----------\n"
        "model = build_model(num_classes=NUM_CLASSES)\n"
        "if '_ensure_channel_compat' in globals():\n"
        "    model = _ensure_channel_compat(model, INPUT_SHAPE[0])\n"
        "_weights_path = os.environ.get(\n"
        "    'AGENT_WEIGHTS_PATH',\n"
        "    str(next(Path('/kaggle/input').rglob('*.pt'), Path('checkpoint.pt'))),\n"
        ")\n"
        "_state = torch.load(_weights_path, map_location='cpu', weights_only=False)\n"
        "if isinstance(_state, dict) and 'state_dict' in _state:\n"
        "    _state = _state['state_dict']\n"
        "model.load_state_dict(_state)\n"
        "model.eval()\n"
        "print(f'[model] loaded weights from {_weights_path}')\n"
    )


# ---------------------------------------------------------------------------
# local CSV inference -- run the best model on data/raw/test_soundscapes
# ---------------------------------------------------------------------------


def build_local_csv_for_study(study: Study, settings: Settings) -> Path:
    """Run the best experiment's model locally and write submission.csv.

    Reads ``data/raw/test_soundscapes/*.ogg``, slices each into 5 s windows,
    mel-specs each window with the same params as training, runs the model,
    and writes a wide-format ``submission.csv`` next to the notebook.

    Returns the path to the written CSV. Raises ``SubmissionValidationError``
    when prerequisites (best experiment, checkpoint, raw test dir) are
    missing -- callers can show those errors to the user without crashing
    the request.

    The function is import-safe: torch / librosa are loaded lazily inside
    the body so test environments without them can still import the
    builder module.
    """
    if not study.best_experiment_id:
        raise SubmissionValidationError(
            "study has no best_experiment_id; nothing to submit",
            remediation=["Run at least one successful experiment first."],
        )
    best = next(
        (e for e in study.experiments if e.id == study.best_experiment_id),
        None,
    )
    if best is None or best.code is None:
        raise SubmissionValidationError(
            "best experiment is missing code on disk",
            remediation=["Re-run the study to repopulate experiment.code."],
        )
    if not best.checkpoint_path or not Path(best.checkpoint_path).exists():
        raise SubmissionValidationError(
            "best experiment has no usable checkpoint on disk",
            remediation=[
                "Make sure training wrote AGENT_CHECKPOINT_OUT and the file "
                "still exists.",
            ],
        )

    raw_test = Path(settings.task.processed_data_dir).parent.parent / "raw" / "test_soundscapes"
    if not raw_test.exists():
        raise SubmissionValidationError(
            f"missing test audio dir: {raw_test}",
            remediation=[
                "Download the BirdCLEF+ 2026 dump and place "
                "test_soundscapes/ under data/raw/.",
            ],
        )

    sample_sub = Path(settings.task.processed_data_dir).parent.parent / "raw" / "sample_submission.csv"
    if not sample_sub.exists():
        raise SubmissionValidationError(
            f"missing {sample_sub}",
            remediation=["Place sample_submission.csv from the Kaggle dump under data/raw/."],
        )

    try:
        import csv as _csv
        import importlib.util
        import sys

        import librosa  # type: ignore[import-not-found]
        import numpy as np
        import torch
    except ImportError as exc:
        raise SubmissionValidationError(
            f"local inference requires librosa, numpy, torch: {exc}",
            remediation=[
                "Install the optional ML deps via `uv pip install -e .`.",
            ],
        )

    out_dir = Path(settings.paths.experiments_dir) / study.id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the LLM-built model from the experiment's stored code (the
    # full skeleton with the LLM block already spliced in). We import it
    # as an isolated module so its train-time main() does not run.
    model_path = out_dir / "submission_model.py"
    model_path.write_text(best.code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        f"_submission_{study.id}", model_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    num_classes = settings.task.expected_num_classes
    model = module.build_model(num_classes=num_classes)
    if hasattr(module, "_ensure_channel_compat"):
        model = module._ensure_channel_compat(
            model, settings.task.input_tensor_shape[0]
        )
    state = torch.load(best.checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    with sample_sub.open() as fh:
        header = next(_csv.reader(fh))
    species_columns = header[1:]
    n_species = len(species_columns)

    sr = 32_000
    win_sec = 5
    win_samples = sr * win_sec
    n_fft, hop, n_mels, fmin, fmax = 2048, 512, 128, 20, 16_000

    def _mel(chunk):
        m = librosa.feature.melspectrogram(
            y=chunk, sr=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, fmin=fmin, fmax=fmax,
        )
        return librosa.power_to_db(m, ref=np.max).astype("float32")

    rows: list[dict] = []
    for path in sorted(raw_test.glob("*.ogg")):
        base = path.stem
        audio, _ = librosa.load(path, sr=sr, mono=True)
        n_windows = len(audio) // win_samples
        if n_windows == 0:
            continue
        mels = np.stack([
            _mel(audio[i * win_samples:(i + 1) * win_samples])
            for i in range(n_windows)
        ])[:, None, :, :]
        x = torch.from_numpy(mels)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).numpy()
        if probs.shape[1] < n_species:
            pad = np.full((probs.shape[0], n_species - probs.shape[1]), 0.5, dtype="float32")
            probs = np.concatenate([probs, pad], axis=1)
        elif probs.shape[1] > n_species:
            probs = probs[:, :n_species]
        for i, p in enumerate(probs):
            end_seconds = (i + 1) * win_sec
            row = {"row_id": f"{base}_{end_seconds}"}
            for sp, val in zip(species_columns, p):
                row[sp] = float(val)
            rows.append(row)

    csv_path = out_dir / "submission.csv"
    with csv_path.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
