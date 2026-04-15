"""SubmissionExporter — packages the best experiment as a Kaggle notebook.

Contract with the Kaggle BirdCLEF 2026 competition:
  - Submission is a Jupyter notebook
  - Must run on CPU only (no CUDA, no JAX, no GPU calls)
  - Must complete within 90 minutes
  - Must produce a CSV submission file in the notebook's working directory

What this module does
---------------------
1. Look up the best experiment in a Study.
2. Read the model architecture from the experiment's code.py.
3. Load the class mapping from the experiment's class_mapping.json.
4. Generate a **self-contained** inference-only notebook with:
     - All model architecture code inlined (no local imports)
     - Audio preprocessing inlined (librosa mel-spectrograms)
     - Model weights loaded from a Kaggle dataset
     - Test soundscapes processed into predictions
     - submission.csv output
5. Copy best_model.pt alongside the notebook for Kaggle dataset upload.

The generated notebook does NOT train — inference only.
"""

from __future__ import annotations

import json
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.models import Experiment, Study


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SubmissionError(Exception):
    """Base class for submission-exporter failures."""


class NoBestExperimentError(SubmissionError):
    pass


class SubmissionValidationError(SubmissionError):
    pass


# ---------------------------------------------------------------------------
# Notebook builder helpers
# ---------------------------------------------------------------------------


def _nbformat_notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an nbformat v4 notebook dict for the given cells."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Model architecture detection
# ---------------------------------------------------------------------------


def _detect_backbone(code: str) -> str:
    """Extract the torchvision backbone name from training code.

    Looks for patterns like TorchvisionAdapter("efficientnet_b0", ...)
    or TorchvisionAdapter('resnet18', ...). Defaults to efficientnet_b0.
    """
    match = re.search(
        r'TorchvisionAdapter\(\s*["\'](\w+)["\']', code
    )
    return match.group(1) if match else "efficientnet_b0"


def _detect_uses_custom_model(code: str) -> bool:
    """Check if the experiment uses a custom model (not TorchvisionAdapter)."""
    return "TorchvisionAdapter" not in code and "CnnSmallV1" not in code


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


@dataclass
class SubmissionExporter:
    """Builds and validates a Kaggle-ready inference notebook."""

    target_runtime_seconds: int = 5400  # 90 min

    def export_best(
        self,
        *,
        study: Study,
        study_dir: Path,
        sandbox_dir: Path,
        output_path: Path | None = None,
    ) -> Path:
        """Find the best experiment and generate a Kaggle submission notebook.

        Also copies best_model.pt to the output directory so it can be
        uploaded to Kaggle as a dataset alongside the notebook.

        Returns the path to the written notebook.
        """
        if not study.best_experiment_id:
            raise NoBestExperimentError(
                f"Study {study.study_id} has no best_experiment_id yet."
            )

        exp_id = study.best_experiment_id
        sandbox_exp = sandbox_dir / exp_id

        # Locate required files
        code_path = sandbox_exp / "code.py"
        if not code_path.exists():
            raise NoBestExperimentError(
                f"Code file not found: {code_path}"
            )
        code = code_path.read_text()

        # Detect model architecture
        backbone = _detect_backbone(code)
        if _detect_uses_custom_model(code):
            print(
                f"WARNING: experiment {exp_id} uses a custom model, "
                "not TorchvisionAdapter/CnnSmallV1. The generated "
                "notebook may need manual editing."
            )

        # Load class mapping (or build from dataset profile)
        class_mapping_path = sandbox_exp / "class_mapping.json"
        if class_mapping_path.exists():
            mapping = json.loads(class_mapping_path.read_text())
            num_classes = mapping["num_classes"]
            class_ids = mapping["class_ids"]
        else:
            # Fallback: read from dataset profile
            profile_path = Path("data/processed/dataset_profile.json")
            if not profile_path.exists():
                raise SubmissionValidationError(
                    f"No class_mapping.json in {sandbox_exp} and no "
                    "dataset_profile.json found. Cannot determine class order."
                )
            profile = json.loads(profile_path.read_text())
            num_classes = profile["num_classes"]
            class_ids = [s["class_id"] for s in profile["class_stats"]]

        # Load taxonomy for full 234-species column list
        taxonomy_path = Path("data/raw/taxonomy.csv")
        if taxonomy_path.exists():
            import csv
            with taxonomy_path.open() as f:
                reader = csv.DictReader(f)
                all_species = [row["primary_label"] for row in reader]
        else:
            # Fallback: read from sample_submission.csv header
            sample_sub = Path("data/raw/sample_submission.csv")
            if sample_sub.exists():
                with sample_sub.open() as f:
                    header = f.readline().strip().split(",")
                    all_species = header[1:]  # skip row_id
            else:
                raise SubmissionValidationError(
                    "Cannot find taxonomy.csv or sample_submission.csv "
                    "to determine submission column order."
                )

        # Output path
        if output_path is None:
            output_path = (
                study_dir / "submissions" / "submission.ipynb"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy model weights if they exist
        weights_src = sandbox_exp / "best_model.pt"
        weights_dst = output_path.parent / "best_model.pt"
        has_weights = weights_src.exists()
        if has_weights:
            shutil.copy2(weights_src, weights_dst)
            print(f"Copied model weights to {weights_dst}")
        else:
            print(
                f"WARNING: No best_model.pt found in {sandbox_exp}. "
                "The notebook will need manual weight attachment."
            )

        # Copy optimized thresholds if they exist
        thresholds_src = sandbox_exp / "best_thresholds.npy"
        thresholds_dst = output_path.parent / "best_thresholds.npy"
        has_thresholds = thresholds_src.exists()
        if has_thresholds:
            shutil.copy2(thresholds_src, thresholds_dst)
            print(f"Copied optimized thresholds to {thresholds_dst}")

        # Build notebook
        cells = self._build_inference_cells(
            study=study,
            exp_id=exp_id,
            backbone=backbone,
            num_classes=num_classes,
            class_ids=class_ids,
            all_species=all_species,
            has_weights=has_weights,
            has_thresholds=has_thresholds,
        )
        notebook = _nbformat_notebook(cells)
        output_path.write_text(json.dumps(notebook, indent=1))
        return output_path

    # -- notebook cell construction -----------------------------------------

    def _build_inference_cells(
        self,
        *,
        study: Study,
        exp_id: str,
        backbone: str,
        num_classes: int,
        class_ids: list[str],
        all_species: list[str],
        has_weights: bool,
        has_thresholds: bool = False,
    ) -> list[dict[str, Any]]:
        """Build a self-contained inference-only notebook."""

        best_score = ""
        if study.best_score is not None:
            best_score = f"\n**Best F1:** {study.best_score:.4f}"

        header = _markdown_cell(textwrap.dedent(f"""\
            # BirdCLEF 2026 — Submission Notebook

            **Study:** `{study.study_id}`
            **Experiment:** `{exp_id}`{best_score}
            **Generated by:** autonomous research agent

            This notebook is **self-contained** — no local imports needed.
            It loads pre-trained model weights, processes test soundscapes,
            and writes `submission.csv`. Runs on CPU within 90 minutes.

            ## Setup

            Before submitting to Kaggle:
            1. Upload `best_model.pt` as a Kaggle dataset (e.g., `your-username/birdclef-model`)
            2. Attach that dataset to this notebook
            3. Update `WEIGHTS_PATH` below to match the dataset path
        """))

        config_cell = _code_cell(textwrap.dedent(f"""\
            import os
            import json
            import time
            import numpy as np
            import pandas as pd
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            # === Force CPU-only execution ===
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = torch.device("cpu")
            torch.set_num_threads(os.cpu_count() or 4)

            # === Paths — update WEIGHTS_PATH for your Kaggle dataset ===
            # On Kaggle, attached datasets are at /kaggle/input/<dataset-name>/
            WEIGHTS_PATH = "/kaggle/input/birdclef-model/best_model.pt"
            THRESHOLDS_PATH = "/kaggle/input/birdclef-model/best_thresholds.npy"
            TEST_SOUNDSCAPES = "/kaggle/input/birdclef-2026/test_soundscapes"
            SAMPLE_SUBMISSION = "/kaggle/input/birdclef-2026/sample_submission.csv"

            # === Audio preprocessing config (must match training) ===
            SAMPLE_RATE = 32000
            N_MELS = 128
            N_FFT = 2048
            HOP_LENGTH = 512
            FMIN = 20.0
            WINDOW_SECONDS = 5.0
            TOP_DB = 80.0

            # === Model config ===
            NUM_CLASSES = {num_classes}
            BACKBONE = "{backbone}"

            # === Class mapping: model output index -> species ID ===
            CLASS_IDS = {json.dumps(class_ids)}

            print(f"Device: {{device}}")
            print(f"Weights: {{WEIGHTS_PATH}}")
            print(f"Test dir: {{TEST_SOUNDSCAPES}}")
            print(f"Classes: {{NUM_CLASSES}} (trained) / {len(all_species)} (submission)")
        """))

        model_cell_header = _markdown_cell(
            "## Model architecture\n\n"
            f"TorchvisionAdapter wrapping `{backbone}` — defined inline, "
            "no local imports needed."
        )

        model_cell = _code_cell(textwrap.dedent("""\
            import torchvision.models as tvm

            class TorchvisionAdapter(nn.Module):
                \"\"\"Wraps a torchvision backbone for single-channel spectrograms.

                Input: (B, 1, 128, 313) -> resize to (B, 3, 224, 224) -> backbone -> head.
                \"\"\"
                def __init__(self, backbone_name, num_classes, pretrained=False):
                    super().__init__()
                    ctor = getattr(tvm, backbone_name)
                    weights = None
                    if pretrained:
                        weights_cls_name = "".join(
                            p.capitalize() for p in backbone_name.split("_")
                        ) + "_Weights"
                        weights_cls = getattr(tvm, weights_cls_name, None)
                        if weights_cls is not None:
                            weights = getattr(weights_cls, "DEFAULT", None)
                    try:
                        backbone = ctor(weights=weights)
                    except Exception:
                        backbone = ctor(weights=None)

                    # Replace classifier head with Identity
                    if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Module):
                        backbone.fc = nn.Identity()
                    elif hasattr(backbone, "classifier") and isinstance(backbone.classifier, nn.Module):
                        backbone.classifier = nn.Identity()

                    self.backbone = backbone
                    self.head = nn.LazyLinear(num_classes)
                    self.input_size = (224, 224)

                def forward(self, x):
                    if x.size(1) == 1:
                        x = x.expand(-1, 3, -1, -1)
                    if x.shape[-2:] != self.input_size:
                        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
                    features = self.backbone(x)
                    if features.dim() > 2:
                        features = features.flatten(1)
                    return self.head(features)

            # This wrapper matches the EXACT structure used during training.
            # The LLM-generated code.py wraps TorchvisionAdapter in an extra
            # EfficientNetHead class, producing state_dict keys like
            # backbone.backbone.features... — we must replicate that nesting.
            class EfficientNetHead(nn.Module):
                def __init__(self, backbone):
                    super().__init__()
                    self.backbone = backbone
                    self.head = nn.LazyLinear(NUM_CLASSES)

                def forward(self, x):
                    x = self.backbone(x)
                    x = x.flatten(1)
                    return self.head(x)

            # Instantiate with the same double-wrapped structure as training
            backbone = TorchvisionAdapter(BACKBONE, NUM_CLASSES)
            model = EfficientNetHead(backbone=backbone)
            model = model.to(device)

            # Initialize lazy layers with a dummy forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, 1, N_MELS, 313, device=device)
                model(dummy)

            # Load trained weights
            if os.path.exists(WEIGHTS_PATH):
                state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                print(f"Loaded weights from {WEIGHTS_PATH}")
            else:
                print(f"WARNING: {WEIGHTS_PATH} not found! Using random weights.")

            model.eval()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {BACKBONE} with {n_params:,} parameters")
        """))

        preprocess_header = _markdown_cell(
            "## Audio preprocessing\n\n"
            "Converts raw audio to log-mel-spectrograms using the same "
            "parameters as training."
        )

        preprocess_cell = _code_cell(textwrap.dedent("""\
            import librosa

            def audio_to_spectrograms(audio_path, sr=SAMPLE_RATE):
                \"\"\"Load audio and split into 5-second mel-spectrogram windows.

                Returns list of (window_end_seconds, spectrogram) tuples.
                \"\"\"
                y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
                y = y.astype(np.float32)

                window_samples = int(sr * WINDOW_SECONDS)
                specs = []
                start = 0
                window_idx = 0

                while start + window_samples <= len(y):
                    chunk = y[start : start + window_samples]
                    mel = librosa.feature.melspectrogram(
                        y=chunk, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        n_mels=N_MELS, fmin=FMIN, fmax=sr / 2, power=2.0,
                    )
                    mel_db = librosa.power_to_db(mel, top_db=TOP_DB).astype(np.float32)
                    end_seconds = (start + window_samples) // sr * WINDOW_SECONDS
                    # BirdCLEF row_id uses end time: 5, 10, 15, ...
                    end_sec = int((window_idx + 1) * WINDOW_SECONDS)
                    specs.append((end_sec, mel_db))
                    start += window_samples
                    window_idx += 1

                # Handle remaining audio (pad if needed)
                if start < len(y) and len(y) - start > sr:  # at least 1 second
                    chunk = np.zeros(window_samples, dtype=np.float32)
                    remaining = y[start:]
                    chunk[:len(remaining)] = remaining
                    mel = librosa.feature.melspectrogram(
                        y=chunk, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        n_mels=N_MELS, fmin=FMIN, fmax=sr / 2, power=2.0,
                    )
                    mel_db = librosa.power_to_db(mel, top_db=TOP_DB).astype(np.float32)
                    end_sec = int((window_idx + 1) * WINDOW_SECONDS)
                    specs.append((end_sec, mel_db))

                return specs

            print("Audio preprocessing ready")
        """))

        # Build the class-id to submission-column mapping
        inference_header = _markdown_cell(
            "## Inference on test soundscapes\n\n"
            "Process each test soundscape, predict species probabilities "
            "using optimized per-class thresholds, and build the submission "
            "DataFrame."
        )

        all_species_json = json.dumps(all_species)
        inference_cell = _code_cell(textwrap.dedent(f"""\
            from pathlib import Path
            import glob

            # === TTA (Test-Time Augmentation) config ===
            TTA_ENABLED = True
            TTA_N_AUGMENTS = 5  # number of augmented views per window

            def tta_predict(model, spec_tensor, n_augments=TTA_N_AUGMENTS):
                \"\"\"Average predictions over multiple augmented views of the spectrogram.

                Augmentations: original + random time shifts + small Gaussian noise.
                This reduces variance from arbitrary window boundaries.
                \"\"\"
                if not TTA_ENABLED or n_augments <= 1:
                    with torch.inference_mode():
                        logits = model(spec_tensor)
                        return torch.sigmoid(logits).cpu().numpy()[0]

                all_probs = []
                spec_np = spec_tensor.cpu().numpy()[0, 0]  # (n_mels, T)

                for i in range(n_augments):
                    aug = spec_np.copy()
                    if i > 0:
                        max_shift = int(aug.shape[1] * 0.2)
                        shift = np.random.randint(-max_shift, max_shift + 1)
                        aug = np.roll(aug, shift, axis=1)
                        aug = aug + np.random.normal(0, 0.005, size=aug.shape).astype(np.float32)

                    t = torch.from_numpy(aug).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        logits = model(t)
                        probs = torch.sigmoid(logits).cpu().numpy()[0]
                    all_probs.append(probs)

                return np.mean(all_probs, axis=0)

            # Full species list for submission (234 species)
            ALL_SPECIES = {all_species_json}

            # NOTE: Competition metric is macro-averaged ROC-AUC, so we output
            # raw probabilities, NOT binary predictions.
            print(f"TTA: {{'enabled' if TTA_ENABLED else 'disabled'}} ({{TTA_N_AUGMENTS}} augments)")
            print("Submission mode: outputting raw probabilities (ROC-AUC metric)")

            # Map our model's class indices to submission column positions
            model_idx_to_sub_col = {{}}
            for model_idx, cid in enumerate(CLASS_IDS):
                if cid in ALL_SPECIES:
                    sub_col = ALL_SPECIES.index(cid)
                    model_idx_to_sub_col[model_idx] = sub_col

            print(f"Mapped {{len(model_idx_to_sub_col)}}/{{NUM_CLASSES}} model classes to submission columns")

            # Read sample submission to get the expected row_ids
            sample_sub = pd.read_csv(SAMPLE_SUBMISSION)
            expected_row_ids = set(sample_sub["row_id"].tolist())
            print(f"Expected {{len(expected_row_ids)}} rows in submission")

            # Process test soundscapes
            test_dir = Path(TEST_SOUNDSCAPES)
            audio_files = sorted(glob.glob(str(test_dir / "*.ogg")))
            if not audio_files:
                audio_files = sorted(glob.glob(str(test_dir / "*.wav")))
            if not audio_files:
                audio_files = sorted(glob.glob(str(test_dir / "*.flac")))

            print(f"Found {{len(audio_files)}} test soundscape files")

            rows = []
            start_time = time.time()

            for file_idx, audio_path in enumerate(audio_files):
                filename = Path(audio_path).stem
                specs = audio_to_spectrograms(audio_path)

                for end_sec, mel_db in specs:
                    row_id = f"{{filename}}_{{end_sec}}"

                    if row_id not in expected_row_ids:
                        continue

                    spec_tensor = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).to(device)

                    # Use TTA for more robust predictions
                    probs = tta_predict(model, spec_tensor)

                    row = np.zeros(len(ALL_SPECIES), dtype=np.float32)
                    for model_idx, sub_col in model_idx_to_sub_col.items():
                        row[sub_col] = float(probs[model_idx])

                    rows.append({{"row_id": row_id, **dict(zip(ALL_SPECIES, row))}})

                if (file_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Processed {{file_idx + 1}}/{{len(audio_files)}} files ({{elapsed:.0f}}s)")

            elapsed = time.time() - start_time
            print(f"Inference complete: {{len(rows)}} predictions in {{elapsed:.0f}}s")
        """))

        submission_header = _markdown_cell("## Write submission.csv")

        submission_cell = _code_cell(textwrap.dedent("""\
            # Build submission DataFrame
            if rows:
                submission = pd.DataFrame(rows)
            else:
                # No test files found (normal during Kaggle draft mode).
                # Create an empty DataFrame with the correct columns.
                submission = pd.DataFrame(columns=["row_id"] + ALL_SPECIES)

            # Ensure all expected row_ids are present (fill missing with uniform prior)
            if len(submission) < len(sample_sub):
                existing_ids = set(submission["row_id"]) if len(submission) > 0 else set()
                missing_ids = set(sample_sub["row_id"]) - existing_ids
                if missing_ids:
                    print(f"WARNING: {len(missing_ids)} missing row_ids, filling with uniform prior")
                    uniform = 1.0 / len(ALL_SPECIES)
                    fill_rows = []
                    for rid in missing_ids:
                        row = {"row_id": rid}
                        row.update({sp: uniform for sp in ALL_SPECIES})
                        fill_rows.append(row)
                    submission = pd.concat([submission, pd.DataFrame(fill_rows)], ignore_index=True)

            # Ensure correct column order
            submission = submission[["row_id"] + ALL_SPECIES]

            # Save
            submission.to_csv("submission.csv", index=False)
            print(f"Wrote submission.csv: {submission.shape[0]} rows x {submission.shape[1]} columns")
            print(submission.head())
        """))

        return [
            header,
            config_cell,
            model_cell_header,
            model_cell,
            preprocess_header,
            preprocess_cell,
            inference_header,
            inference_cell,
            submission_header,
            submission_cell,
        ]


__all__ = [
    "SubmissionExporter",
    "SubmissionError",
    "NoBestExperimentError",
    "SubmissionValidationError",
]
