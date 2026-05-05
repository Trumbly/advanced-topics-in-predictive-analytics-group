"""Track B BirdCLEF+ 2026 adapter (ADR-003).

Reads `data/processed/mels/metadata.parquet` for the dataset profile when
present; falls back to the values declared in `config/tasks/track_b.yaml` so
the agent loop can run on synthetic shards before preprocessing has been done.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from lab.config import Settings
from lab.core.models import DatasetProfile
from lab.tasks.base import TaskAdapter


class BirdclefAdapter(TaskAdapter):
    name = "track_b"
    kind = "audio_multilabel"
    primary_metric = "roc_auc_macro"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._metadata_path = (
            Path(settings.task.processed_data_dir) / "metadata.parquet"
        )

    # ----------------------------------------------------------------- profile

    def profile(self) -> DatasetProfile:
        meta = _read_metadata(self._metadata_path)
        if meta is not None:
            num_classes = int(meta["num_classes"])
            num_train = int(meta["num_train"])
            shape = tuple(int(x) for x in meta["input_tensor_shape"])
            class_imbalance = meta.get("class_imbalance")
        else:
            num_classes = self.settings.task.expected_num_classes
            num_train = 0
            shape = tuple(self.settings.task.input_tensor_shape)
            class_imbalance = None

        if num_classes != self.settings.task.expected_num_classes:
            # The canonical class set comes from sample_submission.csv and
            # is bumped to 234 to include soundscape-only species. A stale
            # metadata sidecar reporting 206 (the old training-only count)
            # is a downgrade, not an error -- warn and use the expected
            # value so the model head is sized for the full submission.
            import logging

            logging.getLogger("lab.task.track_b").warning(
                "BirdCLEF metadata reports %d classes but config expects "
                "%d; using %d (rebuild metadata via `lab preprocess` to "
                "silence this)",
                num_classes,
                self.settings.task.expected_num_classes,
                self.settings.task.expected_num_classes,
            )
            num_classes = self.settings.task.expected_num_classes
        return DatasetProfile(
            num_classes=num_classes,
            num_train=num_train,
            input_tensor_shape=shape,
            class_imbalance=class_imbalance,
        )

    # -------------------------------------------------------------- prompt slots

    def prompt_slots(self) -> dict[str, str]:
        shape = tuple(self.settings.task.input_tensor_shape)
        families = ", ".join(self.settings.task.valid_architecture_families)
        skeleton = _read_skeleton_excerpt(Path(self.settings.task.skeleton_path))
        return {
            "task_description": self.settings.task.task_description.strip(),
            "num_classes": str(self.settings.task.expected_num_classes),
            "input_tensor_shape": str(shape),
            "valid_architecture_families": families,
            "code_skeleton_content": skeleton,
        }

    def model_block_signature(self) -> tuple[str, str]:
        return ("build_model", "num_classes")

    def spawn_triggering_calls(self) -> Iterable[str]:
        return ("DataLoader", "MultiProcessingDataLoaderIter", "torch.multiprocessing")

    # --------------------------------------------------------------- submission

    def build_submission(self, code: str, experiment_id: str, out_dir: Path) -> Path:
        """Submission stub - the actual notebook builder lives in I-14."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "submission.ipynb"
        if not target.exists():
            target.write_text("{}\n", encoding="utf-8")
        return target


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_metadata(path: Path) -> dict | None:
    """Return metadata dict if a parquet (or json) sidecar exists; else None.

    Real code uses pandas.read_parquet, but pulling pandas in just to surface a
    schema is wasteful; we accept either a JSON sidecar or fall back to None.
    """
    if not path.exists():
        json_sidecar = path.with_suffix(".json")
        if json_sidecar.exists():
            import json

            return json.loads(json_sidecar.read_text(encoding="utf-8"))
        return None

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return None

    df = pd.read_parquet(path)
    classes = df["primary_label"].nunique() if "primary_label" in df.columns else None
    num_train = len(df)
    if "n_mels" in df.columns and "n_frames" in df.columns:
        shape = (1, int(df["n_mels"].iloc[0]), int(df["n_frames"].iloc[0]))
    else:
        shape = None
    if classes is None or shape is None:
        return None
    counts = (
        df["primary_label"].value_counts().to_dict()
        if "primary_label" in df.columns
        else None
    )
    return {
        "num_classes": int(classes),
        "num_train": int(num_train),
        "input_tensor_shape": list(shape),
        "class_imbalance": {str(k): int(v) for k, v in counts.items()} if counts else None,
    }


def _read_skeleton_excerpt(path: Path, *, max_chars: int = 1500) -> str:
    if not path.exists():
        return "(skeleton not yet on disk; will be rendered by I-11)"
    text = path.read_text(encoding="utf-8")
    return text[:max_chars]
