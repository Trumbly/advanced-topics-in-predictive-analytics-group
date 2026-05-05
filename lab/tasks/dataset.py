"""Dataset bootstrap helpers.

The training skeleton loads ``processed_dir/train.pt`` and ``val.pt``.
When the real shards are missing (e.g. demo run on a fresh clone with no
Kaggle data downloaded), the agent loop would otherwise fail at the
executor stage with ``FileNotFoundError: missing processed shard``.

``ensure_dataset_present`` returns a possibly-rewritten ``Settings`` whose
``task.processed_data_dir`` points at a *separate* synthetic directory
(``<real>_synthetic`` next to the real one) when the real shard is absent.
The real ``data/processed/mels/`` is never written to from this helper, so
real Kaggle mels and synthetic stand-ins stay strictly disjoint on disk.
"""

from __future__ import annotations

from pathlib import Path

from lab.config import Settings

_DEFAULT_TRAIN_SAMPLES = 200
_DEFAULT_VAL_SAMPLES = 50

_SYNTHETIC_SUFFIX = "_synthetic"


def synthetic_dir_for(real_processed_dir: Path | str) -> Path:
    """Return the synthetic-shard directory parallel to ``real_processed_dir``."""
    real = Path(real_processed_dir)
    return real.parent / (real.name + _SYNTHETIC_SUFFIX)


def ensure_dataset_present(
    settings: Settings,
    *,
    n_train: int = _DEFAULT_TRAIN_SAMPLES,
    n_val: int = _DEFAULT_VAL_SAMPLES,
) -> tuple[Settings, bool]:
    """Materialise synthetic shards if real data is missing.

    Returns the (possibly rewritten) ``Settings`` and a bool marking whether
    synthetic stand-ins were used (``True`` when the loop should run on
    synthetic data, ``False`` when real data is on disk and untouched).
    """
    real_dir = Path(settings.task.processed_data_dir)
    if (real_dir / "train.pt").exists():
        return settings, False

    synthetic = synthetic_dir_for(real_dir)
    if not (synthetic / "train.pt").exists():
        _write_synthetic(synthetic, settings, n_train, n_val)

    new_task = settings.task.model_copy(
        update={"processed_data_dir": str(synthetic)}
    )
    return settings.model_copy(update={"task": new_task}), True


def write_synthetic_shards(
    settings: Settings,
    *,
    n_train: int = _DEFAULT_TRAIN_SAMPLES,
    n_val: int = _DEFAULT_VAL_SAMPLES,
    overwrite: bool = True,
) -> Path:
    """Write synthetic shards into the ``_synthetic`` directory and return it."""
    target = synthetic_dir_for(settings.task.processed_data_dir)
    if overwrite:
        (target / "train.pt").unlink(missing_ok=True)
        (target / "val.pt").unlink(missing_ok=True)
    _write_synthetic(target, settings, n_train, n_val)
    return target


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _write_synthetic(
    target_dir: Path, settings: Settings, n_train: int, n_val: int
) -> None:
    import torch  # heavy; only loaded on the synth path

    target_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(settings.task.input_tensor_shape)
    classes = settings.task.expected_num_classes

    torch.manual_seed(0)
    for split, n in (("train", n_train), ("val", n_val)):
        x = torch.randn(n, *shape)
        y = torch.zeros(n, classes)
        idx = torch.randint(0, classes, (n,))
        y[torch.arange(n), idx] = 1.0
        torch.save({"x": x, "y": y}, target_dir / f"{split}.pt")
