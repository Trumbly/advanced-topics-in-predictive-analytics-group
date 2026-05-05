"""Dataset bootstrap helpers.

The training skeleton loads ``processed_dir/train.pt`` and ``val.pt``.
When those are missing (e.g. demo run on a fresh clone with no Kaggle data
downloaded), the agent loop would otherwise fail at the executor stage with
``FileNotFoundError: missing processed shard``. ``ensure_dataset_present``
detects this and writes small synthetic stand-ins so the loop has something
to chew on; real data is left untouched when present.
"""

from __future__ import annotations

from pathlib import Path

from lab.config import Settings

_DEFAULT_TRAIN_SAMPLES = 200
_DEFAULT_VAL_SAMPLES = 50


def ensure_dataset_present(
    settings: Settings,
    *,
    n_train: int = _DEFAULT_TRAIN_SAMPLES,
    n_val: int = _DEFAULT_VAL_SAMPLES,
) -> bool:
    """Generate synthetic shards if the real ``train.pt`` is missing.

    Returns ``True`` when synthetic shards were written (real data absent),
    ``False`` when the real shard already exists and was left alone.
    """
    processed = Path(settings.task.processed_data_dir)
    train_path = processed / "train.pt"
    if train_path.exists():
        return False

    import torch  # local import — torch is heavy and only needed on the synth path

    processed.mkdir(parents=True, exist_ok=True)
    shape = tuple(settings.task.input_tensor_shape)
    classes = settings.task.expected_num_classes

    torch.manual_seed(0)
    for split, n in (("train", n_train), ("val", n_val)):
        x = torch.randn(n, *shape)
        y = torch.zeros(n, classes)
        idx = torch.randint(0, classes, (n,))
        y[torch.arange(n), idx] = 1.0
        torch.save({"x": x, "y": y}, processed / f"{split}.pt")

    return True
