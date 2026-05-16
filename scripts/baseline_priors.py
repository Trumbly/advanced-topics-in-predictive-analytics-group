"""Predict per-class priors for every window. Reports ROC-AUC macro on val.

Reproduces the trivial baseline cited in section 3 of the report. Re-run
before the final draft if the val index changes.

Implementation note: the val / train indices on this repo live at
``data/processed/mels/{train,val}_index.json`` and already carry the
multi-hot label as a list of integer class indices under ``class_indices``.
We compute the per-class prior from the train split and score the val split
with that constant vector. Because every val window receives the *same*
score, per-class ROC-AUC is mathematically forced to 0.5 (no per-sample
variation -> all pairs tie); we still compute it explicitly so the number
in the report is reproducible from this script rather than asserted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

TRAIN_INDEX = Path("data/processed/mels/train_index.json")
VAL_INDEX = Path("data/processed/mels/val_index.json")


def _multi_hot(samples: list[dict], num_classes: int) -> np.ndarray:
    y = np.zeros((len(samples), num_classes), dtype=np.float32)
    for i, s in enumerate(samples):
        for c in s["class_indices"]:
            y[i, c] = 1.0
    return y


def main() -> None:
    train_idx = json.loads(TRAIN_INDEX.read_text())
    val_idx = json.loads(VAL_INDEX.read_text())
    num_classes = train_idx["num_classes"]
    assert val_idx["num_classes"] == num_classes

    y_train = _multi_hot(train_idx["samples"], num_classes)
    y_val = _multi_hot(val_idx["samples"], num_classes)

    priors = y_train.mean(axis=0)  # shape (num_classes,)
    y_score = np.broadcast_to(priors, y_val.shape)

    aucs: list[float] = []
    for j in range(num_classes):
        if y_val[:, j].sum() == 0 or y_val[:, j].sum() == y_val.shape[0]:
            continue
        aucs.append(roc_auc_score(y_val[:, j], y_score[:, j]))

    print(
        f"priors_baseline_macro_auc = {np.mean(aucs):.4f}  "
        f"(n_classes_scored={len(aucs)})"
    )


if __name__ == "__main__":
    main()
