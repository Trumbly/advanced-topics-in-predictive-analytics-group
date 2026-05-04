"""Fixture: successful run that writes a valid results.json."""
import json
from pathlib import Path

results = {
    "primary_score": 0.42,
    "primary_metric": "roc_auc_macro",
    "metrics": {"roc_auc_macro": 0.42, "f1_macro": 0.31},
    "history": [
        {"epoch": 1, "loss": 0.7, "roc_auc_macro": 0.30},
        {"epoch": 2, "loss": 0.5, "roc_auc_macro": 0.42},
    ],
    "stopped_early": False,
    "duration_seconds": 1.2,
}
Path("results.json").write_text(json.dumps(results))
print("done")
