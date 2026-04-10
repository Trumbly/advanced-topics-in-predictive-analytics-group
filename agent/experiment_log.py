import json
import os
from pathlib import Path
import yaml

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

MEMORY_FILE = Path(cfg["agent"]["memory_file"])
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_history() -> list[dict]:
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return []

def save_result(result: dict):
    history = load_history()
    history.append(result)
    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def get_next_experiment_id(history: list[dict]) -> str:
    n = len(history) + 1
    return f"experiment_{n:03d}"

def print_summary(history: list[dict]):
    if not history:
        print("No experiments run yet.")
        return
    best = max(history, key=lambda x: x.get("val_accuracy", 0))
    print(f"\n{'='*50}")
    print(f"Total experiments: {len(history)}")
    print(f"Best: {best['experiment_id']} | val_acc={best.get('val_accuracy',0):.4f} | {best.get('approach','N/A')}")
    print(f"{'='*50}\n")
