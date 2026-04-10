import json
import os
import yaml

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

def load_system_prompt() -> str:
    path = "prompts/system_prompt.txt"
    with open(path) as f:
        return f.read()

def build_user_prompt(history: list[dict], data_summary: dict) -> str:
    history_str = ""
    if history:
        last = history[-5:]  # last 5 experiments
        for h in last:
            history_str += (
                f"\nExperiment {h['experiment_id']}: "
                f"val_acc={h.get('val_accuracy', 'N/A'):.4f}, "
                f"approach={h.get('approach', 'N/A')}"
            )
    else:
        history_str = "No experiments run yet."

    best = max(history, key=lambda x: x.get("val_accuracy", 0), default=None)
    best_str = (
        f"Best so far: Experiment {best['experiment_id']} "
        f"val_acc={best.get('val_accuracy', 0):.4f} ({best.get('approach', 'N/A')})"
        if best else "No best result yet."
    )

    return f"""
DATA SUMMARY:
{json.dumps(data_summary, indent=2)}

EXPERIMENT HISTORY (last 5):
{history_str}

{best_str}

Generate the next experiment. Return ONLY executable Python code.
- Load data from: {cfg['data']['spectrograms']} and {cfg['data']['labels']}
- Save results to: experiments/logs/EXPERIMENT_ID/results.json
- results.json must contain: val_accuracy, approach, notes
"""
