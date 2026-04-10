import json
import sys
import yaml
from agent.llm_client import call_llm
from agent.prompt_builder import load_system_prompt, build_user_prompt
from agent.code_runner import run_experiment
from agent.experiment_log import load_history, save_result, get_next_experiment_id, print_summary
from pathlib import Path

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

def load_data_summary() -> dict:
    path = cfg["data"]["summary"]
    with open(path) as f:
        return json.load(f)

def run_agent(max_experiments: int = None):
    max_exp = max_experiments or cfg["agent"]["max_experiments"]
    system_prompt = load_system_prompt()
    data_summary = load_data_summary()
    history = load_history()

    print(f"Starting BirdCLEF agent | max_experiments={max_exp}")
    print_summary(history)

    for i in range(max_exp):
        experiment_id = get_next_experiment_id(history)
        print(f"\n[{i+1}/{max_exp}] Generating {experiment_id}...")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(history, data_summary)},
        ]

        code = call_llm(messages)

        # Save generated code to sandbox for inspection
        sandbox_path = Path(f"sandbox/runs/{experiment_id}.py")
        sandbox_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sandbox_path, "w") as f:
            f.write(code)

        print(f"  Running {experiment_id}...")
        result = run_experiment(code, experiment_id)
        save_result(result)
        history.append(result)

        status = result.get("status", "unknown")
        acc = result.get("val_accuracy", 0)
        approach = result.get("approach", "N/A")
        print(f"  Status: {status} | val_acc: {acc:.4f} | {approach}")
        print_summary(history)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_agent(n)
