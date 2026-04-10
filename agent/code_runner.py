import subprocess
import os
import json
from pathlib import Path

def run_experiment(code: str, experiment_id: str) -> dict:
    exp_dir = Path(f"experiments/logs/{experiment_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    script_path = exp_dir / "train.py"
    result_path = exp_dir / "results.json"

    with open(script_path, "w") as f:
        f.write(code)

    try:
        proc = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        stdout = proc.stdout
        stderr = proc.stderr

        with open(exp_dir / "stdout.log", "w") as f:
            f.write(stdout)
        with open(exp_dir / "stderr.log", "w") as f:
            f.write(stderr)

        if result_path.exists():
            with open(result_path) as f:
                results = json.load(f)
            results["experiment_id"] = experiment_id
            results["status"] = "success"
        else:
            results = {
                "experiment_id": experiment_id,
                "status": "failed",
                "error": stderr[-1000:],
                "val_accuracy": 0.0,
                "approach": "unknown",
            }
    except subprocess.TimeoutExpired:
        results = {
            "experiment_id": experiment_id,
            "status": "timeout",
            "val_accuracy": 0.0,
            "approach": "unknown",
        }

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
