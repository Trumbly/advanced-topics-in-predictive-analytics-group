import json, os
from datetime import datetime

class ExperimentLog:
    def __init__(self, log_dir="experiments"):
        self.log_dir = log_dir
        self.experiments = []
        os.makedirs(log_dir, exist_ok=True)

    def create_experiment_dir(self, iteration: int) -> str:
        dir_name = f"{self.log_dir}/experiment_{iteration:03d}"
        os.makedirs(dir_name, exist_ok=True)
        return dir_name

    def save_experiment(self, data: dict, exp_dir: str):
        data["timestamp"] = datetime.now().isoformat()
        self.experiments.append(data)
        with open(f"{exp_dir}/results.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_summary(self) -> str:
        if not self.experiments:
            return "No experiments run yet. This is the first iteration."
        summary = "PAST EXPERIMENTS SUMMARY:\n"
        for exp in self.experiments:
            success = "SUCCESS" if exp["result"]["success"] else "FAILED"
            summary += f"""
Experiment {exp["iteration"]}: {success}
  Proposal: {str(exp.get("proposal",""))[:150]}
  ROC-AUC: {exp.get("score", 0):.4f}
  Analysis: {str(exp.get("analysis",""))[:200]}
"""
        best = sorted(self.experiments, key=lambda x: x.get("score", 0), reverse=True)[0]
        summary += f"\nBest so far: Experiment #{best['iteration']} with ROC-AUC {best.get('score',0):.4f}"
        return summary
