import subprocess, re, os

class CodeRunner:
    def __init__(self, timeout_seconds=300):
        self.timeout = timeout_seconds

    def run(self, script_path: str) -> dict:
        if not os.path.exists("data/spectrograms.npy"):
            return {"success": False, "error": "DATA FILES MISSING: data/spectrograms.npy not found. Run data pipeline first.", "roc_auc": 0.0}
        if not os.path.exists("data/labels.npy"):
            return {"success": False, "error": "DATA FILES MISSING: data/labels.npy not found. Run data pipeline first.", "roc_auc": 0.0}
        try:
            proc = subprocess.run(
                ["python", script_path],
                capture_output=True, text=True,
                timeout=self.timeout,
                cwd=os.path.abspath(os.getcwd())
            )
            output = proc.stdout + proc.stderr
            roc_auc = self._parse_metric(output, "ROC-AUC")
            if proc.returncode == 0:
                return {"success": True, "output": output, "roc_auc": roc_auc}
            else:
                return {"success": False, "error": output[:1000], "roc_auc": 0.0}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout: training took too long", "roc_auc": 0.0}
        except Exception as e:
            return {"success": False, "error": str(e), "roc_auc": 0.0}

    def _parse_metric(self, output: str, metric_name: str) -> float:
        pattern = rf"{metric_name}[:\s]+([0-9]+\.?[0-9]*)"
        match = re.search(pattern, output, re.IGNORECASE)
        return float(match.group(1)) if match else 0.0
