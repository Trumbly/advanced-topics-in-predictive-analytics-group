import os

files = {}

files["agent/llm_client.py"] = '''import ollama
import re

class LLMClient:
    def __init__(self, model_name="gemma4:e4b"):
        self.model_name = model_name
        self.conversation_history = []
        print(f"LLM Client ready: {model_name}")

    def ask(self, prompt: str, keep_history: bool = False) -> str:
        messages = [
            {"role": "system", "content": """You are an expert ML engineer 
specializing in audio classification and deep learning.
Your job is to design and generate Python code for training 
bird species classifiers using mel spectrograms.
Always return executable Python code when asked to generate code.
Be concise and practical."""},
            {"role": "user", "content": prompt}
        ]
        if keep_history and self.conversation_history:
            messages = [messages[0]] + self.conversation_history + [messages[1]]

        response = ollama.chat(model=self.model_name, messages=messages)
        reply = response["message"]["content"]

        if keep_history:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def extract_code(self, text: str) -> str:
        pattern = r\'\'\'```python\\n(.*?)```\'\'\'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return text

    def reset_history(self):
        self.conversation_history = []
'''

files["agent/experiment_log.py"] = '''import json, os
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
        summary = "PAST EXPERIMENTS SUMMARY:\\n"
        for exp in self.experiments:
            success = "SUCCESS" if exp["result"]["success"] else "FAILED"
            summary += f"""
Experiment {exp["iteration"]}: {success}
  Proposal: {str(exp.get("proposal",""))[:150]}
  ROC-AUC: {exp.get("score", 0):.4f}
  Analysis: {str(exp.get("analysis",""))[:200]}
"""
        best = sorted(self.experiments, key=lambda x: x.get("score", 0), reverse=True)[0]
        summary += f"\\nBest so far: Experiment #{best[\'iteration\']} with ROC-AUC {best.get(\'score\',0):.4f}"
        return summary
'''

files["agent/code_runner.py"] = '''import subprocess, re, os

class CodeRunner:
    def __init__(self, timeout_seconds=300):
        self.timeout = timeout_seconds

    def run(self, script_path: str) -> dict:
        try:
            proc = subprocess.run(
                ["python", script_path],
                capture_output=True, text=True,
                timeout=self.timeout,
                cwd=os.path.dirname(os.path.abspath(script_path))
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
        pattern = rf"{metric_name}[:\\s]+([0-9]+\\.?[0-9]*)"
        match = re.search(pattern, output, re.IGNORECASE)
        return float(match.group(1)) if match else 0.0
'''

files["agent/prompt_builder.py"] = '''class PromptBuilder:

    def build_proposal_prompt(self, data_info: dict, past_experiments: str, iteration: int) -> str:
        return f"""
You are an ML researcher working on BirdCLEF 2026.
Task: Multi-label bird species classification from mel spectrograms.
Classes: {data_info.get("num_species", 234)} species (multi-label).
Input: Mel spectrogram images of shape (128, 256), normalized to [0, 1].
Metric: Macro-averaged ROC-AUC (higher is better, max 1.0).
Constraint: Model must train in under 5 minutes on Apple M4 Pro CPU/GPU.

{past_experiments}

This is iteration {iteration}. Based on past experiments above,
propose ONE new experiment. Consider:
- If no experiments yet: start with a simple 3-layer CNN baseline
- If baseline worked: try adding more layers or dropout
- If overfitting: reduce model size or add regularization
- If underfitting: increase model capacity or epochs
- Never repeat a failed approach

Describe in 3-5 sentences: WHAT architecture to try and WHY.
"""

    def build_code_prompt(self, proposal: str, data_info: dict) -> str:
        return f"""
Based on this experiment proposal:
{proposal}

Generate a complete runnable Python script that:
1. Loads mel spectrograms from numpy files in \'data/spectrograms/\' (shape: 128x256)
2. Loads labels from \'data/labels.npy\' (shape: num_samples x {data_info.get("num_species",234)})
3. Implements the proposed architecture using TensorFlow/Keras
4. Trains for MAX 3 epochs, batch_size=32, 20% validation split
5. Uses sigmoid output (NOT softmax) — this is multi-label classification
6. Uses binary_crossentropy loss
7. Evaluates with sklearn roc_auc_score (macro average)
8. Prints result in EXACTLY this format: ROC-AUC: 0.XXXX
9. Saves model to \'best_model.h5\' only if ROC-AUC > 0.60

IMPORTANT: Keep model small — must train in under 5 minutes.
Return ONLY the Python code in ```python ... ``` blocks.
"""

    def build_analysis_prompt(self, proposal: str, code: str, result: dict) -> str:
        outcome = "succeeded" if result["success"] else "FAILED"
        info = f"ROC-AUC: {result.get(\'roc_auc\',0):.4f}" if result["success"] else f"Error: {result.get(\'error\',\'\')[:300]}"
        return f"""
Experiment {outcome}. {info}
Proposal was: {proposal[:400]}

In 2-3 sentences:
1. Why did this result happen?
2. What ONE specific change would improve it next time?
Be brief and actionable.
"""
'''

files["agent/main.py"] = '''import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import LLMClient
from code_runner import CodeRunner
from experiment_log import ExperimentLog
from prompt_builder import PromptBuilder

def run_agent(num_iterations=10, model="gemma4:e4b"):
    llm = LLMClient(model_name=model)
    runner = CodeRunner(timeout_seconds=300)
    log = ExperimentLog(log_dir="experiments")
    builder = PromptBuilder()

    print("=" * 60)
    print("BIRDCLEF AI AGENT STARTING")
    print(f"Model: {model} | Iterations: {num_iterations}")
    print("=" * 60)

    with open("data/data_summary.json") as f:
        data_info = json.load(f)

    best_score = 0.0

    for iteration in range(1, num_iterations + 1):
        print(f"\\n--- EXPERIMENT {iteration}/{num_iterations} ---")

        # Step 1: Ask LLM to propose experiment
        prompt = builder.build_proposal_prompt(
            data_info=data_info,
            past_experiments=log.get_summary(),
            iteration=iteration
        )
        print("Asking Gemma 4 to propose an experiment...")
        proposal = llm.ask(prompt)
        print(f"Proposal: {proposal[:200]}...")

        # Step 2: Generate code
        print("Generating Python training code...")
        code_prompt = builder.build_code_prompt(proposal, data_info)
        raw_response = llm.ask(code_prompt)
        generated_code = llm.extract_code(raw_response)

        # Step 3: Save to file
        exp_dir = log.create_experiment_dir(iteration)
        script_path = f"{exp_dir}/train.py"
        with open(script_path, "w") as f:
            f.write(generated_code)
        print(f"Code saved to {script_path}")

        # Step 4: Run the code
        print("Running generated code...")
        result = runner.run(script_path)

        # Step 5: Show results
        if result["success"]:
            score = result.get("roc_auc", 0.0)
            print(f"SUCCESS — ROC-AUC: {score:.4f}")
            if score > best_score:
                best_score = score
                print(f"NEW BEST SCORE!")
        else:
            score = 0.0
            print(f"FAILED — {result[\'error\'][:200]}")

        # Step 6: Ask LLM to analyze
        analysis = llm.ask(builder.build_analysis_prompt(proposal, generated_code, result))

        # Step 7: Log everything
        log.save_experiment({
            "iteration": iteration,
            "proposal": proposal,
            "code": generated_code,
            "result": result,
            "analysis": analysis,
            "score": score,
            "best_score": best_score
        }, exp_dir)

        print(f"Best overall: {best_score:.4f}")
        time.sleep(1)

    print(f"\\nAGENT DONE. Best ROC-AUC: {best_score:.4f}")

if __name__ == "__main__":
    run_agent(num_iterations=10, model="gemma4:e4b")
'''

files["explore_data.py"] = '''import pandas as pd
import os, json

DATASET_PATH = os.path.expanduser("~/birdclef-2026")

print("=" * 60)
print("BIRDCLEF 2026 - DATASET EXPLORATION")
print("=" * 60)

train_df = pd.read_csv(f"{DATASET_PATH}/train.csv")
print(f"\\ntrain.csv shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print(f"\\nFirst row:")
print(train_df.iloc[0])

species_list = sorted(train_df["primary_label"].unique().tolist())
print(f"\\nTotal unique species: {len(species_list)}")
print(f"First 10: {species_list[:10]}")

train_audio_path = f"{DATASET_PATH}/train_audio"
total_files = sum(len(files) for _, _, files in os.walk(train_audio_path))
print(f"\\nTotal audio files: {total_files}")

counts = train_df["primary_label"].value_counts()
print(f"\\nMost common: {counts.head(5).to_dict()}")
print(f"Least common: {counts.tail(5).to_dict()}")

first_species = species_list[0]
species_folder = f"{train_audio_path}/{first_species}"
first_file = os.listdir(species_folder)[0]
example_path = f"{species_folder}/{first_file}"
print(f"\\nExample audio file: {example_path}")

os.makedirs("data", exist_ok=True)
summary = {
    "num_train": len(train_df),
    "num_species": len(species_list),
    "species_list": species_list,
    "total_audio_files": total_files,
    "dataset_path": DATASET_PATH,
    "train_audio_path": train_audio_path,
    "example_file": example_path,
    "columns": list(train_df.columns)
}
with open("data/data_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\\ndata/data_summary.json saved!")
print("=" * 60)
'''

files["test_spectrogram.py"] = '''import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import json, os

with open("data/data_summary.json") as f:
    summary = json.load(f)

audio_path = summary["example_file"]
print(f"Loading: {audio_path}")

audio, sr = librosa.load(audio_path, sr=32000, duration=5)
print(f"Audio loaded: {len(audio)} samples at {sr}Hz")

mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmin=20, fmax=16000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", fmin=20, fmax=16000)
plt.colorbar(format="%+2.0f dB")
species = os.path.basename(os.path.dirname(audio_path))
plt.title(f"Mel Spectrogram — Species: {species}")
plt.tight_layout()
plt.savefig("data/test_spectrogram.png", dpi=150)
print("Spectrogram saved to data/test_spectrogram.png")
print("Run: open data/test_spectrogram.png")
'''

# Create all files
for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll files created successfully!")
print("\nNext steps:")
print("1. python explore_data.py")
print("2. python test_spectrogram.py")
print("3. open data/test_spectrogram.png")
