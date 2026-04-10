content = open("agent/prompt_builder.py", "w")
content.write("""class PromptBuilder:

    def build_proposal_prompt(self, data_info, past_experiments, iteration):
        num_species = data_info.get("num_species", 206)
        return f"You are an ML researcher on BirdCLEF 2026. Task: multi-label bird classification from mel spectrograms. {num_species} species. Input shape (128,256). Metric: ROC-AUC. Train under 5 mins on M4 Pro.\\n\\n{past_experiments}\\n\\nIteration {iteration}: Propose ONE experiment. If first: simple 3-layer CNN. If overfitting: add dropout. If underfitting: more layers. Never repeat failures. Describe in 3-5 sentences WHAT and WHY."

    def build_code_prompt(self, proposal, data_info):
        n = data_info.get("num_species", 206)
        return f\"\"\"{proposal}

Write a complete Python script that does EXACTLY this:

STEP 1 - Load data (copy this exactly, do not change):
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
X = np.load("data/spectrograms.npy")
y = np.load("data/labels.npy")
X = X[..., np.newaxis]
X = X[:2000]; y = y[:2000]
split = 1600
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

STEP 2 - Build the CNN you proposed (input shape (128,256,1), output Dense({n}, activation=sigmoid), loss=binary_crossentropy, 3 epochs max, batch_size=32)

STEP 3 - Evaluate (copy this exactly):
y_pred = model.predict(X_val)
score = roc_auc_score(y_val, y_pred, average="macro")
print(f"ROC-AUC: {{score:.4f}}")

Rules: NO dummy data. NO np.random for labels. Return only Python code in triple backtick python blocks.\"\"\"

    def build_analysis_prompt(self, proposal, code, result):
        ok = result["success"]
        info = f"ROC-AUC: {result.get('roc_auc',0):.4f}" if ok else f"Error: {result.get('error','')[:200]}"
        return f"Experiment {'succeeded' if ok else 'FAILED'}. {info}. Proposal: {proposal[:300]}. In 2 sentences: why did this happen and what ONE change improves it next time?"
""")
content.close()
print("Fixed!")
