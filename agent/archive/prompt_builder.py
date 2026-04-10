class PromptBuilder:

    def build_proposal_prompt(self, data_info, past_experiments, iteration):
        num_species = data_info.get("num_species", 206)
        return f"You are an ML researcher on BirdCLEF 2026. Task: multi-label bird classification from mel spectrograms. {num_species} species. Input shape (128,256). Metric: ROC-AUC. Train under 5 mins on M4 Pro.\n\n{past_experiments}\n\nIteration {iteration}: Propose ONE experiment. If first: simple 3-layer CNN. If overfitting: add dropout. If underfitting: more layers. Never repeat failures. Describe in 3-5 sentences WHAT and WHY."

    def build_code_prompt(self, proposal, data_info):
        n = data_info.get("num_species", 206)
        return f"""Proposal: {proposal}

Write a complete Python script that trains on REAL DATA FILES.

START WITH EXACTLY THESE LINES — copy them verbatim, zero changes:

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

X = np.load("data/spectrograms.npy")
y = np.load("data/labels.npy")
X = X[..., np.newaxis]
X = X[:2000]
y = y[:2000]
X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[1600:]
print(f"Shapes — X_train: {{X_train.shape}}, y_train: {{y_train.shape}}")

HARD RULES — any violation crashes the agent:
1. NO try/except blocks anywhere — not even for roc_auc_score
2. NO np.random to create X or y data — not anywhere in the script
3. NO dummy data, mock data, or placeholder data generation
4. The model input_shape MUST be hardcoded as (128, 256, 1)
5. The final Dense layer MUST be Dense({n}, activation='sigmoid')
6. model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32)
7. Define ALL functions BEFORE calling them

END WITH EXACTLY THESE LINES — copy them verbatim:

valid_cols = np.where(y_val.sum(axis=0) > 0)[0]
y_pred = model.predict(X_val)
score = roc_auc_score(y_val[:, valid_cols], y_pred[:, valid_cols], average="macro")
print(f"ROC-AUC: {{score:.4f}}")

Return ONLY the Python code inside ```python blocks."""


    def build_analysis_prompt(self, proposal, code, result):
        ok = result["success"]
        info = f"ROC-AUC: {result.get('roc_auc',0):.4f}" if ok else f"Error: {result.get('error','')[:200]}"
        return f"Experiment {'succeeded' if ok else 'FAILED'}. {info}. Proposal: {proposal[:300]}. In 2 sentences: why did this happen and what ONE change improves it next time?"
