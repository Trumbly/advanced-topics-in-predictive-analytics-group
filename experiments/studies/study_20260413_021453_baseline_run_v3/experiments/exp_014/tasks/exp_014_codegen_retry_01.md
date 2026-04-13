# Task exp_014_codegen_retry_01

- **Experiment:** exp_014
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-13 05:39:48.576009+00:00
- **Completed:** 2026-04-13 05:42:13.394433+00:00

## Prompt Used
```
[SYSTEM]
You are a Python ML engineer. Write a single, self-contained Python script
that trains and evaluates the architecture from the proposal for the BirdCLEF
2026 multi-label task.

## Input tensor shape —...

[USER]
chitecture above. Use the skeleton in the system prompt as your structure.
Fill in the model definition and any proposal-specific training changes.

Return ONLY the Python code. No explanations, no markdown fences, no prose.


## VALIDATION ERROR (attempt 1)
Your previous code was rejected by the validator:
  Error type: SyntaxError
  Message: Line 198: invalid syntax

Fix this specific issue and return the COMPLETE corrected script.
Return ONLY Python code — no explanations, no markdown fences.
```

## LLM Response
```
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight
from pipelines.models import CnnSmallV1

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from environment variable BIRDCLEF_EPOCHS
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===
class CnnGruHybrid(nn.Module):
    """
    CNN_Small_V1 backbone output features sequence fed into 2-layer GRU(128) head.
    Input: (B, 1, 128, 313)
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: cnn_small_v1
        self.backbone = CnnSmallV1()

        # 2. Global Pooling: Collapse spatial dims (H, W) -> (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Temporal reshaping: (B, C, 1, 1) -> (B, 1, C) for GRU input
        # The feature dimension C will be the input size for GRU.
        # We need to determine the output channels of the backbone first.
        # For cnn_small_v1, the final feature channel count is 128.
        # The input size to GRU will be 128.
        self.gru_input_dim = 128 # Based on cnn_small_v1's final feature map depth
        
        # 4. GRU Head: 2 layers, hidden size 128
        self.gru = nn.GRU(
            input_size=self.gru_input_dim, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=False # Standard GRU usage
        )
        
        # 5. Classifier Head: Final linear projection
        # Using LazyLinear for robustness, although here the size is fixed (128 -> num_classes)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input x shape: (B, 1, 128, 313)
        
        # 1. Backbone pass
        x = self.backbone(x) # Output shape: (B, 128, H', W')
        
        # 2. Global Pooling
        x = self.pool(x) # Output shape: (B, 128, 1, 1)
        
        # 3. Reshaping for GRU: (B, C, 1, 1) -> (B, 1, C)
        # We squeeze the spatial dims and then add a sequence dimension (1)
        x = x.squeeze(-1).squeeze(-1) # Shape: (B, 128)
        x = x.unsqueeze(1) # Shape: (B, 1, 128) -> (Batch, SeqLen=1, FeatureDim=128)
        
        # 4. GRU pass (batch_first=False: (SeqLen, B, HiddenSize))
        # Since SeqLen=1, output is (1, B, 128)
        output, _ = self.gru(x)
        
        # 5. Final classification layer
        # output shape: (1, B, 128)
        output = output.squeeze(0) # Shape: (B, 128)
        logits = self.head(output) # Shape: (B, num_classes)
        
        return logits

# ============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads batch_size/num_workers from env vars
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = CnnGruHybrid(num_classes=num_classes)
        model = model.to(device)
        
        # --- Initialization of Lazy/Registry Layers ---
        # Pass a dummy tensor through the model to build all necessary weights
        with torch.no_grad():
            # Dummy input shape: (B=1, C=1, H=128, W=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
            
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
        for epoch in range(EPOCHS):
            print(
                f"epoch {epoch + 1}/{EPOCHS} starting "
                f"({n_train_batches} batches)...",
                flush=True,
            )
            model.train()
            epoch_losses = []
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

                if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == n_train_batches:
                    pct = 100.0 * (batch_idx + 1) / n_train_batches
                    print(
                        f"  epoch {epoch + 1} [{pct:5.1f}%] "
                        f"batch {batch_idx + 1}/{n_train_batches} "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )

            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Calculate probabilities (sigmoid) and move results to CPU/Numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # Note: average_precision_score is used as a proxy for AP@k=5 based on common practices
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 at threshold 0.5 ---
            preds_binary = (probs >= 0.5).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_binary, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            curves["f1_macro"].append(val_f1)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc={val_auc:.4f} "
                f"cmap@5={val_cmap5:.4f} f1={val_f1:.4f}",
                flush=True,
            )

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "f1_macro": curves["f1_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f} "
            f"f1={curves['f1_macro'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)
```

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight
from pipelines.models import CnnSmallV1

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from environment variable BIRDCLEF_EPOCHS
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===
class CnnGruHybrid(nn.Module):
    """
    CNN_Small_V1 backbone output features sequence fed into 2-layer GRU(128) head.
    Input: (B, 1, 128, 313)
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: cnn_small_v1
        self.backbone = CnnSmallV1()

        # 2. Global Pooling: Collapse spatial dims (H, W) -> (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Temporal reshaping: (B, C, 1, 1) -> (B, 1, C) for GRU input
        # The feature dimension C will be the input size for GRU.
        # We need to determine the output channels of the backbone first.
        # For cnn_small_v1, the final feature channel count is 128.
        # The input size to GRU will be 128.
        self.gru_input_dim = 128 # Based on cnn_small_v1's final feature map depth
        
        # 4. GRU Head: 2 layers, hidden size 128
        self.gru = nn.GRU(
            input_size=self.gru_input_dim, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=False # Standard GRU usage
        )
        
        # 5. Classifier Head: Final linear projection
        # Using LazyLinear for robustness, although here the size is fixed (128 -> num_classes)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input x shape: (B, 1, 128, 313)
        
        # 1. Backbone pass
        x = self.backbone(x) # Output shape: (B, 128, H', W')
        
        # 2. Global Pooling
        x = self.pool(x) # Output shape: (B, 128, 1, 1)
        
        # 3. Reshaping for GRU: (B, C, 1, 1) -> (B, 1, C)
        # We squeeze the spatial dims and then add a sequence dimension (1)
        x = x.squeeze(-1).squeeze(-1) # Shape: (B, 128)
        x = x.unsqueeze(1) # Shape: (B, 1, 128) -> (Batch, SeqLen=1, FeatureDim=128)
        
        # 4. GRU pass (batch_first=False: (SeqLen, B, HiddenSize))
        # Since SeqLen=1, output is (1, B, 128)
        output, _ = self.gru(x)
        
        # 5. Final classification layer
        # output shape: (1, B, 128)
        output = output.squeeze(0) # Shape: (B, 128)
        logits = self.head(output) # Shape: (B, num_classes)
        
        return logits

# ============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads batch_size/num_workers from env vars
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = CnnGruHybrid(num_classes=num_classes)
        model = model.to(device)
        
        # --- Initialization of Lazy/Registry Layers ---
        # Pass a dummy tensor through the model to build all necessary weights
        with torch.no_grad():
            # Dummy input shape: (B=1, C=1, H=128, W=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
            
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
        for epoch in range(EPOCHS):
            print(
                f"epoch {epoch + 1}/{EPOCHS} starting "
                f"({n_train_batches} batches)...",
                flush=True,
            )
            model.train()
            epoch_losses = []
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

                if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == n_train_batches:
                    pct = 100.0 * (batch_idx + 1) / n_train_batches
                    print(
                        f"  epoch {epoch + 1} [{pct:5.1f}%] "
                        f"batch {batch_idx + 1}/{n_train_batches} "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )

            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Calculate probabilities (sigmoid) and move results to CPU/Numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # Note: average_precision_score is used as a proxy for AP@k=5 based on common practices
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 at threshold 0.5 ---
            preds_binary = (probs >= 0.5).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_binary, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            curves["f1_macro"].append(val_f1)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc={val_auc:.4f} "
                f"cmap@5={val_cmap5:.4f} f1={val_f1:.4f}",
                flush=True,
            )

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "f1_macro": curves["f1_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f} "
            f"f1={curves['f1_macro'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

```

## Output
- **code:** import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight
from pipelines.models import CnnSmallV1

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from environment variable BIRDCLEF_EPOCHS
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===
class CnnGruHybrid(nn.Module):
    """
    CNN_Small_V1 backbone output features sequence fed into 2-layer GRU(128) head.
    Input: (B, 1, 128, 313)
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: cnn_small_v1
        self.backbone = CnnSmallV1()

        # 2. Global Pooling: Collapse spatial dims (H, W) -> (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Temporal reshaping: (B, C, 1, 1) -> (B, 1, C) for GRU input
        # The feature dimension C will be the input size for GRU.
        # We need to determine the output channels of the backbone first.
        # For cnn_small_v1, the final feature channel count is 128.
        # The input size to GRU will be 128.
        self.gru_input_dim = 128 # Based on cnn_small_v1's final feature map depth
        
        # 4. GRU Head: 2 layers, hidden size 128
        self.gru = nn.GRU(
            input_size=self.gru_input_dim, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=False # Standard GRU usage
        )
        
        # 5. Classifier Head: Final linear projection
        # Using LazyLinear for robustness, although here the size is fixed (128 -> num_classes)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input x shape: (B, 1, 128, 313)
        
        # 1. Backbone pass
        x = self.backbone(x) # Output shape: (B, 128, H', W')
        
        # 2. Global Pooling
        x = self.pool(x) # Output shape: (B, 128, 1, 1)
        
        # 3. Reshaping for GRU: (B, C, 1, 1) -> (B, 1, C)
        # We squeeze the spatial dims and then add a sequence dimension (1)
        x = x.squeeze(-1).squeeze(-1) # Shape: (B, 128)
        x = x.unsqueeze(1) # Shape: (B, 1, 128) -> (Batch, SeqLen=1, FeatureDim=128)
        
        # 4. GRU pass (batch_first=False: (SeqLen, B, HiddenSize))
        # Since SeqLen=1, output is (1, B, 128)
        output, _ = self.gru(x)
        
        # 5. Final classification layer
        # output shape: (1, B, 128)
        output = output.squeeze(0) # Shape: (B, 128)
        logits = self.head(output) # Shape: (B, num_classes)
        
        return logits

# ============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads batch_size/num_workers from env vars
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = CnnGruHybrid(num_classes=num_classes)
        model = model.to(device)
        
        # --- Initialization of Lazy/Registry Layers ---
        # Pass a dummy tensor through the model to build all necessary weights
        with torch.no_grad():
            # Dummy input shape: (B=1, C=1, H=128, W=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
            
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
        for epoch in range(EPOCHS):
            print(
                f"epoch {epoch + 1}/{EPOCHS} starting "
                f"({n_train_batches} batches)...",
                flush=True,
            )
            model.train()
            epoch_losses = []
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

                if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == n_train_batches:
                    pct = 100.0 * (batch_idx + 1) / n_train_batches
                    print(
                        f"  epoch {epoch + 1} [{pct:5.1f}%] "
                        f"batch {batch_idx + 1}/{n_train_batches} "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )

            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Calculate probabilities (sigmoid) and move results to CPU/Numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # Note: average_precision_score is used as a proxy for AP@k=5 based on common practices
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 at threshold 0.5 ---
            preds_binary = (probs >= 0.5).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_binary, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            curves["f1_macro"].append(val_f1)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc={val_auc:.4f} "
                f"cmap@5={val_cmap5:.4f} f1={val_f1:.4f}",
                flush=True,
            )

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "f1_macro": curves["f1_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f} "
            f"f1={curves['f1_macro'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

