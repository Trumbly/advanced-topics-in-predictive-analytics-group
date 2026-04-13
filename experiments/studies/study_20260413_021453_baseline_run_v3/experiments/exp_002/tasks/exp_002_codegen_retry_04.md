# Task exp_002_codegen_retry_04

- **Experiment:** exp_002
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-13 02:29:41.288515+00:00
- **Completed:** 2026-04-13 02:32:11.264007+00:00

## Prompt Used
```
[SYSTEM]
You are a Python ML engineer. Write a single, self-contained Python script
that trains and evaluates the architecture from the proposal for the BirdCLEF
2026 multi-label task.

## Input tensor shape —...

[USER]
n the system prompt as your structure.
Fill in the model definition and any proposal-specific training changes.

Return ONLY the Python code. No explanations, no markdown fences, no prose.


## VALIDATION ERROR (attempt 4)
Your previous code was rejected by the validator:
  Error type: SyntaxError
  Message: Line 189: unterminated string literal (detected at line 189)

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

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Use the augmentation settings from the proposal, while adhering to the
# skeleton's structure for the data loader call.
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class SelfAttention(nn.Module):
    """
    A simplified self-attention mechanism applied across the feature dimension
    after global pooling, acting as a feature re-weighting layer.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        # Learnable weights to compute attention scores (Q, K, V projections)
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (B, C) where C = 128 (the feature dimension)
        B = x.size(0)
        
        # 1. Calculate Query, Key, Value representations
        Q = self.query(x) # (B, C)
        K = self.key(x)   # (B, C)
        V = self.value(x) # (B, C)
        
        # 2. Compute attention scores: Q * K^T (dot product)
        # Transpose K to get (B, C, C) -> (B, C) * (B, C)
        scores = torch.matmul(Q, K.unsqueeze(-1)) # (B, C, C)
        
        # 3. Normalize scores using softmax across the last dimension (feature dimension)
        attention_weights = self.softmax(scores) # (B, C, C)
        
        # 4. Apply weights to Value: Attention_Weights * V
        context_vector = torch.matmul(attention_weights, V) # (B, C, C)
        
        # 5. Aggregate (average) the context vector back to (B, C)
        output = context_vector.mean(dim=2)
        return output


class CnnAttentionBlock(nn.Module):
    """
    3-conv CNN front-end followed by Self-Attention pooling layer (as per proposal).
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 3-Conv CNN Front-end ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.ReLU(inplace=True)
        
        # --- Self-Attention Pooling ---
        # The feature dimension C is 128 (output of conv3)
        self.attention = SelfAttention(feature_dim=128)
        
        # --- Classification Head ---
        # Use AdaptiveAvgPool2d to ensure fixed size before the lazy layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LazyLinear is used to handle the dynamically sized feature dimension
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Stack
        x = self.conv1(x)
        x = self.pool1(x) # (B, 32, 128, 313)
        
        x = self.conv2(x)
        x = self.pool2(x) # (B, 64, 128, 313)
        
        x = self.conv3(x)
        x = self.pool3(x) # (B, 128, 128, 313)
        
        # 2. Global Context Pooling (B, 128, 1, 1)
        x = self.global_pool(x)
        
        # 3. Flatten for Attention (B, 128)
        x = torch.flatten(x, start_dim=1)
        
        # 4. Self-Attention Layer (B, 128)
        attention_output = self.attention(x)
        
        # 5. Classification Head (B, num_classes)
        logits = self.head(attention_output)
        return logits

# =============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # The data loader reads augmentation settings from the environment/config
        # and uses the fixed DataLoader structure.
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
        model = CnnAttentionBlock(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (B=1, C=1, M=128, T=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        # Log every 10% of batches, or at least every batch if less than 10 batches total
        log_every = max(1, n_train_batches // 10)

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
                # Move each batch to the training device. Float32 only.
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
                    # Sigmoid for probabilities, move to CPU for numpy conversion
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.
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

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Use the augmentation settings from the proposal, while adhering to the
# skeleton's structure for the data loader call.
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class SelfAttention(nn.Module):
    """
    A simplified self-attention mechanism applied across the feature dimension
    after global pooling, acting as a feature re-weighting layer.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        # Learnable weights to compute attention scores (Q, K, V projections)
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (B, C) where C = 128 (the feature dimension)
        B = x.size(0)
        
        # 1. Calculate Query, Key, Value representations
        Q = self.query(x) # (B, C)
        K = self.key(x)   # (B, C)
        V = self.value(x) # (B, C)
        
        # 2. Compute attention scores: Q * K^T (dot product)
        # Transpose K to get (B, C, C) -> (B, C) * (B, C)
        scores = torch.matmul(Q, K.unsqueeze(-1)) # (B, C, C)
        
        # 3. Normalize scores using softmax across the last dimension (feature dimension)
        attention_weights = self.softmax(scores) # (B, C, C)
        
        # 4. Apply weights to Value: Attention_Weights * V
        context_vector = torch.matmul(attention_weights, V) # (B, C, C)
        
        # 5. Aggregate (average) the context vector back to (B, C)
        output = context_vector.mean(dim=2)
        return output


class CnnAttentionBlock(nn.Module):
    """
    3-conv CNN front-end followed by Self-Attention pooling layer (as per proposal).
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 3-Conv CNN Front-end ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.ReLU(inplace=True)
        
        # --- Self-Attention Pooling ---
        # The feature dimension C is 128 (output of conv3)
        self.attention = SelfAttention(feature_dim=128)
        
        # --- Classification Head ---
        # Use AdaptiveAvgPool2d to ensure fixed size before the lazy layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LazyLinear is used to handle the dynamically sized feature dimension
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Stack
        x = self.conv1(x)
        x = self.pool1(x) # (B, 32, 128, 313)
        
        x = self.conv2(x)
        x = self.pool2(x) # (B, 64, 128, 313)
        
        x = self.conv3(x)
        x = self.pool3(x) # (B, 128, 128, 313)
        
        # 2. Global Context Pooling (B, 128, 1, 1)
        x = self.global_pool(x)
        
        # 3. Flatten for Attention (B, 128)
        x = torch.flatten(x, start_dim=1)
        
        # 4. Self-Attention Layer (B, 128)
        attention_output = self.attention(x)
        
        # 5. Classification Head (B, num_classes)
        logits = self.head(attention_output)
        return logits

# =============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # The data loader reads augmentation settings from the environment/config
        # and uses the fixed DataLoader structure.
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
        model = CnnAttentionBlock(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (B=1, C=1, M=128, T=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        # Log every 10% of batches, or at least every batch if less than 10 batches total
        log_every = max(1, n_train_batches // 10)

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
                # Move each batch to the training device. Float32 only.
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
                    # Sigmoid for probabilities, move to CPU for numpy conversion
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.

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

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Use the augmentation settings from the proposal, while adhering to the
# skeleton's structure for the data loader call.
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class SelfAttention(nn.Module):
    """
    A simplified self-attention mechanism applied across the feature dimension
    after global pooling, acting as a feature re-weighting layer.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        # Learnable weights to compute attention scores (Q, K, V projections)
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (B, C) where C = 128 (the feature dimension)
        B = x.size(0)
        
        # 1. Calculate Query, Key, Value representations
        Q = self.query(x) # (B, C)
        K = self.key(x)   # (B, C)
        V = self.value(x) # (B, C)
        
        # 2. Compute attention scores: Q * K^T (dot product)
        # Transpose K to get (B, C, C) -> (B, C) * (B, C)
        scores = torch.matmul(Q, K.unsqueeze(-1)) # (B, C, C)
        
        # 3. Normalize scores using softmax across the last dimension (feature dimension)
        attention_weights = self.softmax(scores) # (B, C, C)
        
        # 4. Apply weights to Value: Attention_Weights * V
        context_vector = torch.matmul(attention_weights, V) # (B, C, C)
        
        # 5. Aggregate (average) the context vector back to (B, C)
        output = context_vector.mean(dim=2)
        return output


class CnnAttentionBlock(nn.Module):
    """
    3-conv CNN front-end followed by Self-Attention pooling layer (as per proposal).
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 3-Conv CNN Front-end ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.ReLU(inplace=True)
        
        # --- Self-Attention Pooling ---
        # The feature dimension C is 128 (output of conv3)
        self.attention = SelfAttention(feature_dim=128)
        
        # --- Classification Head ---
        # Use AdaptiveAvgPool2d to ensure fixed size before the lazy layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LazyLinear is used to handle the dynamically sized feature dimension
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Stack
        x = self.conv1(x)
        x = self.pool1(x) # (B, 32, 128, 313)
        
        x = self.conv2(x)
        x = self.pool2(x) # (B, 64, 128, 313)
        
        x = self.conv3(x)
        x = self.pool3(x) # (B, 128, 128, 313)
        
        # 2. Global Context Pooling (B, 128, 1, 1)
        x = self.global_pool(x)
        
        # 3. Flatten for Attention (B, 128)
        x = torch.flatten(x, start_dim=1)
        
        # 4. Self-Attention Layer (B, 128)
        attention_output = self.attention(x)
        
        # 5. Classification Head (B, num_classes)
        logits = self.head(attention_output)
        return logits

# =============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # The data loader reads augmentation settings from the environment/config
        # and uses the fixed DataLoader structure.
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
        model = CnnAttentionBlock(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (B=1, C=1, M=128, T=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        # Log every 10% of batches, or at least every batch if less than 10 batches total
        log_every = max(1, n_train_batches // 10)

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
                # Move each batch to the training device. Float32 only.
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
                    # Sigmoid for probabilities, move to CPU for numpy conversion
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.

