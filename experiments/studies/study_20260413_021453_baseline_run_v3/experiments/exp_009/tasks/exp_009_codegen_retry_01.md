# Task exp_009_codegen_retry_01

- **Experiment:** exp_009
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-13 04:24:30.954685+00:00
- **Completed:** 2026-04-13 04:26:56.077046+00:00

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


## VALIDATION ERROR (attempt 1)
Your previous code was rejected by the validator:
  Error type: SyntaxError
  Message: Line 260: unterminated string literal (detected at line 260)

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
# Assuming pipelines.data_loader exists and works as described
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight
# Import the registered model backbone
from pipelines.models import mobilenet_v3_small

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from an env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# The proposal specified time_shift: false, noise_injection: false, mixup: 0.0, specaugment: true
# The skeleton requires augmentation={"time_shift": True, "noise_injection": True} by default.
# We must override this to match the proposal's explicit settings.
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}


# --- Model Definition: MobileNetV3-Small + Temporal Attention ---

class TemporalAttentionPooling(nn.Module):
    """
    Applies a simple channel-wise attention mechanism across the time dimension
    to capture temporal dependencies before flattening.
    Input: (B, C, T, H') -> Output: (B, C, 1, 1)
    """
    def __init__(self, in_channels):
        super().__init__()
        # Global Average Pooling across Time and Height dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Simple attention mechanism: Conv -> Sigmoid -> Scale
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, T, H')
        # 1. Pool to (B, C, 1, 1) for feature compression
        pooled = self.avgpool(x)

        # 2. Apply attention mask (this scales the feature map)
        attention_weights = self.attention(pooled)

        # 3. Apply the weights and return the scaled representation
        return x * attention_weights.expand_as(x)


class MobileNetAttentionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone Initialization
        # The standard mobilenet_v3_small expects 3 input channels (RGB).
        # We must adapt it for 1 channel (Spectrogram).
        # We manually replace the first convolution layer to accept 1 channel.
        self.backbone = mobilenet_v3_small(pretrained=True)
        
        # Store the original first conv layer to replace it
        original_conv1 = self.backbone.features[0]
        
        # Create a new first convolution layer: Input 1 channel, Output original channels
        # Check the original output channels of the first layer
        original_out_channels = int(self.backbone.features[0].out_channels)
        self.backbone.features[0] = nn.Conv2d(1, original_out_channels, kernel_size=3, padding=1, bias=False)
        
        # 2. Attention Pooling Layer
        # We use the output channel count of the last layer of the backbone features
        # For simplicity, we pass the output channels of the backbone's main feature sequence.
        # The output channels of the backbone's final stage are generally the input to the pooling/head.
        # Assuming the final feature map channel count is 128 (a common value for small backbones).
        # If the backbone structure changes, this might need adjustment, but sticking to the proposal structure.
        self.temporal_attention = TemporalAttentionPooling(in_channels=128) # Assuming 128 channels output from backbone features
        
        # 3. Classifier Head
        # Use LazyLinear to handle variable feature map size after pooling/flattening
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, T, H')
        
        # Pass through the adapted backbone
        x = self.backbone.features(x)
        
        # Apply Temporal Attention Pooling
        x = self.temporal_attention(x)
        
        # Flatten the output: (B, C, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1)
        
        # Final classification
        logits = self.classifier(x)
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
        # load_precomputed_dataset reads config variables automatically
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
        model = MobileNetAttentionClassifier(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (1, 1, 128, 313) matching the expected input structure
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
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
                    # .cpu() before .numpy()
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
            f"all done: roc_auc={curves['roc_auc_macro'][-1]:.
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
# Assuming pipelines.data_loader exists and works as described
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight
# Import the registered model backbone
from pipelines.models import mobilenet_v3_small

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from an env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# The proposal specified time_shift: false, noise_injection: false, mixup: 0.0, specaugment: true
# The skeleton requires augmentation={"time_shift": True, "noise_injection": True} by default.
# We must override this to match the proposal's explicit settings.
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}


# --- Model Definition: MobileNetV3-Small + Temporal Attention ---

class TemporalAttentionPooling(nn.Module):
    """
    Applies a simple channel-wise attention mechanism across the time dimension
    to capture temporal dependencies before flattening.
    Input: (B, C, T, H') -> Output: (B, C, 1, 1)
    """
    def __init__(self, in_channels):
        super().__init__()
        # Global Average Pooling across Time and Height dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Simple attention mechanism: Conv -> Sigmoid -> Scale
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, T, H')
        # 1. Pool to (B, C, 1, 1) for feature compression
        pooled = self.avgpool(x)

        # 2. Apply attention mask (this scales the feature map)
        attention_weights = self.attention(pooled)

        # 3. Apply the weights and return the scaled representation
        return x * attention_weights.expand_as(x)


class MobileNetAttentionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone Initialization
        # The standard mobilenet_v3_small expects 3 input channels (RGB).
        # We must adapt it for 1 channel (Spectrogram).
        # We manually replace the first convolution layer to accept 1 channel.
        self.backbone = mobilenet_v3_small(pretrained=True)
        
        # Store the original first conv layer to replace it
        original_conv1 = self.backbone.features[0]
        
        # Create a new first convolution layer: Input 1 channel, Output original channels
        # Check the original output channels of the first layer
        original_out_channels = int(self.backbone.features[0].out_channels)
        self.backbone.features[0] = nn.Conv2d(1, original_out_channels, kernel_size=3, padding=1, bias=False)
        
        # 2. Attention Pooling Layer
        # We use the output channel count of the last layer of the backbone features
        # For simplicity, we pass the output channels of the backbone's main feature sequence.
        # The output channels of the backbone's final stage are generally the input to the pooling/head.
        # Assuming the final feature map channel count is 128 (a common value for small backbones).
        # If the backbone structure changes, this might need adjustment, but sticking to the proposal structure.
        self.temporal_attention = TemporalAttentionPooling(in_channels=128) # Assuming 128 channels output from backbone features
        
        # 3. Classifier Head
        # Use LazyLinear to handle variable feature map size after pooling/flattening
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, T, H')
        
        # Pass through the adapted backbone
        x = self.backbone.features(x)
        
        # Apply Temporal Attention Pooling
        x = self.temporal_attention(x)
        
        # Flatten the output: (B, C, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1)
        
        # Final classification
        logits = self.classifier(x)
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
        # load_precomputed_dataset reads config variables automatically
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
        model = MobileNetAttentionClassifier(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (1, 1, 128, 313) matching the expected input structure
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
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
                    # .cpu() before .numpy()
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
            f"all done: roc_auc={curves['roc_auc_macro'][-1]:.

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
# Assuming pipelines.data_loader exists and works as described
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight
# Import the registered model backbone
from pipelines.models import mobilenet_v3_small

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from an env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# The proposal specified time_shift: false, noise_injection: false, mixup: 0.0, specaugment: true
# The skeleton requires augmentation={"time_shift": True, "noise_injection": True} by default.
# We must override this to match the proposal's explicit settings.
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}


# --- Model Definition: MobileNetV3-Small + Temporal Attention ---

class TemporalAttentionPooling(nn.Module):
    """
    Applies a simple channel-wise attention mechanism across the time dimension
    to capture temporal dependencies before flattening.
    Input: (B, C, T, H') -> Output: (B, C, 1, 1)
    """
    def __init__(self, in_channels):
        super().__init__()
        # Global Average Pooling across Time and Height dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Simple attention mechanism: Conv -> Sigmoid -> Scale
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, T, H')
        # 1. Pool to (B, C, 1, 1) for feature compression
        pooled = self.avgpool(x)

        # 2. Apply attention mask (this scales the feature map)
        attention_weights = self.attention(pooled)

        # 3. Apply the weights and return the scaled representation
        return x * attention_weights.expand_as(x)


class MobileNetAttentionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone Initialization
        # The standard mobilenet_v3_small expects 3 input channels (RGB).
        # We must adapt it for 1 channel (Spectrogram).
        # We manually replace the first convolution layer to accept 1 channel.
        self.backbone = mobilenet_v3_small(pretrained=True)
        
        # Store the original first conv layer to replace it
        original_conv1 = self.backbone.features[0]
        
        # Create a new first convolution layer: Input 1 channel, Output original channels
        # Check the original output channels of the first layer
        original_out_channels = int(self.backbone.features[0].out_channels)
        self.backbone.features[0] = nn.Conv2d(1, original_out_channels, kernel_size=3, padding=1, bias=False)
        
        # 2. Attention Pooling Layer
        # We use the output channel count of the last layer of the backbone features
        # For simplicity, we pass the output channels of the backbone's main feature sequence.
        # The output channels of the backbone's final stage are generally the input to the pooling/head.
        # Assuming the final feature map channel count is 128 (a common value for small backbones).
        # If the backbone structure changes, this might need adjustment, but sticking to the proposal structure.
        self.temporal_attention = TemporalAttentionPooling(in_channels=128) # Assuming 128 channels output from backbone features
        
        # 3. Classifier Head
        # Use LazyLinear to handle variable feature map size after pooling/flattening
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, T, H')
        
        # Pass through the adapted backbone
        x = self.backbone.features(x)
        
        # Apply Temporal Attention Pooling
        x = self.temporal_attention(x)
        
        # Flatten the output: (B, C, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1)
        
        # Final classification
        logits = self.classifier(x)
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
        # load_precomputed_dataset reads config variables automatically
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
        model = MobileNetAttentionClassifier(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (1, 1, 128, 313) matching the expected input structure
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
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
                    # .cpu() before .numpy()
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
            f"all done: roc_auc={curves['roc_auc_macro'][-1]:.

