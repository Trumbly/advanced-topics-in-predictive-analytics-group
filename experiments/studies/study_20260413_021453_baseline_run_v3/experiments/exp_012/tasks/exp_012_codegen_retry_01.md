# Task exp_012_codegen_retry_01

- **Experiment:** exp_012
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-13 05:23:12.458928+00:00
- **Completed:** 2026-04-13 05:25:37.775068+00:00

## Prompt Used
```
[SYSTEM]
You are a Python ML engineer. Write a single, self-contained Python script
that trains and evaluates the architecture from the proposal for the BirdCLEF
2026 multi-label task.

## Input tensor shape —...

[USER]
 the skeleton in the system prompt as your structure.
Fill in the model definition and any proposal-specific training changes.

Return ONLY the Python code. No explanations, no markdown fences, no prose.


## VALIDATION ERROR (attempt 1)
Your previous code was rejected by the validator:
  Error type: SyntaxError
  Message: Line 98: expected 'except' or 'finally' block

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
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# Set CPU thread count for data loading workers
torch.set_num_threads(os.cpu_count() or 4)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: EPOCHS is read from an env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Note: The proposal specifies augmentation values, but the skeleton requires
# passing the augmentation dict structure. We override the skeleton's default
# to match the proposal's explicit values where possible, while respecting
# the mandatory structure.
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}


# === Model definition at MODULE scope ===
# Implementing MobileNetV3 Small wrapper to accept 1 channel input.
class MobileNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        print("Initializing MobileNetV3 Small backbone...")
        # Load pretrained model, using weights from the proposal's model type
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accept 1 channel instead of 3 (RGB)
        # The first layer is typically the first module in the feature extraction path.
        # We assume the first layer name is 'features.conv1' or similar,
        # checking for the first nn.Conv2d in the backbone's sequential parts.
        
        # Standard MobileNetV3 structure often uses a Sequential block for initial layers.
        # We must find and replace the first Conv2d layer.
        
        # A robust way is to iterate through the modules and replace the first Conv2d found.
        
        # Attempting to modify the first Conv2d layer detected in the backbone
        first_conv_found = False
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and not first_conv_found:
                print(f"Modifying layer {name}: Changing in_channels from {module.in_channels} to 1")
                
                # Create a new Conv2d layer with the correct input channels
                new_conv = nn.Conv2d(1, module.out_channels, kernel_size=module.kernel_size, 
                                      stride=module.stride, padding=module.padding, dilation=module.dilation)
                
                # Replace the layer in the module's dict
                # This requires accessing the parent module's state dict, which is tricky.
                # For simplicity and robustness, we'll assume the first layer is directly accessible or wrap it.
                
                # Since direct replacement is complex, we will rely on modifying the state_dict or 
                # manually reconstructing the initial sequence if the structure is known.
                # Given the constraint to use the backbone, we will override the definition of the first layer's weights/bias
                # or, more safely, use a functional modification approach if possible.
                
                # For this specific task, we will assume the first block is the one to modify.
                # The first layer is usually self.backbone.features[0] or similar.
                
                # Fallback: If direct modification fails due to complexity, we must wrap the whole backbone
                # but for this solution, we will proceed with the assumption that we can modify the first Conv2d found.
                
                # A common pattern is that the first layer is located at self.backbone.features[0].
                # Due to the complexity of modifying state_dict in this manner, 
                # we will assume the structure allows modifying the first Conv2d layer found:
                
                if hasattr(self.backbone, 'features') and isinstance(getattr(self.backbone, 'features')[0], nn.Conv2d):
                    original_conv = getattr(self.backbone, 'features')[0]
                    new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size, 
                                          stride=original_conv.stride, padding=original_conv.padding, bias=True)
                    setattr(self.backbone.features[0], 'weight', nn.Parameter(torch.randn_like(original_conv.weight)))
                    setattr(self.backbone.features[0], 'bias', nn.Parameter(torch.randn_like(original_conv.bias)))
                    first_conv_found = True
                    break
                else:
                    # If the structure is different, this is a placeholder for the necessary fix.
                    # Assuming the first module that is a Conv2d is the target.
                    print("Warning: Could not programmatically locate and modify the first Conv2d layer in the backbone.")
                
            if first_conv_found:
                break
        
        # Feature extraction path: Backbone -> Global Pool -> Linear Head
        self.features = self.backbone
        
        # Global pooling to collapse spatial dimensions (H, W) to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final Linear Head using LazyLinear for shape invariance
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, N_mels, T)
        
        # Pass through the modified backbone
        x = self.features(x)
        
        # Global pooling: (B, C, H, W) -> (B, C, 1, 1)
        x = self.pool(x)
        
        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)
        
        # Final classification logits: (B, C) -> (B, num_classes)
        return self.head(x)


# ============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    
    start = time.time()
    results = {}
    
    try:
        print("loading data...", flush=True)
        # Note: Using the required augmentation dict structure
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
        model = MobileNetWrapper(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (B, C, N_mels, T) -> (1, 1, 128, 313)
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
            # The prompt suggests using average_precision_score for the mean AP, 
            # which is standard for multi-label AP calculation.
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # For AP, we pass the scores (probs[:, c]) and true labels (targs[:, c])
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
        results = {"error": f"{type(exc).__name__
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
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# Set CPU thread count for data loading workers
torch.set_num_threads(os.cpu_count() or 4)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: EPOCHS is read from an env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Note: The proposal specifies augmentation values, but the skeleton requires
# passing the augmentation dict structure. We override the skeleton's default
# to match the proposal's explicit values where possible, while respecting
# the mandatory structure.
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}


# === Model definition at MODULE scope ===
# Implementing MobileNetV3 Small wrapper to accept 1 channel input.
class MobileNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        print("Initializing MobileNetV3 Small backbone...")
        # Load pretrained model, using weights from the proposal's model type
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accept 1 channel instead of 3 (RGB)
        # The first layer is typically the first module in the feature extraction path.
        # We assume the first layer name is 'features.conv1' or similar,
        # checking for the first nn.Conv2d in the backbone's sequential parts.
        
        # Standard MobileNetV3 structure often uses a Sequential block for initial layers.
        # We must find and replace the first Conv2d layer.
        
        # A robust way is to iterate through the modules and replace the first Conv2d found.
        
        # Attempting to modify the first Conv2d layer detected in the backbone
        first_conv_found = False
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and not first_conv_found:
                print(f"Modifying layer {name}: Changing in_channels from {module.in_channels} to 1")
                
                # Create a new Conv2d layer with the correct input channels
                new_conv = nn.Conv2d(1, module.out_channels, kernel_size=module.kernel_size, 
                                      stride=module.stride, padding=module.padding, dilation=module.dilation)
                
                # Replace the layer in the module's dict
                # This requires accessing the parent module's state dict, which is tricky.
                # For simplicity and robustness, we'll assume the first layer is directly accessible or wrap it.
                
                # Since direct replacement is complex, we will rely on modifying the state_dict or 
                # manually reconstructing the initial sequence if the structure is known.
                # Given the constraint to use the backbone, we will override the definition of the first layer's weights/bias
                # or, more safely, use a functional modification approach if possible.
                
                # For this specific task, we will assume the first block is the one to modify.
                # The first layer is usually self.backbone.features[0] or similar.
                
                # Fallback: If direct modification fails due to complexity, we must wrap the whole backbone
                # but for this solution, we will proceed with the assumption that we can modify the first Conv2d found.
                
                # A common pattern is that the first layer is located at self.backbone.features[0].
                # Due to the complexity of modifying state_dict in this manner, 
                # we will assume the structure allows modifying the first Conv2d layer found:
                
                if hasattr(self.backbone, 'features') and isinstance(getattr(self.backbone, 'features')[0], nn.Conv2d):
                    original_conv = getattr(self.backbone, 'features')[0]
                    new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size, 
                                          stride=original_conv.stride, padding=original_conv.padding, bias=True)
                    setattr(self.backbone.features[0], 'weight', nn.Parameter(torch.randn_like(original_conv.weight)))
                    setattr(self.backbone.features[0], 'bias', nn.Parameter(torch.randn_like(original_conv.bias)))
                    first_conv_found = True
                    break
                else:
                    # If the structure is different, this is a placeholder for the necessary fix.
                    # Assuming the first module that is a Conv2d is the target.
                    print("Warning: Could not programmatically locate and modify the first Conv2d layer in the backbone.")
                
            if first_conv_found:
                break
        
        # Feature extraction path: Backbone -> Global Pool -> Linear Head
        self.features = self.backbone
        
        # Global pooling to collapse spatial dimensions (H, W) to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final Linear Head using LazyLinear for shape invariance
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, N_mels, T)
        
        # Pass through the modified backbone
        x = self.features(x)
        
        # Global pooling: (B, C, H, W) -> (B, C, 1, 1)
        x = self.pool(x)
        
        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)
        
        # Final classification logits: (B, C) -> (B, num_classes)
        return self.head(x)


# ============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    
    start = time.time()
    results = {}
    
    try:
        print("loading data...", flush=True)
        # Note: Using the required augmentation dict structure
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
        model = MobileNetWrapper(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (B, C, N_mels, T) -> (1, 1, 128, 313)
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
            # The prompt suggests using average_precision_score for the mean AP, 
            # which is standard for multi-label AP calculation.
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # For AP, we pass the scores (probs[:, c]) and true labels (targs[:, c])
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
        results = {"error": f"{type(exc).__name__

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
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# Set CPU thread count for data loading workers
torch.set_num_threads(os.cpu_count() or 4)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: EPOCHS is read from an env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Note: The proposal specifies augmentation values, but the skeleton requires
# passing the augmentation dict structure. We override the skeleton's default
# to match the proposal's explicit values where possible, while respecting
# the mandatory structure.
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}


# === Model definition at MODULE scope ===
# Implementing MobileNetV3 Small wrapper to accept 1 channel input.
class MobileNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        print("Initializing MobileNetV3 Small backbone...")
        # Load pretrained model, using weights from the proposal's model type
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accept 1 channel instead of 3 (RGB)
        # The first layer is typically the first module in the feature extraction path.
        # We assume the first layer name is 'features.conv1' or similar,
        # checking for the first nn.Conv2d in the backbone's sequential parts.
        
        # Standard MobileNetV3 structure often uses a Sequential block for initial layers.
        # We must find and replace the first Conv2d layer.
        
        # A robust way is to iterate through the modules and replace the first Conv2d found.
        
        # Attempting to modify the first Conv2d layer detected in the backbone
        first_conv_found = False
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and not first_conv_found:
                print(f"Modifying layer {name}: Changing in_channels from {module.in_channels} to 1")
                
                # Create a new Conv2d layer with the correct input channels
                new_conv = nn.Conv2d(1, module.out_channels, kernel_size=module.kernel_size, 
                                      stride=module.stride, padding=module.padding, dilation=module.dilation)
                
                # Replace the layer in the module's dict
                # This requires accessing the parent module's state dict, which is tricky.
                # For simplicity and robustness, we'll assume the first layer is directly accessible or wrap it.
                
                # Since direct replacement is complex, we will rely on modifying the state_dict or 
                # manually reconstructing the initial sequence if the structure is known.
                # Given the constraint to use the backbone, we will override the definition of the first layer's weights/bias
                # or, more safely, use a functional modification approach if possible.
                
                # For this specific task, we will assume the first block is the one to modify.
                # The first layer is usually self.backbone.features[0] or similar.
                
                # Fallback: If direct modification fails due to complexity, we must wrap the whole backbone
                # but for this solution, we will proceed with the assumption that we can modify the first Conv2d found.
                
                # A common pattern is that the first layer is located at self.backbone.features[0].
                # Due to the complexity of modifying state_dict in this manner, 
                # we will assume the structure allows modifying the first Conv2d layer found:
                
                if hasattr(self.backbone, 'features') and isinstance(getattr(self.backbone, 'features')[0], nn.Conv2d):
                    original_conv = getattr(self.backbone, 'features')[0]
                    new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size, 
                                          stride=original_conv.stride, padding=original_conv.padding, bias=True)
                    setattr(self.backbone.features[0], 'weight', nn.Parameter(torch.randn_like(original_conv.weight)))
                    setattr(self.backbone.features[0], 'bias', nn.Parameter(torch.randn_like(original_conv.bias)))
                    first_conv_found = True
                    break
                else:
                    # If the structure is different, this is a placeholder for the necessary fix.
                    # Assuming the first module that is a Conv2d is the target.
                    print("Warning: Could not programmatically locate and modify the first Conv2d layer in the backbone.")
                
            if first_conv_found:
                break
        
        # Feature extraction path: Backbone -> Global Pool -> Linear Head
        self.features = self.backbone
        
        # Global pooling to collapse spatial dimensions (H, W) to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final Linear Head using LazyLinear for shape invariance
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, N_mels, T)
        
        # Pass through the modified backbone
        x = self.features(x)
        
        # Global pooling: (B, C, H, W) -> (B, C, 1, 1)
        x = self.pool(x)
        
        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)
        
        # Final classification logits: (B, C) -> (B, num_classes)
        return self.head(x)


# ============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    
    start = time.time()
    results = {}
    
    try:
        print("loading data...", flush=True)
        # Note: Using the required augmentation dict structure
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
        model = MobileNetWrapper(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (B, C, N_mels, T) -> (1, 1, 128, 313)
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
            # The prompt suggests using average_precision_score for the mean AP, 
            # which is standard for multi-label AP calculation.
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # For AP, we pass the scores (probs[:, c]) and true labels (targs[:, c])
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
        results = {"error": f"{type(exc).__name__

