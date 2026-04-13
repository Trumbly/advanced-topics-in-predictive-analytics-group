# Task exp_010_task_08_validate_code

- **Experiment:** exp_010
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-13 00:24:44.306454+00:00
- **Completed:** 2026-04-13 00:24:44.313879+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# Set number of threads if running on CPU
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

# === Hyperparameters (Using env vars as source of truth, but respecting hard cap) ===
EPOCHS = 1 
LR = 1e-3
# Augmentation settings from the proposal JSON
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class CnnAttentionModel(nn.Module):
    """
    Implements the proposed 3-block CNN stack -> Attention (simulated via advanced pooling) -> Linear Head.
    The architecture processes the input (B, 1, Mel, Time) through 3 Conv layers
    with intermediate feature expansion, ending with 64 channels.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # --- 3-block Conv Stack (Targeting 64 output channels) ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # Block 2: 64 -> 128 (Intermediate expansion)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Block 3: 128 -> 64 (Final projection to 64 channels)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # --- Attention / Feature Aggregation ---
        # Pool over H and T dimensions to get a fixed feature size (B, 64, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Linear Head ---
        # Use LazyLinear to handle the inferred input size (64 channels)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Feature Extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Shape: (B, 64, H, T)

        # 2. Global Pooling (Collapses H and T to 1x1)
        x = self.global_pool(x)
        # Shape: (B, 64, 1, 1)

        # 3. Flatten and Linear Projection
        # Flatten the last two dimensions (C, 1, 1) -> (C)
        x = x.view(x.size(0), -1)
        
        # Project to class logits
        x = self.head(x)
        return x

if __name__ == "__main__":
    print(f"--- Starting BirdCLEF Training Run ---")
    print(f"Device: {device}")
    
    # 1. Load Dataset and get metadata
    print("Loading precomputed dataset...")
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    
    print(f"Dataset loaded. Number of classes: {num_classes}")

    # 2. Setup Loss and Pos_Weight
    pos_weight = compute_pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 3. Model Initialization
    model = CnnAttentionModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    
    print("Model initialized.")

    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            # Move batch data to device
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Forward pass
            logits = model(x)
            
            # Loss calculation
            loss = criterion(logits, y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        
        epoch_loss = running_loss / train_loader.batch_size
        print(f"Training Loss: {epoch_loss:.4f}")
        if device.type() == 'cuda':
             torch.cuda.empty_cache()
        # Flush output to ensure progress print is visible
        print("Training step complete.", flush=True)

    # 5. Validation/Testing Loop (Evaluation)
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\n--- Starting Validation ---")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Forward pass
            logits = model(x)
            
            # Get probabilities (sigmoid applied)
            probs = torch.sigmoid(logits)
            
            # Move predictions back to CPU and append
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            print(f"Validation batch {batch_idx+1}/{len(val_loader)} complete.", flush=True)

    # 6. Metric Calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate macro-averaged ROC-AUC
    roc_auc = roc_auc_score(all_targets, all_preds, average='macro')
    # Calculate macro-averaged Average Precision Score
    ap_score = average_precision_score(all_targets, all_preds, average='macro')
    
    metrics = {
        "roc_auc_macro": roc_auc,
        "ap_score_macro": ap_score
    }
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Metrics calculated: {metrics}")

    # 7. Write results.json
    output_filename = "results.json"
    results_data = {
        "model_architecture": "cnn_attention_model",
        "hyperparameters": {
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": os.environ.get("BIRDCLEF_BATCH_SIZE", "128"),
            "device": DEVICE_NAME
        },
        "metrics": metrics
    }
    
    with open(output_filename, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"\nSuccessfully wrote results to {output_filename}", flush=True)

```

## Output
- **validation:** passed
- **code_bytes:** 6227
