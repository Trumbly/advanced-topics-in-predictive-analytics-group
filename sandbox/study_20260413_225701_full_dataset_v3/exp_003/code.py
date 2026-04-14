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
# The orchestrator resolves `training.device` from config.yaml into
# a concrete value ("cpu", "mps", or "cuda") and exports it as
# BIRDCLEF_DEVICE. Just trust the env var — do NOT call
# torch.cuda.is_available() or torch.backends.mps.is_available().
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from an env var so the orchestrator can override it.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "7"))
LR = 0.001 # From proposal
AUGMENTATION = {"time_shift": False, "noise_injection": True, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class SelfAttentionBlock(nn.Module):
    """
    A simplified attention mechanism applied to the feature map.
    It computes channel-wise attention weights (similar to SE block)
    and multiplies features by these weights.
    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, channel_dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_dim // 4, channel_dim, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 1. Global Average Pooling across spatial dimensions
        y = self.avg_pool(x).view(b, c, 1, 1) # (B, C, 1, 1)
        
        # 2. MLP transformation to get attention weights
        # Flatten (B, C) and apply linear layer (C -> C)
        attn = self.fc(y.view(b, c))
        
        # 3. Reshape back and apply sigmoid to get weights (B, C, 1, 1)
        attn = attn.unsqueeze(2).unsqueeze(3)
        attention_weights = self.sigmoid(attn)
        
        # 4. Re-weight the input features
        return x * attention_weights

class CnnAttentionModel(nn.Module):
    """
    3-conv CNN feature extractor -> Self-Attention Block -> Global Average Pooling -> Linear Head
    """
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # --- CNN Feature Extractor ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # H/2, W/2

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # H/4, W/4

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # H/8, W/8

        # --- Attention Block ---
        self.attention = SelfAttentionBlock(128)
        
        # --- Classifier Head ---
        # Use LazyLinear to handle variable output size after pooling/attention
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x) # Output: (B, 128, H/8, W/8)

        # Self-Attention
        x = self.attention(x)

        # Global Average Pooling (required by proposal)
        x = nn.AdaptiveAvgPool2d((1, 1))(x) # (B, 128, 1, 1)
        
        # Flatten: (B, 128)
        x = x.view(x.size(0), -1)
        
        # Dropout and Classifier Head
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits

# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # The data loader reads augmentation settings from the environment
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
        model = CnnAttentionModel(num_classes=num_classes, dropout_rate=0.1)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # This step is CRITICAL for lazy layers.
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channels=1, Height=128, Width=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
        PATIENCE = 3
        best_auc = 0.0
        epochs_no_improve = 0
        
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
                    # Sigmoid for probabilities, move to CPU/Numpy for sklearn
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                # Only calculate if there is at least one positive label in the validation set for this class
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # Note: The competition metric description implies using AP, not AP@5, 
                    # as average_precision_score computes the full AP.
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
            scheduler.step()

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(
                    f"early stopping at epoch {epoch + 1} "
                    f"(no improvement for {PATIENCE} epochs)",
                    flush=True,
                )
                break

        # Final results reporting based on the last epoch run
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
