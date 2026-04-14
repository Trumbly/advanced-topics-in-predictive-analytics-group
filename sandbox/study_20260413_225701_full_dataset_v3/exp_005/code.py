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

# === Hyperparameters from the proposal ===
# Note: LR is set in the skeleton block below, but we track the proposal's value: 0.001
LR = 1e-3

# Read EPOCHS from environment variable as per mandatory structure
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))

# Overriding default augmentation to match the proposal
AUGMENTATION = {
    "time_shift": False,
    "noise_injection": False,
    "mixup": 0.0,
    "specaugment": True
}

# === Model definition at MODULE scope ===
class CnnGruHybridModel(nn.Module):
    """
    3-block CNN feature extractor (32->64 channels) followed by nn.GRU(128)
    for temporal sequence modeling.
    Input: (B, 1, Mel, Time) -> Output: (B, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 3-Block CNN Feature Extractor ---
        # Input: (B, 1, 128, 313)
        self.cnn_blocks = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Output: (B, 32, 128, 313)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: (B, 64, 128, 313)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 64 -> 64 channels (keeping feature depth consistent for GRU)
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Output: (B, 64, 128, 313)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsample the Mel dimension (128) to 1, keeping Time (313) intact.
        self.mel_pool = nn.AdaptiveAvgPool2d((1, 1)) # Output: (B, 64, 1, 313)
        
        # --- GRU Layer ---
        # Input to GRU: (B, T, F) -> T=313, F=64
        self.gru = nn.GRU(
            input_size=64, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True # Crucial for sequence processing
        )
        
        # --- Final Classification Head ---
        # After GRU, we pool the sequence output (B, T, 128) down to a fixed vector (B, 128)
        # Then map to num_classes (206).
        self.fc_head = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1. CNN Feature Extraction
        x = self.cnn_blocks(x) # (B, 64, 128, 313)
        
        # 2. Pool Mel dimension
        x = self.mel_pool(x) # (B, 64, 1, 313)
        
        # 3. Permute for GRU: (B, C, T) -> (B, T, C)
        # Here C=64, T=313
        x = x.squeeze(2).permute(0, 2, 1) # (B, 313, 64)
        
        # 4. GRU Sequence Modeling
        # Output: (B, T, H) -> (B, 313, 128)
        output, _ = self.gru(x)
        
        # 5. Sequence Pooling: Average over the time dimension (dim=1)
        # Result: (B, 128)
        pooled_output = torch.mean(output, dim=1)
        
        # 6. Final Linear Classification Head
        logits = self.fc_head(pooled_output) # (B, 206)
        return logits


# ======================================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    
    # Variables derived from the dataset profile (hardcoded for this task)
    NUM_CLASSES = 206 

    try:
        print("loading data...", flush=True)
        # batch_size / num_workers / persistent_workers / prefetch_factor
        # are read from BIRDCLEF_* env vars (sourced from config.yaml).
        train_loader, val_loader, num_classes_check = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        
        # Safety check (should match 206)
        if num_classes_check != NUM_CLASSES:
             raise ValueError(f"Expected {NUM_CLASSES} classes, but data loader provided {num_classes_check}")
        
        print(
            f"data loaded: {num_classes_check} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected device.
        model = CnnGruHybridModel(num_classes=NUM_CLASSES)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # The final layer is nn.Linear, which is not lazy, but we run this
        # check to ensure correct parameter counting/initialization.
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channels=1, Mel=128, Time=313)
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
                    # Sigmoid for probabilities, move to CPU for numpy/sklearn
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    try:
                        aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
                    except ValueError: # Handle case where all labels are the same in a column
                        pass
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # average_precision_score computes AP, which is the integral of PR curve.
                    # For BirdCLEF, this serves as the proxy for the required metric.
                    try:
                        aps.append(average_precision_score(targs[:, c], probs[:, c]))
                    except ValueError:
                        pass
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

        # Final results compilation
        final_metrics = {
            "roc_auc_macro": curves["roc_auc_macro"][-1],
            "cmap_at_5": curves["cmap_at_5"][-1],
            "f1_macro": curves["f1_macro"][-1],
            "loss": curves["loss"][-1],
        }
        
        results = {
            "metrics": final_metrics,
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={final_metrics['roc_auc_macro']:.4f} "
            f"cmap@5={final_metrics['cmap_at_5']:.4f} "
            f"f1={final_metrics['f1_macro']:.4f}",
            flush=True,
        )
        
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    # Write results.json
    with open("results.json", "w") as fh:
        json.dump(results, fh)
