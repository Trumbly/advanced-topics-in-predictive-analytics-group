# Task exp_003_task_03_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-10 13:08:58.070286+00:00
- **Completed:** 2026-04-10 13:08:58.072795+00:00

## Code Used
```python
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import CNNSmallV1
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# --- Configuration from Proposal ---
CONFIG = {
    "architecture": "cnn_small_v1",
    "pretrained_model": None,
    "hyperparams": {
        "lr": 0.0005,
        "batch_size": 32,
        "epochs": 75,
        "optimizer": "AdamW",
        "weight_decay": 0.02,
        "dropout": 0.15
    },
    "augmentation": {
        "time_shift": False,
        "noise_injection": False,
        "mixup": 0.0,
        "specaugment": True
    }
}

# --- Main Training Function ---
def train_model():
    start_time = time.time()
    
    try:
        # 1. Load Data
        train_loader, val_loader, num_classes = load_precomputed_dataset()
        
        # 2. Initialize Model
        # Assuming CNNSmallV1 takes (num_classes) for initialization
        model = CNNSmallV1(num_classes=num_classes)
        
        # 3. Setup Training Components
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        params = CONFIG["hyperparams"]
        optimizer_name = params["optimizer"]
        
        if optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        criterion = torch.nn.CrossEntropyLoss() # Using CrossEntropyLoss for simplicity, though multi-label often uses BCEWithLogitsLoss
        # For multi-label classification (like BirdCLEF), BCEWithLogitsLoss is standard.
        criterion = torch.nn.BCEWithLogitsLoss() 
        
        # 4. Training Loop
        history = {
            "loss": [],
            "roc_auc_macro": []
        }
        
        print(f"Starting training on {device} for {params['epochs']} epochs...")

        for epoch in range(params["epochs"]):
            model.train()
            running_loss = 0.0
            
            # Training Phase
            train_loss = 0.0
            pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['epochs']} [Train]")
            for batch in pbar_train:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Assuming inputs are (batch_size, 1, 128, 313) and targets are (batch_size, num_classes) float/long
                # Adjusting target handling for BCEWithLogitsLoss (requires float targets)
                if targets.dtype != torch.float:
                    targets = targets.float() 
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                running_loss += loss.item()
                
                pbar_train.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / train_loader.batch_size
            
            # Validation Phase
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{params['epochs']} [Val]")
                for batch in pbar_val:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    if targets.dtype != torch.float:
                        targets = targets.float()
                        
                    outputs = model(inputs)
                    
                    # Validation Loss
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Predictions for AUC: Sigmoid -> Probabilities
                    probs = torch.sigmoid(outputs)
                    all_preds.append(probs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
            
            avg_val_loss = (val_loss / val_loader.batch_size)
            
            # Calculate Macro ROC AUC
            all_preds_np = np.concatenate(all_preds, axis=0)
            all_targets_np = np.concatenate(all_targets, axis=0)
            
            try:
                roc_auc = roc_auc_score(all_targets_np, all_preds_np, average='macro')
            except ValueError:
                # Handle case where only one class is present or all predictions are identical
                roc_auc = 0.0 
                
            history["loss"].append(avg_val_loss)
            history["roc_auc_macro"].append(roc_auc)
            
            print(f"Epoch {epoch+1}/{params['epochs']} | Val Loss: {avg_val_loss:.4f} | Macro AUC: {roc_auc:.4f}")

        # 5. Final Evaluation (Using the last state)
        model.eval()
        final_all_preds = []
        final_all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                probs = torch.sigmoid(outputs)
                final_all_preds.append(probs.cpu().numpy())
                final_all_targets.append(targets.cpu().numpy())
        
        final_all_preds_np = np.concatenate(final_all_preds, axis=0)
        final_all_targets_np = np.concatenate(final_all_targets, axis=0)
        
        final_roc_auc = roc_auc_score(final_all_targets_np, final_all_preds_np, average='macro')
        final_loss = history["loss"][-1]
        
        end_time = time.time()

        # 6. Write Results
        results = {
            "metrics": {
                "roc_auc_macro": final_roc_auc,
                "loss": final_loss
            },
            "training_curves": {
                "loss": history["loss"],
                "roc_auc_macro": history["roc_auc_macro"]
            },
            "duration_seconds": end_time - start_time
        }
        
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        print("Training complete. Results saved to results.json")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        error_results = {
            "metrics": {"roc_auc_macro": None, "loss": None},
            "training_curves": {"loss": None, "roc_auc_macro": None},
            "duration_seconds": time.time() - start_time,
            "error": str(e)
        }
        with open("results.json", "w") as f:
            json.dump(error_results, f, indent=4)

if __name__ == "__main__":
    import numpy as np # Moved import here to ensure numpy is available for AUC calculation
    train_model()

```

## Output
- **validation:** failed

## Error
- **type:** ForbiddenPattern
- **message:** Forbidden pattern found: 'eval('
