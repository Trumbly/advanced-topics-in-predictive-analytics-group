# Task exp_003_task_02_generate_code

- **Experiment:** exp_003
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:08:31.555983+00:00
- **Completed:** 2026-04-10 13:08:58.069178+00:00

## Prompt Used
```
[SYSTEM]
You are a Python ML engineer. Your task is to write a single, self-contained training
script that trains and evaluates a model for the BirdCLEF 2026 multi-label task.

STRICT RULES:
  - Use PyTorch (unless the model registry entry specifies TensorFlow)
  - Import the fixed data loader: `from pipelines.data_loader import load_precomputed_dataset`
    This returns `(train_loader, val_loader, num_classes)`.
  - Import model classes from `pipelines.models` when available, or use the `import_snippet`
    from the registry exactly as given.
  - Write final results to `results.json` with this schema:
      {{"metrics": {{"roc_auc_macro": <float>, "loss": <float>}},
        "training_curves": {{"loss": [...], "roc_auc_macro": [...]}},
        "duration_seconds": <float>}}
  - Do NOT import: `os.system`, `subprocess`, `urllib`, `requests`
  - Do NOT download anything from the internet
  - Do NOT write outside the current directory
  - Catch exceptions and still write a results.json with an `error` field if training fails


[USER]
## Architecture Proposal
{
  "architecture": "cnn_small_v1",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.0005,
    "batch_size": 32,
    "epochs": 75,
    "optimizer": "AdamW",
    "weight_decay": 0.02,
    "dropout": 0.15
  },
  "augmentation": {
    "time_shift": false,
    "noise_injection": false,
    "mixup": 0.0,
    "specaugment": true
  }
}

## Dataset Profile
- num_classes: 206
- num_samples: 233101
- spectrogram_shape: (1, 128, 313)
- sample_rate: 32000 Hz
- imbalance_ratio: 6854.00 (min=1, max=6854)
- split: stratified_kfold (seed=42)
- train/val: 186478/46623
- most-populated classes:
    - coffal1: 6854
    - rufnig1: 4907
    - bncfly: 4630
- least-populated classes:
    - 116570: 1
    - 23724: 1
    - 23150: 3

## Your Task
Write a complete Python training script implementing the architecture above.
Return ONLY the Python code, no explanations, no markdown fences.

```

## LLM Response
```
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
- **code:** from pipelines.data_loader import load_precomputed_dataset
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

