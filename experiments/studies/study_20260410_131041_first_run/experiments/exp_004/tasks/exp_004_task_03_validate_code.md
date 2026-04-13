# Task exp_004_task_03_validate_code

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-10 13:12:38.162929+00:00
- **Completed:** 2026-04-10 13:12:38.165203+00:00

## Code Used
```python
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import cnn_small_v1
from sklearn.metrics import roc_auc_score
import time
import numpy as np

try:
    # Load data
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        num_classes=206,
        train_samples=186478,
        val_samples=46623,
        torch_dtype=torch.float32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = cnn_small_v1(num_classes=num_classes)
    model.to(device)
    
    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0002)
    
    # Hyperparams
    epochs = 80
    batch_size = 256
    dropout = 0.2
    loss_history = []
    roc_auc_history = []
    total_time = 0.0
    start_time = time.time()
    
    # Training curves
    training_curves = {"loss": [], "roc_auc_macro": []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        for batch_idx, (spec, labels, class_weights) in enumerate(train_loader):
            if isinstance(spec, torch.Tensor):
                spec = spec.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        training_curves["loss"].append(avg_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, (spec, labels, class_weights) in enumerate(val_loader):
                if isinstance(spec, torch.Tensor):
                    spec = spec.to(device)
                labels = labels.to(device)
                
                outputs = model(spec)
                val_loss += criterion(outputs, labels).item()
                all_logits.append(outputs.cpu())
                all_labels.append(labels.cpu())
        avg_val_loss = val_loss / (len(val_loader))
        val_loss_history = avg_val_loss
        val_loss_history
        
        # Macro ROC
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        probs = 1 / (1 + np.exp(-all_logits))
        macro_roc = roc_auc_score(all_labels, probs)
        roc_auc_history.append(macro_roc)
        training_curves["roc_auc_macro"].append(macro_roc)
        
        total_time += avg_val_loss * 0.1  # placeholder for timing
    
    duration = time.time() - start_time
    
    # Build results
    results = {
        "metrics": {
            "roc_auc_macro": float(roc_auc_history[-1]),
            "loss": float(val_loss_history)
        },
        "training_curves": training_curves,
        "duration_seconds": float(duration)
    }
    
    # Write results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
except Exception as e:
    results = {"error": str(e)}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

```

## Output
- **validation:** failed

## Error
- **type:** ForbiddenPattern
- **message:** Forbidden pattern found: 'eval('
