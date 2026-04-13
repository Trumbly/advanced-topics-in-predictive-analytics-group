# Task exp_001_task_01_propose_architecture

- **Experiment:** exp_001
- **Type:** llm
- **Name:** propose_architecture
- **Status:** failed
- **Started:** 2026-04-10 13:33:57.488038+00:00
- **Completed:** 2026-04-10 13:34:20.306119+00:00

## Prompt Used
```
[SYSTEM]
You are an ML research agent specializing in audio classification on mel-spectrograms.
You are working on BirdCLEF 2026 — a 234-class MULTI-LABEL bird species classification task.

Your job is to DESIGN architectures, not pick from a menu. Be creative.
Use what you know about audio classification, CNNs, attention, transfer
learning, and BirdCLEF's quirks (noisy field recordings, class imbalance,
variable-length audio, CPU-only inference constraint) to propose something
that has a real chance of improving the current best score.

How to use the Model Registry below:
  - The registry is a list of VERIFIED building blocks you can import by name.
    When a registry entry fits your idea, use it via its exact `import_snippet`
    (this guarantees the import works in the sandbox).
  - You are NOT restricted to the registry. You are encouraged to design
    custom `nn.Module` classes from scratch, combine standard PyTorch
    layers in new ways, or build ensembles. If you write a custom model,
    its full class definition must be included in the generated code.
  - You MAY also import any publicly available module from `torch`,
    `torch.nn`, `torchvision.models`, `torchaudio`, `sklearn`, etc.

Hard constraints (these DO apply, no exceptions):
  - The final Kaggle submission notebook must run on CPU only within 90 minutes.
    Do not propose architectures that cannot plausibly fit that budget for
    both training and inference on 35k spectrograms.
  - Output is a vector of 234 independent sigmoid probabilities (multi-label).
    Your final layer's output_dim MUST be 234. Do not use softmax.
  - Audio is preprocessed into mel-spectrograms by a fixed pipeline. You work
    on the spectrograms directly; do not propose audio-waveform models.
  - No GPU-only ops, no internet downloads, no custom CUDA kernels.


[USER]
No previous experiments yet. Start with the fastest possible baseline so the
pipeline is verified end-to-end: CnnSmallV1 from the registry, 2 epochs,
batch_size 32, Adam lr=1e-3, light augmentation (time_shift + noise_injection).
Goal right now is to confirm the full loop works. More ambitious architectures
come next.

```

## LLM Response
```
```python
# Registry import
from models import CnnSmallV1

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# ------------------- Hyper‑parameters -------------------
input_dim   = 256          # mel‑spectrogram: (seq_len, channels) → assume 256
n_classes   = 234          # BirdCLEF output dimension
batch_size  = 32
epochs      = 2
lr          = 1e-3
device      = torch.device('cpu')

# ------------------- Model -------------------
model = CnnSmallV1(input_dim=input_dim, n_classes=n_classes).to(device)
criterion = nn.BCEWithLogitsLoss()               # sigmoid in each column
optimizer = optim.Adam(model.parameters(), lr=lr)

# ------------------- Light augmentation -------------------
def time_shift_and_noise(x):
    """Shift start of spectrogram by 0‑20 frames (reflect padding) + tiny Gaussian noise."""
    # Random shift per batch (max 20 frames)
    shift = torch.randint(0, 20, (x.shape[0], 1), dtype=torch.long, device=device)
    x = torch.nn.functional.pad(x, (shift, 0), mode='reflect')
    # Noise: std = 0.01 per channel (very light)
    noise = torch.randn_like(x) * 0.01
    x += noise
    return x

# ------------------- Dummy dataset for verification -------------------
def dummy_dataset():
    """Generate 2 dummy batches: (32, 256, 3) spectrograms, random binary labels."""
    x = torch.randn(batch_size, 256, 3, device=device, requires_grad=False)
    y = torch.randint(0, n_classes, (batch_size, n_classes), dtype=torch.float32)  # dummy multi‑label
    return x, y

dataloader = DataLoader(dummy_dataset(), batch_size=batch_size, shuffle=True)

# ------------------- Training loop -------------------
for epoch in range(epochs):
    epoch_loss = 0.0
    for xb, yb in dataloader:
        xb = time_shift_and_noise(xb)          # augmentation (CPU only)
        optimizer.zero_grad()
        logits = model(xb)                     # (32, 234)
        loss = criterion(logits, yb)           # BCE loss, no softmax
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs} – avg loss: {avg_loss:.4f}")
```

**Explanation**

* **Model:** `CnnSmallV1` (registry‑provided) outputs 234 sigmoid‑ready scores.  
* **Augmentation:** `time_shift_and_noise` mimics the light augmentations you requested, all CPU‑compatible.  
* **Training:** 2 epochs, batch = 32, Adam lr = 1e‑3, BCEWithLogitsLoss → final logits have 234 columns.  
* **Verification:** Prints epoch loss; no GPU, no internet, under 90 min CPU runtime.
```

## Error
- **type:** ValueError
- **message:** propose_architecture: LLM did not return valid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
