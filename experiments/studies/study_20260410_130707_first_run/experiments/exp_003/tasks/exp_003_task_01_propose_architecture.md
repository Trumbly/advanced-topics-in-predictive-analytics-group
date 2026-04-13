# Task exp_003_task_01_propose_architecture

- **Experiment:** exp_003
- **Type:** llm
- **Name:** propose_architecture
- **Status:** completed
- **Started:** 2026-04-10 13:08:17.231064+00:00
- **Completed:** 2026-04-10 13:08:31.554884+00:00

## Prompt Used
```
[SYSTEM]
You are an ML research agent specializing in audio classification on mel-spectrograms.
You are working on BirdCLEF 2026 — a 234-class MULTI-LABEL bird species classification task.
Key constraints:
  - The final Kaggle submission notebook must run on CPU only, within 90 minutes.
  - Audio has already been preprocessed into mel-spectrograms by a fixed pipeline.
  - You MUST only propose architectures that use models from the provided Model Registry
    or combinations thereof. Do not invent exotic or unsupported components.
  - Output is a vector of 234 independent sigmoid probabilities (NOT a softmax).


[USER]
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

## Model Registry (available building blocks)
### cnn_small_v1
- family: cnn | framework: torch | params: 0.5M | pretrained: none
- input_shape: 1x128x256 | output_dim: 234
- notes: Fast from-scratch baseline. 3 Conv2D blocks + global pool + linear head. Trains in under 2 minutes on CPU. Use as a sanity check before touching heavier architectures.

### efficientnet_b0
- family: cnn | framework: torch | params: 5.3M | pretrained: imagenet
- input_shape: 3x224x224 | output_dim: 234
- notes: Strong transfer learning baseline. Requires the spectrogram to be resized to 224x224 and stacked to 3 channels. Freeze the backbone for the first few epochs, then fine-tune the last block.

### resnet18
- family: cnn | framework: torch | params: 11.7M | pretrained: imagenet
- input_shape: 3x224x224 | output_dim: 234
- notes: Reliable mid-sized CNN. Heavier than EfficientNet-B0 per forward pass but easier to train stably. Good fallback when EfficientNet overfits on the small labeled subset.


## Previous Experiments (top-k by ROC-AUC, + recent failures)
### Recent Failures
- **exp_002** [failed]
  - arch: cnn_small_v1
  - hyperparams: lr=0.001, batch_size=32, epochs=50, optimizer=AdamW, weight_decay=0.01, dropout=0.2

- **exp_001** [failed]


## Last Result
- id: exp_002
- status: failed
- llm_model: gemma4:e4b
- architecture: cnn_small_v1
- hyperparams: {"lr": 0.001, "batch_size": 32, "epochs": 50, "optimizer": "AdamW", "weight_decay": 0.01, "dropout": 0.2}

## Your Task
Propose the next experiment: a model architecture + training configuration that is
likely to improve on the best score so far. Be specific about hyperparameters.

Return ONLY valid JSON matching this schema:
{
  "type": "object",
  "required": [
    "architecture",
    "pretrained_model",
    "hyperparams",
    "augmentation"
  ],
  "properties": {
    "architecture": {
      "type": "string",
      "description": "Architecture name (free-form) or registry model name"
    },
    "pretrained_model": {
      "type": [
        "string",
        "null"
      ],
      "description": "Name of a model from the registry, or null for from-scratch"
    },
    "hyperparams": {
      "type": "object",
      "required": [
        "lr",
        "batch_size",
        "epochs",
        "optimizer"
      ],
      "properties": {
        "lr": {
          "type": "number"
        },
        "batch_size": {
          "type": "integer"
        },
        "epochs": {
          "type": "integer"
        },
        "optimizer": {
          "type": "string"
        },
        "weight_decay": {
          "type": "number"
        },
        "dropout": {
          "type": "number"
        }
      }
    },
    "augmentation": {
      "type": "object",
      "properties": {
        "time_shift": {
          "type": "boolean"
        },
        "noise_injection": {
          "type": "boolean"
        },
        "mixup": {
          "type": "number"
        },
        "specaugment": {
          "type": "boolean"
        }
      }
    }
  }
}

```

## LLM Response
```
```json
{
  "architecture": "cnn_small_v1",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 5e-4,
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
```
```

## Output
- **architecture_proposal:** {'architecture': 'cnn_small_v1', 'pretrained_model': None, 'hyperparams': {'lr': 0.0005, 'batch_size': 32, 'epochs': 75, 'optimizer': 'AdamW', 'weight_decay': 0.02, 'dropout': 0.15}, 'augmentation': {'time_shift': False, 'noise_injection': False, 'mixup': 0.0, 'specaugment': True}}
