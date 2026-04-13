# Experiment exp_001

- **Study:** study_20260413_214602_full_dataset_v2
- **Status:** running
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 21:46:03.469074+00:00
- **Started:** 2026-04-13 21:46:03.469074+00:00

## Model Config
- architecture: `[custom_cnn] cnn_small_v1 baseline`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
