# Experiment exp_003

- **Study:** study_20260413_203901_full_dataset_v1
- **Status:** running
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 21:23:53.226720+00:00
- **Started:** 2026-04-13 21:23:53.226720+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 3-conv CNN front-end (32->64 channels) → GRU(128) → Linear`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.15
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_codegen_retry_01
- exp_003_task_04_validate_code
- exp_003_codegen_retry_02
- exp_003_task_05_validate_code
- exp_003_codegen_retry_03
- exp_003_task_06_validate_code
- exp_003_codegen_retry_04
- exp_003_task_07_validate_code
