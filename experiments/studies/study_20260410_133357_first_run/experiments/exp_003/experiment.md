# Experiment exp_003

- **Study:** study_20260410_133357_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:35:09.874689+00:00
- **Started:** 2026-04-10 13:35:09.874689+00:00
- **Completed:** 2026-04-10 13:35:49.283596+00:00

## Model Config
- architecture: `Simple 3‑conv2d CNN with global average pooling and dropout (cnn_small_v1)`
- pretrained: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 64
  - epochs: 30
  - optimizer: adamw
  - weight_decay: 0.0005
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: False
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
