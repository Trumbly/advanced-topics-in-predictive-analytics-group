# Experiment exp_005

- **Study:** study_20260410_133357_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:36:37.228122+00:00
- **Started:** 2026-04-10 13:36:37.228122+00:00
- **Completed:** 2026-04-10 13:37:52.884753+00:00

## Model Config
- architecture: `2‑conv small CNN with global average pooling, no pretrained weights`
- hyperparams:
  - lr: 0.0001
  - batch_size: 128
  - epochs: 50
  - optimizer: adamw
  - weight_decay: 0.0005
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_task_04_execute_training
