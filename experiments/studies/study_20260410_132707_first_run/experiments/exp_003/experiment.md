# Experiment exp_003

- **Study:** study_20260410_132707_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:28:32.102807+00:00
- **Started:** 2026-04-10 13:28:32.102807+00:00
- **Completed:** 2026-04-10 13:29:28.051080+00:00

## Model Config
- architecture: `resnet18`
- pretrained: `resnet18`
- hyperparams:
  - lr: 0.001
  - batch_size: 64
  - epochs: 20
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.1
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
