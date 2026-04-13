# Experiment exp_005

- **Study:** study_20260410_131833_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:20:45.600690+00:00
- **Started:** 2026-04-10 13:20:45.600690+00:00
- **Completed:** 2026-04-10 13:21:55.838694+00:00

## Model Config
- architecture: `efficientnet_b0`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.0001
  - batch_size: 64
  - epochs: 30
  - optimizer: Adam
  - weight_decay: 0.0001
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_task_04_execute_training
