# Experiment exp_005

- **Study:** study_20260410_134457_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:47:44.370136+00:00
- **Started:** 2026-04-10 13:47:44.370136+00:00
- **Completed:** 2026-04-10 13:48:52.744490+00:00

## Model Config
- architecture: `cnn_small_v1 with added 1x1 convolution to accommodate 313 time frames`
- pretrained: `cnn_small_v1`
- hyperparams:
  - lr: 0.0001
  - batch_size: 64
  - epochs: 30
  - optimizer: Adam
  - weight_decay: 0.0001
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: True
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_task_04_execute_training
