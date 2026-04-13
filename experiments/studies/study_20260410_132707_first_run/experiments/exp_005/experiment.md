# Experiment exp_005

- **Study:** study_20260410_132707_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:30:08.779375+00:00
- **Started:** 2026-04-10 13:30:08.779375+00:00
- **Completed:** 2026-04-10 13:31:07.275506+00:00

## Model Config
- architecture: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 64
  - epochs: 20
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.0
- augmentation:
  - time_shift: True
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_task_04_execute_training
- exp_005_task_05_capture_metrics
