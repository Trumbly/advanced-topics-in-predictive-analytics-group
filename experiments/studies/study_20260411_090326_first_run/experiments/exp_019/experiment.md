# Experiment exp_019

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:50:22.946250+00:00
- **Started:** 2026-04-11 10:50:22.946250+00:00
- **Completed:** 2026-04-11 10:53:27.866890+00:00

## Model Config
- architecture: `3-conv CNN with fused residual blocks and SE attention`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_019_task_01_propose_architecture
- exp_019_task_02_generate_code
- exp_019_task_03_validate_code
- exp_019_task_04_execute_training
- exp_019_task_05_capture_metrics
- exp_019_recovery_01_error_recovery
- exp_019_task_06_validate_code
- exp_019_recovery_02_error_recovery
- exp_019_task_07_validate_code
- exp_019_task_08_execute_training
- exp_019_task_09_capture_metrics
