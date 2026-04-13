# Experiment exp_005

- **Study:** study_20260412_093854_ui_study_start_test
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 07:48:44.336520+00:00
- **Started:** 2026-04-12 07:48:44.336520+00:00
- **Completed:** 2026-04-12 07:51:28.418195+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 3-conv CNN front-end (2-3 blocks) extracting spectral features, followed by a 1-layer GRU(256) processing the temporal sequence, and a final linear projection.`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0005
  - dropout: 0.3
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_recovery_01_error_recovery
- exp_005_task_04_validate_code
- exp_005_task_05_execute_training
- exp_005_recovery_02_error_recovery
- exp_005_task_06_validate_code
- exp_005_task_07_execute_training
- exp_005_task_08_capture_metrics
