# Experiment exp_016

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 06:05:02.239625+00:00
- **Started:** 2026-04-13 06:05:02.239625+00:00
- **Completed:** 2026-04-13 06:29:41.825846+00:00

## Model Config
- architecture: `[cnn_attention] MobileNetV3 Small backbone followed by a dedicated Time-Aware Self-Attention block`
- pretrained: `mobilenet_v3_small`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_016_task_01_propose_architecture
- exp_016_task_02_generate_code
- exp_016_task_03_validate_code
- exp_016_codegen_retry_01
- exp_016_task_04_validate_code
- exp_016_codegen_retry_02
- exp_016_task_05_validate_code
- exp_016_codegen_retry_03
- exp_016_task_06_validate_code
- exp_016_codegen_retry_04
- exp_016_task_07_validate_code
- exp_016_codegen_retry_05
- exp_016_task_08_validate_code
- exp_016_recovery_01_error_recovery
- exp_016_task_09_validate_code
- exp_016_task_10_execute_training
- exp_016_task_11_capture_metrics
- exp_016_recovery_02_error_recovery
- exp_016_task_12_validate_code
- exp_016_task_13_execute_training
- exp_016_task_14_capture_metrics
- exp_016_recovery_03_error_recovery
- exp_016_task_15_validate_code
- exp_016_task_16_execute_training
- exp_016_task_17_capture_metrics
- exp_016_recovery_04_error_recovery
- exp_016_task_18_validate_code
- exp_016_task_19_execute_training
- exp_016_task_20_capture_metrics
- exp_016_recovery_05_error_recovery
- exp_016_task_21_validate_code
- exp_016_task_22_execute_training
- exp_016_task_23_capture_metrics
