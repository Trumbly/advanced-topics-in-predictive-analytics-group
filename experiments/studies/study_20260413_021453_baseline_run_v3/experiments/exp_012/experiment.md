# Experiment exp_012

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 05:15:00.941023+00:00
- **Started:** 2026-04-13 05:15:00.941023+00:00
- **Completed:** 2026-04-13 05:29:34.758735+00:00

## Model Config
- architecture: `[mobilenet_v3_small] MobileNetV3 Small via TorchvisionAdapter, optimized for spectrogram feature extraction`
- pretrained: `mobilenet_v3_small`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.25
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Results
- cmap_at_5: 0.1008
- f1_macro: 0.0882
- loss: 1.1355
- roc_auc_macro: 0.4597
- duration: 13.9s
- training curves:
  - loss (len=1): first=1.1355, last=1.1355, min=1.1355, max=1.1355
  - roc_auc_macro (len=1): first=0.4597, last=0.4597, min=0.4597, max=0.4597
  - cmap_at_5 (len=1): first=0.1008, last=0.1008, min=0.1008, max=0.1008
  - f1_macro (len=1): first=0.0882, last=0.0882, min=0.0882, max=0.0882

## Tasks
- exp_012_task_01_propose_architecture
- exp_012_task_02_generate_code
- exp_012_task_03_validate_code
- exp_012_task_04_execute_training
- exp_012_task_05_capture_metrics
- exp_012_recovery_01_error_recovery
- exp_012_task_06_validate_code
- exp_012_task_07_execute_training
- exp_012_task_08_capture_metrics
- exp_012_recovery_02_error_recovery
- exp_012_task_09_validate_code
- exp_012_task_10_execute_training
- exp_012_task_11_capture_metrics
- exp_012_recovery_03_error_recovery
- exp_012_task_12_validate_code
- exp_012_codegen_retry_01
- exp_012_task_13_validate_code
- exp_012_codegen_retry_02
- exp_012_task_14_validate_code
- exp_012_task_15_execute_training
- exp_012_recovery_04_error_recovery
- exp_012_task_16_validate_code
- exp_012_task_17_execute_training
- exp_012_task_18_capture_metrics
- exp_012_task_19_analyze_results
