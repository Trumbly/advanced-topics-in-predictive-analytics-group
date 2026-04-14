# Experiment exp_005

- **Study:** study_20260413_225701_full_dataset_v3
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-14 04:36:22.164499+00:00
- **Started:** 2026-04-14 04:36:22.164499+00:00
- **Completed:** 2026-04-14 07:44:49.985869+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 3-block CNN feature extractor (32->64 channels) followed by nn.GRU(128) for temporal sequence modeling`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 7
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Results
- cmap_at_5: 0.0875
- f1_macro: 0.0764
- loss: 0.3689
- roc_auc_macro: 0.8706
- duration: 10948.1s
- training curves:
  - loss (len=7): first=0.4967, last=0.3689, min=0.3689, max=0.4967
  - roc_auc_macro (len=7): first=0.7288, last=0.8706, min=0.7288, max=0.8706
  - cmap_at_5 (len=7): first=0.0202, last=0.0875, min=0.0202, max=0.0875
  - f1_macro (len=7): first=0.0195, last=0.0764, min=0.0195, max=0.0781

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_task_04_execute_training
- exp_005_task_05_capture_metrics
- exp_005_recovery_01_error_recovery
- exp_005_task_06_validate_code
- exp_005_task_07_execute_training
- exp_005_task_08_capture_metrics
- exp_005_task_09_analyze_results
