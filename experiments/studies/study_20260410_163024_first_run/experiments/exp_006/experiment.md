# Experiment exp_006

- **Study:** study_20260410_163024_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 17:17:57.682593+00:00
- **Started:** 2026-04-10 17:17:57.682593+00:00
- **Completed:** 2026-04-10 19:00:21.279653+00:00

## Model Config
- architecture: `cnn_small_v1 baseline with longer training`
- hyperparams:
  - lr: 0.001
  - batch_size: 32
  - epochs: 5
  - optimizer: adam
  - weight_decay: 0
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0
  - specaugment: False

## Results
- loss: 0.0221
- roc_auc_macro: 0.8818
- duration: 6093.2s
- training curves:
  - loss (len=5): first=0.0323, last=0.0221, min=0.0221, max=0.0323
  - roc_auc_macro (len=5): first=0.6995, last=0.8818, min=0.6995, max=0.8818

## Tasks
- exp_006_task_01_propose_architecture
- exp_006_task_02_generate_code
- exp_006_task_03_validate_code
- exp_006_task_04_execute_training
- exp_006_task_05_capture_metrics
- exp_006_task_06_analyze_results
