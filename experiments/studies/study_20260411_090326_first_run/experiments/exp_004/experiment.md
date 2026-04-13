# Experiment exp_004

- **Study:** study_20260411_090326_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 09:11:18.069519+00:00
- **Started:** 2026-04-11 09:11:18.069519+00:00
- **Completed:** 2026-04-11 09:25:13.994073+00:00

## Model Config
- architecture: `4-layer ResNet block with SE attention`
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

## Results
- loss: 0.0921
- roc_auc_macro: 0.5286
- duration: 782.2s
- training curves:
  - loss (len=1): first=0.0921, last=0.0921, min=0.0921, max=0.0921
  - roc_auc_macro (len=1): first=0.5286, last=0.5286, min=0.5286, max=0.5286

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_task_04_execute_training
- exp_004_task_05_capture_metrics
- exp_004_task_06_analyze_results
