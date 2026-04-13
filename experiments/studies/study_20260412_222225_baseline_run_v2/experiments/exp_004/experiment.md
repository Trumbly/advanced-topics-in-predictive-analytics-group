# Experiment exp_004

- **Study:** study_20260412_222225_baseline_run_v2
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 22:54:41.724278+00:00
- **Started:** 2026-04-12 22:54:41.724278+00:00
- **Completed:** 2026-04-12 23:13:34.469278+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 4-conv CNN front-end (BatchNorm+ReLU) -> Flatten -> 2-layer GRU(128) -> Dropout -> Linear Head`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.3
  - specaugment: True

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
