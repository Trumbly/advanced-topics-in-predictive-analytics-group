# Study: full_dataset_v4

- **ID:** study_20260414_211157_full_dataset_v4
- **Status:** completed
- **Mode:** autonomous
- **Hypothesis:** 3 experiments x 3 epochs with pretrained models, optimizing F1 and ROC-AUC
- **Created:** 2026-04-14 21:11:57.620805+00:00
- **Updated:** 2026-04-14 23:25:26.175369+00:00

## Compute Budget
- max_experiments: 3
- max_wallclock_minutes: 480
- max_experiment_seconds: 14400
- max_epochs_per_run: 4

## Configuration
- pipeline: `config/pipelines/default_pipeline.yaml`
- dataset profile: `data/processed/dataset_profile.json`
- model registry: `registry/models.yaml`

## Progress
- experiments run: 3
- best experiment: exp_001
- best score: 0.5857953319833844
- submissions: 0
