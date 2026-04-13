# Study: full_dataset_v1

- **ID:** study_20260413_203901_full_dataset_v1
- **Status:** active
- **Mode:** autonomous
- **Hypothesis:** 25 epochs, cosine LR, pretrained backbones, full 233K dataset
- **Created:** 2026-04-13 20:39:01.717370+00:00
- **Updated:** 2026-04-13 21:23:53.226940+00:00

## Compute Budget
- max_experiments: 10
- max_wallclock_minutes: 480
- max_experiment_seconds: 7200
- max_epochs_per_run: 25

## Configuration
- pipeline: `config/pipelines/default_pipeline.yaml`
- dataset profile: `data/processed/dataset_profile.json`
- model registry: `registry/models.yaml`

## Progress
- experiments run: 3
- best experiment: exp_002
- best score: 0.9590005391279278
- submissions: 0
