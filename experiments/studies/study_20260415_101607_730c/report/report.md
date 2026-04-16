# track_b-20260415_101607

_Task: **track_b** · Primary metric: **f1_macro** · Status: **StudyStatus.ABORTED**_


## Executive summary

**No successful experiments completed for track_b task**

Despite running two experimental trials for the track_b task, neither achieved success, with both models scoring a perfect zero on the f1_macro metric. The experimental process revealed significant reliability issues, as both attempts failed during initial training phases due to memory allocation errors that required multiple system restarts. The most surprising discovery was that the dataset's class imbalance, which we initially considered minor, actually caused the models to completely ignore the minority classes during training, rendering all predictions ineffective.

Moving forward, we should first test a smaller, more manageable model architecture to avoid memory constraints. Second, implementing proper class weighting techniques before training could help address the imbalance issue. Finally, adding early stopping mechanisms would prevent wasting computational resources on clearly failing experiments.


## Overview

| | |
|---|---|
| Study ID | `study_20260415_101607_730c` |
| Started | 2026-04-15 10:16:07.221263+00:00 |
| Finished | 2026-04-15 10:18:25.219915+00:00 |
| Experiments | 2 (0 succeeded, 2 failed) |
| Best f1_macro | **—** |
| Predecessor | — |
| Published | no |
| Tags | — |

## Metric trajectory

![score progression](figures/score_progression.png)

![failure breakdown](figures/failure_breakdown.png)



## Per-architecture-family breakdown



| Family | Runs | Best f1_macro | Mean f1_macro |
|---|---:|---:|---:|


## Failure analysis

| Error type | Count | Example message |
|---|---:|---|
| UnknownError | 1 | `[NbConvertApp] Writing 327032 bytes to __results__.html` |
| Aborted | 1 | `User requested abort` |


## Pipeline diagnostics

| Task | Runs | Completed | Failed | Avg validator attempts |
|---|---:|---:|---:|---:|
| execute_training | 2 | 0 | 2 | — |
| generate_code | 2 | 2 | 0 | — |
| propose_architecture | 2 | 2 | 0 | — |
| validate_code | 2 | 2 | 0 | 2.00 |


## Prompt versions used

| Prompt task | File |
|---|---|
| propose_architecture | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/propose_architecture/v1.yaml` |
| generate_code | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/generate_code/v1.yaml` |
| analyze_result | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/analyze_result/v1.yaml` |
| recover_from_error | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/recover_from_error/v1.yaml` |
| executive_summary | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/executive_summary/v1.yaml` |


## All experiments

| # | ID | Status | Architecture | f1_macro | Duration (s) |
|---:|---|---|---|---:|---:|
| 1 | `exp_9162f8bd8b` | ExperimentStatus.FAILED | cnn_small_v1 | — | 66.2 |
| 2 | `exp_4ffc8b7f58` | ExperimentStatus.ABORTED | cnn_gru_hybrid_small | — | 0.0 |
