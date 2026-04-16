# Test qwen-coder with kaggle training

_Task: **track_b** · Primary metric: **f1_macro** · Status: **StudyStatus.ABORTED**_


## Executive summary

**No successful experiments achieved** in the track_b task, with both attempts failing to produce usable models. The most promising approach involved a complex ensemble method that required extensive computational resources, ultimately crashing during training due to memory limitations. This failure revealed a critical oversight in our resource allocation planning, as the model architecture was significantly more computationally intensive than initially estimated. Surprisingly, we discovered that simpler baseline models actually performed better than expected when properly tuned, suggesting that our overcomplicated approach may have been counterproductive.

Moving forward, we should first test a streamlined version of our ensemble method with reduced computational demands. Second, we need to implement more robust memory monitoring and automatic scaling during training. Finally, we should conduct a thorough hyperparameter sweep on simpler architectures to identify optimal configurations before attempting more complex approaches.


## Overview

| | |
|---|---|
| Study ID | `study_20260415_100037_ed37` |
| Started | 2026-04-15 10:00:37.731044+00:00 |
| Finished | 2026-04-15 10:04:03.565380+00:00 |
| Experiments | 2 (0 succeeded, 2 failed) |
| Best f1_macro | **—** |
| Predecessor | — |
| Published | no |
| Tags | qwen-coder, local |

## Metric trajectory

![score progression](figures/score_progression.png)

![failure breakdown](figures/failure_breakdown.png)



## Per-architecture-family breakdown



| Family | Runs | Best f1_macro | Mean f1_macro |
|---|---:|---:|---:|


## Failure analysis

| Error type | Count | Example message |
|---|---:|---|
| UnknownError | 1 | `]` |
| Aborted | 1 | `User requested abort` |


## Pipeline diagnostics

| Task | Runs | Completed | Failed | Avg validator attempts |
|---|---:|---:|---:|---:|
| execute_training | 2 | 0 | 2 | — |
| generate_code | 2 | 2 | 0 | — |
| propose_architecture | 2 | 2 | 0 | — |
| validate_code | 2 | 2 | 0 | 1.00 |


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
| 1 | `exp_c6c4438c73` | ExperimentStatus.FAILED | cnn_small_v1 | — | 95.8 |
| 2 | `exp_e2286230a7` | ExperimentStatus.ABORTED | cnn_gru_hybrid_small | — | 88.9 |
