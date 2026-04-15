# Ui study start test

_Task: **track_b** · Primary metric: **f1_macro** · Status: **StudyStatus.ABORTED**_


## Executive summary

**Executive Summary**

No model achieved a successful result on the Track B task, with the best macro F1 score standing at 0.0000 across the single experiment attempted, which failed during execution. The most surprising takeaway is that even a single baseline experiment could not complete successfully, suggesting fundamental issues with the data pipeline, environment configuration, or task formulation that must be resolved before any meaningful modeling can begin. As a concrete next step, I recommend running a minimal diagnostic experiment—loading the dataset, verifying label distributions, and fitting a simple majority-class or logistic regression baseline—to isolate whether the failure is infrastructural or methodological.


## Overview

| | |
|---|---|
| Study ID | `study_20260414_155802_3be1` |
| Started | 2026-04-14 15:58:02.308133+00:00 |
| Finished | 2026-04-14 15:58:28.730375+00:00 |
| Experiments | 1 (0 succeeded, 1 failed) |
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
| Aborted | 1 | `User requested abort` |


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
| 1 | `exp_431057453a` | ExperimentStatus.ABORTED | [efficientnet_b0] efficientnet_b0_mixup_focal | — | 0.0 |
