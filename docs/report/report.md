---
title: "Autonomous Research Agent for BirdCLEF+ 2026"
author:
  - Trumbly
  - danish-m-qureshi
  - Lorry171717
  - SebastianMis23
date: "May 2026"
---

# Abstract

<!-- 4 lines: what we built, Track B, headline Kaggle public-LB result, key insight. Filled in Task 7. -->

# 1. Agent architecture

The agent runs a single closed loop: an outer **study** iterates **experiments**, each of which moves through a fixed seven-stage pipeline — *Propose, Generate, Validate, Execute, Capture, Judge*, plus an opportunistic *Recover* branch on failure (Figure 1). The LLM owns three stages (Propose, Generate, Judge) and is invoked again only when Recover fires; the training loop, the data loaders, the optimiser, metric computation, and result serialisation are fixed Python that the LLM never edits. The outer loop in `lab/core/lifecycle.py:StudyRunner.run` stops when one of three conditions trips: the wallclock deadline (`max_wallclock_minutes`), the experiment cap (`max_experiments`), or a per-experiment `abort_study` verdict from the judge. The inner pipeline lives in `lab/core/experiment.py:run_experiment`.

![Figure 1. Autonomous research agent loop. The outer study loop iterates experiments under a deadline plus max-experiments budget; each experiment runs the inner pipeline Propose -> Generate -> Validate -> Execute -> Capture -> Judge. Memory (top-K wins + recent failures) feeds into Propose and the recovery re-prompt. Recovery activates on Validate or soft Execute failures (auto-fix, otherwise LLM re-prompt). The Jinja2 skeleton fixes everything outside `build_model`, bounding the LLM's failure surface.](figures/agent_loop.png)

**LLM interface.** Every call to a model goes through `LLMClient.chat(messages) -> str` in `lab/core/llm.py`. One method, three back-ends: a default Ollama endpoint hosting `gemma4:e4b` on the host machine, plus OpenAI- and Anthropic-compatible HTTP endpoints routed through the same code path. The implementation uses only `urllib` from the standard library, so swapping providers is a YAML change rather than a dependency change. Failures are split into two classes. Transient errors — HTTP 429, 5xx, network timeouts — raise `LLMTransientError` and are retried with exponential backoff up to `retry_attempts`. Permanent errors — 4xx that is not 429, malformed JSON, unknown provider — raise `LLMPermanentError` and fail the surrounding task immediately. Each successful call snapshots its token counts and round-trip time so the dashboard can attribute cost per stage.

**Prompt management.** Every prompt is a versioned YAML file under `config/prompts/{propose_architecture, generate_code, recover_from_error, judge_experiment, judge_study, analyze_result}/vN.yaml`, indexed by `config/prompts/_registry.yaml`, which maps each task to a single active version. The pointer file is the only mutable surface; the version files themselves are immutable once pinned (ADR-004). To change a prompt's wording, the contributor writes a new `vN+1.yaml` and bumps the registry — old versions stay on disk so historical studies remain reproducible. Slots are filled per-prompt at render time by `PromptEngine`: the proposal prompt receives `{task_description}`, `{num_classes}`, `{input_tensor_shape}`, `{eda_summary}`, `{experiment_memory}`, `{personality}`, `{valid_architecture_families}`, and `{code_skeleton_content}`; the codegen prompt receives `{task_description}`, `{num_classes}`, `{input_tensor_shape}`, the rendered `{proposal}`, and `{code_skeleton_content}`; the recovery prompt receives `{broken_code}`, `{error_type}`, `{error_traceback}`, and `{experiment_memory}` so the LLM can learn from prior failures. Every study records the exact prompt paths it used in `Study.prompt_template_paths`.

**Memory.** `lab/core/memory.py` keeps a sliding window of the top-K successful experiments (sorted by `primary_score`) and the N most recent failures, capped at roughly 3 kB of markdown when rendered (ADR-006). Each win contributes a short block — architecture name, family, hyperparameters, a one-line training curve (`loss 0.5 -> 0.3 | macro_auc 0.21 -> 0.42 -> 0.61 | trend improving`), the judge's rationale, and a pointer to its checkpoint when one exists; each failure contributes one line with the error type and message. The whole blob is spliced into the `{experiment_memory}` slot of every propose and recover prompt. Keeping it small is deliberate: it forces the LLM to look at the most informative experiments, it keeps per-call tokens cheap, and it fits inside the modest context window of the local gemma model alongside the skeleton and the task description. Cross-study memory is opt-in via the `agent.memory_enabled` flag (ADR-007); the default is one study at a time.

**Decision policy — "what to try next."** The LLM is the policy. The agent does not encode a Bayesian optimiser, a bandit, or any explicit search algorithm; it shapes the LLM's output through three levers. First, **memory**: the top-K block is an exploitation signal (these architectures scored well, here are their curves), and the recent-failures block is an exclusion signal (do not propose this exact thing again). Second, **personality**: a single string slot — `exploratory` ("try architectures you haven't tried") or `conservative` ("tune the current best family") — toggled per study via `settings.agent.personality` (ADR-008). Third, **judge verdicts**: after every experiment, a dedicated LLM call (`lab/core/judge.py`) emits a `Verdict` in `{promote, keep, discard, abort_study}`; only `promote` and `keep` enter memory as wins, `discard` becomes a failure entry, and `abort_study` ends the run. The proposal schema additionally exposes `init_from_experiment_id` for transfer-learning-style warm starts and `continue_from_experiment_id` for resuming a still-improving run from its checkpoint.

**Error handling.** Recovery is a two-layer self-repair routine (ADR-005). When the validator (`lab/core/validator.py`) rejects a model — syntax error, forbidden import, missing `build_model` signature, hallucinated `nn.*` attribute, or a failing smoke forward pass — or when the subprocess executor (`lab/core/executor.py`) returns a soft failure (shape mismatch, value error, runtime error), `lab/core/recovery.py:Recovery.try_autofix` first matches the failure against a small set of deterministic rules: rename a typoed `nn.Conv2x2d` to `nn.Conv2d` using `difflib`-suggested replacements, inject a missing `num_classes` argument, and so on. When no autofix matches, the agent re-prompts the LLM with `recover_from_error/vN.yaml`, passing the original `build_model` block, the error type, and the traceback. The loop retries up to `max_recovery_attempts`. Hard failures — `Timeout`, `OOM`, `FileNotFound` — bypass recovery entirely and abort the experiment, because no surface-level patch can fix them.

**Constrained codegen — the key design choice.** The LLM only writes the `build_model(num_classes: int) -> nn.Module` block. The surrounding training script — data loaders, the optimiser, the loss, the epoch loop, macro-ROC-AUC computation, and the `results.json` writer — lives in a Jinja2 skeleton at `config/skeletons/audio_multilabel.py.j2` and is spliced around the generated block before execution (ADR-003). Two consequences follow. First, the **failure surface is bounded**: the LLM cannot accidentally break training plumbing, so failures are almost always architectural mistakes — bad layer shapes, hallucinated modules, channels in the wrong dimension — and these are exactly the failures the validator and the recovery layer can repair. Second, **experiments are comparable**: every run uses the same training budget, the same data pipeline, and the same metric implementation, so a 0.02 macro-ROC-AUC gap between two experiments reflects an architectural difference and not a difference in training mechanics. The trade-off is real: the agent cannot discover novel training tricks, only novel architectures within the families the skeleton's training loop can run.

# 2. Experiment log analysis

**Search breadth.** Across roughly a week of wallclock, the agent launched 20 studies and ran 61 individual experiments inside them. Only 8 of the 20 studies finished with status `COMPLETED`; the remaining 12 ended either because they hit the wallclock or experiment cap or because the per-experiment judge returned an `abort_study` verdict that the outer loop honoured. The high abort rate is by design and not a failure mode: the judge is configured to cut studies that stop improving so that compute can be redirected, and a study that is killed after two unproductive proposals is the desired behaviour rather than a regression. The 61 experiments that did launch are the unit of analysis for the rest of §2.

**Architectures tried.** Figure 2 shows the family-level distribution that the proposal LLM produced over those 61 experiments. The agent leant heavily on pretrained backbones: `efficientnet_pretrained` was proposed 34 times, more than every other family combined, with a handful of B0/B1/B2 variants and freezing strategies. The 18 `cnn_scratch` runs served as a cheap control — most appeared early in a new study before the LLM had a top-K win to imitate. The remaining nine experiments probed alternatives: three `perch_embedding`, two `birdnet_embedding`, two `mobilenet_pretrained`, and one `yamnet_feature`. Perch and BirdNET were the only domain-specific bird-audio embeddings the agent attempted as transfer-learning alternatives to ImageNet-pretrained CNNs.

![Figure 2. Distribution of architecture families across 61 experiments. EfficientNet-pretrained variants dominate (34/61); scratch CNNs (18/61) served as a cheap control; domain-specific audio embeddings (Perch, BirdNET, YAMNet) were probed as transfer-learning alternatives.](figures/family_distribution.png)

**What worked.** The headline number is the Kaggle public-LB score of **0.827**, achieved by `study_20260511_092457_n479` / `exp_0001`, an EfficientNet-B0 variant. The corresponding best private-LB submission, **0.817**, came from `study_20260510_174302_f6j2` / `exp_0002`, also an EfficientNet-B0 fine-tune-and-continue chain. Both winning experiments are pretrained EfficientNet-B0 backbones with dropout layers added inside `build_model`; both also chained from an earlier exp in the same study via the proposal schema's `init_from_experiment_id` field, which warm-starts a new experiment from a prior checkpoint and let the agent re-use a converged feature extractor instead of paying the cold-start cost on each iteration.

**What didn't.** The 18 `cnn_scratch` experiments plateaued well below the EfficientNet pack on internal validation — their best primary score sat around 0.54 macro-ROC-AUC, essentially chance on this metric, while the EfficientNet runs cleared 0.99 within a few epochs. The single `yamnet_feature` experiment was tried once and discarded by the judge before a second attempt. The deeper freeze strategies (`freeze4` and similar) underperformed lighter freezes inside the same study, suggesting that freezing too many layers on a small dataset removes more capacity than it stabilises. On the codegen side, recovery patched syntactic mistakes reliably — the LLM-typo `Conv2x2d` -> `Conv2d` was a recurring autofix — but deeper logic bugs (e.g. wrong channel dimension, missing pool) typically triggered a judge-driven abort rather than a successful re-prompt.

**Score progression.** Figure 3 plots the per-experiment internal val macro-ROC-AUC across the 61 runs (top panel) and the four Kaggle submissions (bottom panel). The top panel shows the agent converging on the EfficientNet family within the first dozen experiments and then sitting near the ceiling for the remainder of the run — the best val score, 0.998, comes from `study_20260507_200449_kzu4` / `exp_0002`. The bottom panel tells a sharper story: only 4 of the 61 experiments were ever submitted to Kaggle, and those four LB scores span 0.766 to 0.827 — far below the 0.998 val best. Two likely causes: (i) the val split is drawn from train clips that share recorder and location with their nearest neighbours in training, so the model is rewarded for memorising per-recorder cues that do not transfer; (ii) the hidden test set is soundscape audio where target species are rarer and noisier than the single-label train clips. Either way, the agent's iteration signal inside the loop is optimistic relative to the leaderboard. We revisit this gap in §5.

![Figure 3. Score progression. Top: per-experiment internal validation macro-ROC-AUC, chronological. Bottom: Kaggle leaderboard submissions (2 public, 2 private). The gap between best val (0.998) and best public-LB (0.827) reflects val/test distribution shift — see §5.](figures/score_progression.png)

**Memory ablation.** Only one study (`study_20260508_091552_zy0j`) was launched with `agent.memory_enabled = True` so that the propose prompt could see top-K wins from prior studies rather than only its own. It produced a single judged experiment at val 0.998 and was then aborted before submission, so there is no Kaggle score to compare. With one data point we cannot make a quantitative claim about cross-study memory; we report the run for completeness and flag it as anecdotal evidence, not an ablation.

# 3. Comparison with manual baselines

<!-- 1.5 pages. Filled in Task 3. -->

# 4. Reflections on course content

<!-- 1.5 pages. Filled in Task 4. -->

# 5. Limitations and future work

<!-- 1.0 pages. Filled in Task 5. -->

# 6. Individual contributions

<!-- 0.5 pages. Filled in Task 6. -->

# References

<!-- Filled in Task 7. -->
