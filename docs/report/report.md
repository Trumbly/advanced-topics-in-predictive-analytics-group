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

<!-- 2.0 pages. Filled in Task 2. -->

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
