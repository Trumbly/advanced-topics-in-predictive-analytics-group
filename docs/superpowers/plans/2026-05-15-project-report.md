# Project Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the 10-page Advanced Predictive Analytics course project report (deliverable D4) covering the autonomous research agent built for BirdCLEF+ 2026, with every claim grounded in concrete artefacts under `experiments/studies/`, `lab/`, and `config/`.

**Architecture:** Six PDF-mandated sections plus title block, abstract, and references. Page budget allocated section by section so the team cannot drift past 10 pages. Each section maps to a task; each task lists the source files, the figures to embed, the headline numbers to cite, and the word budget. No prose is written without first locking down the data points.

**Tech Stack:** Markdown source (`docs/report/report.md`) converted to `.docx` via Pandoc — keeps source diff-friendly in git while producing the Word deliverable the team agreed to submit. Figures as `.png` under `docs/report/figures/`. References inline as numbered citations (Pandoc-compatible bracketed style). Final Word file: `docs/report/report.docx` (generated, gitignored after first sanity pass).

**Conversion command:**
```bash
pandoc docs/report/report.md \
  -o docs/report/report.docx \
  --reference-doc=docs/report/reference.docx  # optional, only if team supplies a style template
```

**Decisions locked in (do not re-litigate during execution):**
- Branch: `docs/project-report` (off current branch `feat/per-experiment-submission`, PR against `production`).
- Format: Markdown source → Word via Pandoc.
- Baselines (Task 3): priors baseline only.
- Individual contributions (Task 6): draft from `git blame` + `docs/TEAM_PLAN.md`; circulate after.
- Video link (Task 7): leave as TODO placeholder; team confirms whether they will record one.
- Primary score everywhere: **Kaggle public-LB ROC-AUC**, not val. Best public-LB **0.827** (study `study_20260511_092457_n479`, exp_0001). Val numbers are an internal diagnostic only and must be flagged as such (val/test gap suggests leakage or distribution shift — say so honestly in §5).

**Hard constraints from PDF (`Project_Handout.pdf`):**
- Max 10 pages.
- Must include an architecture diagram in section 1.
- Must cover the six listed topics, in this order.
- Honest discussion of limitations is part of the grading rubric (25%).
- Video link must be embedded in the report.

---

## Sources of truth (read these before writing prose)

| Topic | File(s) |
|---|---|
| Agent loop | `lab/core/lifecycle.py`, `lab/core/experiment.py` |
| Codegen + validation + recovery | `lab/core/{validator,executor,recovery,judge}.py` |
| LLM client + prompts | `lab/core/llm.py`, `config/prompts/**/*.yaml`, `config/prompts/_registry.yaml` |
| Memory | `lab/core/memory.py` |
| Skeleton (fixed scaffold) | `config/skeletons/audio_multilabel.py.j2`, `audio_embedding_multilabel.py.j2` |
| Task contract | `lab/tasks/track_b_birdclef.py` |
| Submission builder | `lab/submission/builder.py` |
| Experiment results | `experiments/studies/study_*/study.json`, `report/`, `run.log.jsonl` |
| Architectural rationale | `docs/ARCHITECTURE.md` (ADRs 001-014) |
| Rewrite blueprint | `docs/REDESIGN_PLAN.md` |
| Production gap analysis | `docs/PRODUCTION_REVIEW.md` |
| Issue ownership (for §6) | `docs/TEAM_PLAN.md`, `docs/issues/` |

## Page budget (hard cap 10)

| Section | Pages | Why this size |
|---|---|---|
| Title block + abstract | 0.5 | Course header + 4-line abstract |
| 1. Agent architecture (+ diagram) | 2.5 | 40% rubric weight — the report's centerpiece |
| 2. Experiment log analysis | 2.0 | Needs 2-3 figures + a results table |
| 3. Comparison with manual baselines | 1.5 | One table, short narrative |
| 4. Reflections on course content | 1.5 | Map agent decisions to course chapters |
| 5. Limitations & future work | 1.0 | Rubric explicitly rewards honesty |
| 6. Individual contributions | 0.5 | One paragraph per teammate |
| References + video link | 0.5 | BibTeX entries + YouTube URL |
| **Total** | **10.0** | |

If a section overruns, cut from §1 prose first (the diagram carries most of the weight) and §4 second; never cut §5.

---

### Task 0: Set up the report skeleton

**Files:**
- Create: `docs/report/report.md` (Markdown source — single file, sectioned)
- Create: `docs/report/figures/.gitkeep` (figures directory)
- Create: `docs/report/Makefile` (one-liner to build the `.docx`)

- [ ] **Step 1: Create the Markdown skeleton**

```markdown
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

<!-- 2.5 pages. Filled in Task 1. -->

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
```

- [ ] **Step 2: Create the Makefile**

```makefile
# docs/report/Makefile
.PHONY: docx clean
docx: report.docx
report.docx: report.md
	pandoc report.md -o report.docx
clean:
	rm -f report.docx
```

- [ ] **Step 3: Verify Pandoc is installed and the build works**

```bash
which pandoc || brew install pandoc       # macOS
cd docs/report && make docx
open report.docx                          # quick visual sanity check
```

Expected: empty Word file with all six section headings, table of contents implicit.

- [ ] **Step 4: Commit the scaffold (Markdown source only; the `.docx` is a build artifact)**

Add to repo `.gitignore`:
```
docs/report/report.docx
```

Then:
```bash
git add docs/report/report.md docs/report/Makefile docs/report/figures/.gitkeep .gitignore
git commit -m "docs(report): scaffold the project report"
```

---

### Task 1: Section 1 — Agent architecture (2.5 pages, with diagram)

**PDF requirement:** *"How does the loop work? How do you prompt the LLM? How does the agent decide what to try next? Include a diagram."*

**Files:**
- Create: `docs/report/figures/agent_loop.png` (the diagram, built in draw.io / excalidraw / mermaid → PNG export)
- Modify: `docs/report/report.md` — fill `# 1. Agent architecture`

**Inputs to read first:**
- `lab/core/lifecycle.py:StudyRunner.run()` — the outer loop
- `lab/core/experiment.py:run_experiment()` — the inner stages (propose → generate → validate → execute → judge)
- `lab/core/llm.py:LLMClient.chat()` — provider abstraction
- `config/prompts/_registry.yaml` — versioned prompts
- `lab/core/memory.py` — ≤3 kB markdown memory
- `lab/core/judge.py` — per-experiment + per-study verdicts
- `lab/core/recovery.py` — two-layer self-repair
- `docs/ARCHITECTURE.md` — ADRs 001–014

- [ ] **Step 1: Draft the architecture diagram**

Diagram must show:
1. Outer `StudyRunner` loop with deadline + max-experiments stopping criteria.
2. Inner pipeline arrows: **Propose** (LLM) → **Generate code** (LLM, returns `build_model`) → **Validate** (AST + smoke forward pass) → **Execute** (sandbox subprocess) → **Capture** (`results.json`) → **Judge** (LLM, returns verdict).
3. Side channels: **Memory** (top-K + recent failures fed into Propose and Recover prompts) and **Recovery** (auto-fix → LLM re-prompt branch from Validate/Execute on failure).
4. Skeleton/template wraps Generate (LLM only writes `build_model`, not the whole training loop) — call out as a green box.

Save to `docs/report/figures/agent_loop.png` (300 DPI minimum, legible at print size, no 6 pt text).

- [ ] **Step 2: Write the architecture prose (≈900 words)**

Cover, in this order:

1. **Loop overview (1 paragraph + figure reference):** seven-stage pipeline per experiment; LLM owns architectural choices, skeleton owns the training plumbing. Cite `lab/core/experiment.py:run_experiment` lines 87–157.

2. **LLM interface (1 paragraph):** provider-agnostic `LLMClient.chat()`; default `ollama:gemma4:e4b`; transient/permanent error split with exponential backoff. Cite `lab/core/llm.py`.

3. **Prompt management (1 paragraph):** versioned YAML prompts under `config/prompts/{propose_architecture, generate_code, recover_from_error, judge_experiment, analyze_results}/vN.yaml`. Pointer file `_registry.yaml`. Slot-filled at render time with task, memory, EDA summary, personality. Each prompt is immutable once pinned (ADR-004). Cite the registry file.

4. **Memory (1 paragraph):** ≤3 kB markdown (top-K successes + recent failures) injected into every Propose and Recover prompt. Optional cross-study memory toggle (`agent.memory_enabled`, ADR-007). Trade-off: small budget keeps prompts cheap and forces the LLM to focus on the most informative experiments. Cite `lab/core/memory.py`.

5. **Decision policy — "what to try next" (1 paragraph):** the LLM is the policy. Memory provides exploitation signal (top-K), recent failures provide exclusion signal, and `personality` (`exploratory` vs `conservative`) biases the search. The Judge produces a per-experiment verdict ∈ {promote, keep, discard, abort_study} that gates whether the experiment enters memory and whether the study continues. Cite `lab/core/judge.py`.

6. **Error handling (1 paragraph):** two-layer self-repair. Deterministic auto-fix (e.g., `Conv2x2d` → `Conv2d`) tried first; on failure, LLM re-prompted with `recover_from_error/vN.yaml`. Hard errors (Timeout, OOM, FileNotFound) abort retries; soft errors (ValueError, ShapeMismatch) trigger recovery up to `max_recovery_attempts`. Cite `lab/core/recovery.py`, `lab/core/executor.py`.

7. **Constrained codegen (1 paragraph, key design choice):** the LLM only writes the `build_model` block; the surrounding training loop, data loaders, augmentation, and metric computation are fixed in a Jinja2 skeleton (`config/skeletons/audio_multilabel.py.j2`). This bounds the failure surface and makes experiments comparable. Cite ADR-003.

- [ ] **Step 3: Commit**

```bash
git add docs/report/report.md docs/report/figures/agent_loop.png
git commit -m "docs(report): write architecture section + agent loop diagram"
```

---

### Task 2: Section 2 — Experiment log analysis (2 pages)

**PDF requirement:** *"What architectures did the agent try? What worked and what didn't? Show the progression of scores across iterations."*

**Files:**
- Modify: `docs/report/report.md` — fill `# 2. Experiment log analysis`
- Create: `docs/report/figures/score_progression.png`
- Create: `docs/report/figures/family_distribution.png`
- Create: `scripts/report_metrics.py` — pulls headline numbers from `experiments/studies/*/study.json` (one source of truth, including `kaggle_scores`)

**Headline numbers to cite (already extracted; re-verify before final draft):**

| Metric | Value | Source |
|---|---|---|
| Studies run | 20 | `experiments/studies/` directory count |
| Studies completed (status=COMPLETED) | 8 | aggregate over `study.json` |
| Total experiments | 61 | aggregate |
| **Best Kaggle public-LB ROC-AUC** | **0.827** (study `study_20260511_092457_n479`, exp_0001) | `study.json:kaggle_scores` |
| Best Kaggle private-LB ROC-AUC | 0.817 (study `study_20260510_174302_f6j2`, exp_0002) | `study.json:kaggle_scores` |
| Best internal val macro-AUC (diagnostic only) | 0.998 (study `study_20260507_200449_kzu4`, exp_0002) | `study.json:best_score` |
| Distribution of architecture families | efficientnet_pretrained=34, cnn_scratch=18, perch_embedding=3, birdnet_embedding=2, mobilenet_pretrained=2, yamnet_feature=1 | aggregate over `proposal.family` |

**Important honesty note (mandatory in §2 prose):** the gap between best val macro-AUC (0.998) and best Kaggle public-LB (0.827) is large. Report Kaggle as the headline; explain the gap (likely val-set leakage or distribution shift from soundscapes vs single-label clips); revisit in §5 limitations.

- [ ] **Step 1: Write the aggregation script**

```python
# scripts/report_metrics.py
"""Re-derive every number cited in §2 + §3 of the report. Run before final draft."""
import json
from collections import Counter
from pathlib import Path

base = Path("experiments/studies")
fams: Counter[str] = Counter()
rows: list[dict] = []
kaggle: list[dict] = []
for d in sorted(base.iterdir()):
    f = d / "study.json"
    if not f.exists():
        continue
    s = json.loads(f.read_text())
    rows.append({
        "id": s.get("id", d.name),
        "status": s.get("status"),
        "n_exp": len(s.get("experiments") or []),
        "best_val": s.get("best_score"),
    })
    for e in s.get("experiments") or []:
        prop = e.get("proposal") or {}
        fams[prop.get("family") or prop.get("architecture_name") or "?"] += 1
    for ks in s.get("kaggle_scores") or []:
        kaggle.append({"study": s.get("id", d.name), **ks})

best_val = max((r["best_val"] for r in rows if r["best_val"] is not None), default=None)
public = [k for k in kaggle if k.get("leaderboard") == "public"]
private = [k for k in kaggle if k.get("leaderboard") == "private"]
best_public = max(public, key=lambda k: k["score"], default=None)
best_private = max(private, key=lambda k: k["score"], default=None)

print(f"studies={len(rows)} completed={sum(1 for r in rows if r['status']=='COMPLETED')}")
print(f"experiments={sum(r['n_exp'] for r in rows)} best_val_macro_auc={best_val:.4f}")
if best_public:
    print(f"best_kaggle_public={best_public['score']} (study={best_public['study']} exp={best_public['experiment_id']})")
if best_private:
    print(f"best_kaggle_private={best_private['score']} (study={best_private['study']} exp={best_private['experiment_id']})")
print("\nFamily distribution:")
for k, v in fams.most_common():
    print(f"  {v:3d}  {k}")
print("\nAll Kaggle scores:")
for k in sorted(kaggle, key=lambda x: -x["score"]):
    print(f"  {k['score']:.3f} {k['leaderboard']:8s} {k['study']}/{k['experiment_id']}")
```

Run: `python scripts/report_metrics.py` and paste output into the report. Re-run before submitting.

- [ ] **Step 2: Build the score-progression figure**

Two-panel matplotlib figure:
- Top panel: per-experiment internal val macro-AUC across all 61 experiments, x = chronological experiment index, y = val score, colored by family.
- Bottom panel: the 4 Kaggle submission scores plotted at their `recorded_at` timestamp on a shared x-axis (or alongside the experiment index they correspond to). Highlight best public-LB.

Save to `docs/report/figures/score_progression.png` (300 DPI). Use matplotlib only.

- [ ] **Step 3: Build the family-distribution figure**

Horizontal bar chart of how many experiments each architecture family received, ordered descending. Save to `docs/report/figures/family_distribution.png` (300 DPI).

- [ ] **Step 4: Write the prose (≈700 words)**

Sub-structure:

1. **Search breadth (1 paragraph):** 20 studies / 61 experiments / 8 completed (the rest aborted on budget or judge verdict). The high abort rate is by design — the Judge cuts unproductive studies early.

2. **Architectures tried (1 paragraph + figure ref):** dominant exploitation of `efficientnet_pretrained` (34 of 61) reflects strong early-iteration signal from ImageNet-pretrained CNNs on mel-spectrograms; `cnn_scratch` (18) used as a control / cheap baseline; `perch_embedding`, `birdnet_embedding`, `mobilenet_pretrained`, and `yamnet_feature` explored as transfer-learning alternatives.

3. **What worked (1 paragraph):** EfficientNet-B0 variants drove the top Kaggle scores. Best **public-LB 0.827** from `study_20260511_092457_n479`/exp_0001; best private-LB 0.817 from `study_20260510_174302_f6j2`/exp_0002. Warm-starting from prior best (`Proposal.init_from_experiment_id`) shortened the per-experiment training time.

4. **What didn't (1 paragraph):** scratch CNNs plateaued well below the EfficientNet variants on val; YAMNet single experiment was abandoned by the judge; very aggressive backbone freezing (`freeze4`) hurt more than it helped. Recovery succeeded on most syntactic mistakes (typos like `Conv2x2d`) and on common shape-mismatch errors; deeper logic bugs typically led to judge-driven aborts rather than further retries.

5. **Score progression (1 paragraph + figure ref):** Figure 2 shows monotonically improving best-so-far on val across the run, with the agent settling on EfficientNet-B0 by iteration ~5. The Kaggle scores tell a different story (Figure 2 bottom panel): public-LB ranges only 0.766–0.827, while val reaches 0.998. **Lead with this gap.** Most likely causes: (i) val set is constructed from train clips that share recorder / location with train, so models memorize per-recorder cues; (ii) the test set covers soundscape windows where target species are rarer and noisier than single-label clips. Implication: the agent's iteration signal is optimistic; revisit in §5.

6. **Memory ablation (1 sentence):** only one study (`study_20260508_091552_zy0j`) used cross-study memory. Not enough data for a quantitative claim; treat as anecdotal.

- [ ] **Step 5: Commit**

```bash
git add docs/report/report.md docs/report/figures/score_progression.png docs/report/figures/family_distribution.png scripts/report_metrics.py
git commit -m "docs(report): write experiment-log analysis section"
```

---

### Task 3: Section 3 — Comparison with manual baselines (1.5 pages)

**PDF requirement:** *"How do the agent's best models compare to baselines you built manually?"*

**Files:**
- Modify: `docs/report/report.md` — fill `# 3. Comparison with manual baselines`
- Create: `scripts/baseline_priors.py`

**Scope (locked in):** priors baseline only.

- [ ] **Step 1: Implement the priors baseline**

```python
# scripts/baseline_priors.py
"""Predict per-class priors for every window. Reports ROC-AUC macro on val.

Reproduces the trivial baseline cited in §3 of the report. Re-run before
final draft if the val index changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = Path("data/processed/labels.csv")
VAL_INDEX = Path("data/processed/val_index.json")

def main() -> None:
    labels = pd.read_csv(LABELS)
    val_index = json.loads(VAL_INDEX.read_text())
    val_ids = {row["id"] for row in val_index}
    train = labels[~labels["id"].isin(val_ids)]
    val = labels[labels["id"].isin(val_ids)]
    class_cols = [c for c in labels.columns if c not in {"id", "filename"}]
    priors = train[class_cols].mean().values  # shape (234,)
    y_true = val[class_cols].values
    y_score = np.broadcast_to(priors, y_true.shape)
    aucs = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() == 0:
            continue
        aucs.append(roc_auc_score(y_true[:, j], y_score[:, j]))
    print(f"priors_baseline_macro_auc = {np.mean(aucs):.4f}  (n_classes_scored={len(aucs)})")

if __name__ == "__main__":
    main()
```

Run:
```bash
python scripts/baseline_priors.py
```

Expected output: a single line `priors_baseline_macro_auc = 0.50…` (broadcast priors give exactly 0.5 per class because the score has no per-sample variation; macro-AUC therefore equals 0.5 by definition). Record the exact number in the table below.

- [ ] **Step 2: Build the comparison table (Markdown)**

```markdown
| Model | Author | Score (macro ROC-AUC) | Source |
|---|---|---|---|
| Class priors | Manual | 0.50 (val) | `scripts/baseline_priors.py` |
| EfficientNet-B0 (best agent, internal val) | Agent | 0.998 | `study_20260507_200449_kzu4/exp_0002` |
| EfficientNet-B0 (best agent, Kaggle public-LB) | Agent | **0.827** | `study_20260511_092457_n479/exp_0001` |
| EfficientNet-B0 (best agent, Kaggle private-LB) | Agent | 0.817 | `study_20260510_174302_f6j2/exp_0002` |
```

- [ ] **Step 3: Write the prose (≈400 words)**

Cover, in this order:
1. **What baseline (1 paragraph):** uniform class-prior predictor; intentionally trivial to anchor the lower bound. Why not a logreg-on-mels baseline: out of scope given the deadline.
2. **Gap (1 paragraph):** agent's Kaggle public-LB (0.827) is ~33 points above the priors baseline (0.50). Most of that lift comes from ImageNet transfer + SpecAugment, neither of which the priors baseline can exploit.
3. **Internal vs leaderboard (1 paragraph):** 0.998 val vs 0.827 public-LB. Explain the gap honestly (same root cause as in §2). Lead with the leaderboard number when comparing to anything external.
4. **What this comparison does and doesn't tell us (1 paragraph):** it confirms the agent is doing real work (well above random), but does not validate the agent's iteration loop against a strong manual baseline. Flag as a limitation; revisit in §5.

- [ ] **Step 4: Commit**

```bash
git add docs/report/report.md scripts/baseline_priors.py
git commit -m "docs(report): write baseline comparison section + priors baseline"
```

---

### Task 4: Section 4 — Reflections on course content (1.5 pages)

**PDF requirement:** *"Which deep learning techniques from the course did the agent use? Which ones proved most effective for this task and why?"*

**Files:**
- Modify: `docs/report/report.tex` — fill `\section{Reflections on course content}`

**Course chapters → agent techniques mapping (verified against `config/skeletons/audio_multilabel.py.j2` AND sampled generated `build_model` code under `sandbox/exp_*/kernel/code.py`):**

| Course chapter | Technique | Where in the agent | Verified |
|---|---|---|---|
| Signal preprocessing | Mel-spectrogram (sr=32k, n_mels=128, hop=512, dB) | `lab/tasks/audio_mels.py:MelParams` | yes |
| CNNs (2D) | Conv2D + pooling stacks | `cnn_scratch` family proposals | yes |
| Transfer learning | ImageNet-pretrained EfficientNet / MobileNet / ResNet | `efficientnet_pretrained`, `mobilenet_pretrained` families | yes |
| Pretrained domain models | YAMNet, BirdNET, Perch audio embeddings | `yamnet_feature`, `birdnet_embedding`, `perch_embedding` families | yes |
| Regularization (dropout) | `nn.Dropout` layers inserted by LLM in `build_model` | LLM-generated code (most experiments have 2-4 dropout layers) | yes |
| Regularization (freezing) | Selectively requires_grad=False on backbone stages | LLM-generated code | yes |
| Optimization (LR schedules) | Constant / cosine / one-cycle via env knobs | skeleton + `AGENT_LR_SCHEDULE` | yes |
| Multi-label heads | `BCEWithLogitsLoss`, sigmoid output, macro-ROC-AUC | skeleton `train_one_epoch` + metrics | yes |

**Honest accounting (key correction):**
- **The skeleton contains NO data augmentation.** `grep -c "augment" config/skeletons/audio_multilabel.py.j2 = 0`.
- **SpecAugment appears only in 5 of 61 generated `build_model` blocks** (the LLM occasionally adds `MaskAlongAxis` / `FrequencyMask` / `TimeMask` inside its model code). It is not part of the agent's standard training recipe.
- Several experiment **names** contain `specaug` (e.g., `efficientnet_b0_specaug_dropout03`), but the corresponding code typically does NOT implement SpecAugment — the suffix is aspirational LLM naming. **Do not cite SpecAugment as a course-content technique exercised by the agent; cite dropout-based regularization instead.**

- [ ] **Step 1: Re-verify the mapping**

Read `config/skeletons/audio_multilabel.py.j2` and `config/skeletons/audio_embedding_multilabel.py.j2` end to end. Sample 3-5 successful generated `build_model` blocks from `sandbox/exp_*/kernel/code.py`. Confirm every row in the table above is actually wired up. If a row is *not* implemented anywhere (skeleton OR sample of generated code), cross it out — the report must reflect what the agent really did, not what proposal names imply.

- [ ] **Step 2: Write the prose (≈600 words)**

Three paragraphs:

1. **Coverage paragraph:** the agent exercised every major chapter listed in the table. Reference Figure 2 (family distribution) — the search prioritized transfer learning over scratch CNNs, which mirrors the course's emphasis on pretrained backbones for limited-data audio problems.

2. **What proved most effective and why:** ImageNet-pretrained EfficientNet-B0 dominated. Three reasons: (i) 128×313 mels are visually similar to ImageNet inputs in low-frequency texture, so feature transfer is non-trivial but works; (ii) the small effective dataset (~233 k single-label + 739 soundscape windows for 234 classes) means random initialization underfits — pretraining is the cheapest source of inductive bias; (iii) regularization via dropout layers added inside `build_model` plus selective backbone freezing prevented the pretrained features from being overwritten on the small dataset.

3. **What course topics were not exercised:** RNNs and Transformers were absent — the chosen task is 2D image-like, not sequential, so the agent never proposed them. This is a deliberate restriction of the search space, not an oversight; if the team had chosen Track A (Disaster Tweets), the family list would shift accordingly.

- [ ] **Step 3: Commit**

```bash
git add docs/report/report.md
git commit -m "docs(report): write course-content reflections section"
```

---

### Task 5: Section 5 — Limitations and future work (1 page)

**PDF requirement:** *"What could the agent not do? What would you improve?"*

**Files:**
- Modify: `docs/report/report.md` — fill `# 5. Limitations and future work`

**Limitations to cover (grounded in code, not speculation):**

1. **Val/leaderboard gap** — internal val macro-AUC 0.998 vs Kaggle public-LB 0.827. Strongly suggests val-set leakage (per-recorder or per-location correlation between train and val) and/or distribution shift between single-label clips and the soundscape test windows. The agent's iteration signal is optimistic. **This is the most important limitation; lead with it.**
2. **CPU-only execution** — `lab/core/executor.py` runs experiments locally; no Kaggle/Modal/GPU dispatch (ADR-010). Limits the depth of each experiment.
3. **No Kaggle auto-submit** — submission notebooks built (`lab/submission/builder.py`) but uploaded by hand. README §"Video plan" calls this out explicitly.
4. **Only 4 leaderboard submissions across 20 studies** — most studies never reached Kaggle, so the agent's selection signal during iteration was dominated by the (leaky) val score.
5. **No automated EDA generation** — the EDA summary fed into prompts is precomputed once per task; the agent never re-runs EDA.
6. **Single Track (B only)** — no Track A adapter implemented. The task-adapter ABC supports it but no team chose to build it.
7. **Cross-study memory is opt-in and barely tested** — `agent.memory_enabled` defaults to off (ADR-007). Only one study (`study_20260508_091552_zy0j`) used it.
8. **Recovery has known false-negatives** — the judge previously treated recovered codegen failures as terminal study errors. Fixed in commit `65e6f6f` ("ignore recovered codegen failures when judging"), but earlier studies pre-dating this fix are biased toward early aborts.
9. **Small dataset for some classes** — 28 species absent from `train.csv`; only present in 739 soundscape windows. The agent has no class-rebalancing strategy.
10. **Compute budget is per-experiment, not per-study** — the agent cannot "save" compute on a cheap experiment to spend on an expensive one later.

**Future work items (each one a paragraph, not a list):**
- **Recorder-aware val split** to close the val/leaderboard gap: group-K-fold by `recorder_id` so the val score is no longer optimistic.
- GPU executor adapter behind the existing `LocalExecutor` interface (Kaggle Notebook API or Modal).
- Continuous Kaggle submission: auto-submit the best experiment of every study so the agent gets real-test feedback in the loop, not only val.
- Active class-balancing inside the skeleton (oversampling rare classes during training).
- Auto-EDA stage: an LLM call that re-derives the dataset summary at the start of every study.

- [ ] **Step 1: Write the prose (≈500 words)**

Open with the val/leaderboard gap as the headline limitation, then one sentence per remaining limitation, then a paragraph each for the top three future-work items (recorder-aware split, GPU executor, auto-submission). Be specific. Avoid "we would improve everything" — name the file or interface that would change.

- [ ] **Step 2: Commit**

```bash
git add docs/report/report.md
git commit -m "docs(report): write limitations and future-work section"
```

---

### Task 6: Section 6 — Individual contributions (0.5 page)

**PDF requirement:** *"Brief description of each team member's role."*

**Files:**
- Modify: `docs/report/report.md` — fill `# 6. Individual contributions`
- Create: `scripts/contributions_from_blame.py` (extracts per-author file ownership)

**Input source:** `docs/TEAM_PLAN.md` for the original issue assignment + `git blame` / `git log --author=<name>` for actual delivered files.

- [ ] **Step 1: Build a per-author file-ownership summary**

```python
# scripts/contributions_from_blame.py
"""Summarise per-author file ownership across lab/, config/, scripts/, tests/.

Prints, for each known author, the top N files where they wrote the most
lines (per git blame). Used to seed §6 of the report; teammates still confirm
their own paragraph.
"""
from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

AUTHORS = ["Trumbly", "danish-m-qureshi", "Lorry171717", "SebastianMis23"]
ROOTS = ["lab", "config", "scripts", "tests", "docs"]
TOP_N = 8

def main() -> None:
    files = []
    for root in ROOTS:
        files += [str(p) for p in Path(root).rglob("*") if p.is_file()]
    per_author: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in files:
        try:
            out = subprocess.run(
                ["git", "blame", "--line-porcelain", f],
                capture_output=True, text=True, check=False,
            ).stdout
        except Exception:
            continue
        for line in out.splitlines():
            if line.startswith("author "):
                a = line[len("author "):].strip()
                if a in AUTHORS:
                    per_author[a][f] += 1
    for a in AUTHORS:
        print(f"\n## {a}")
        ranked = sorted(per_author[a].items(), key=lambda kv: -kv[1])[:TOP_N]
        for f, n in ranked:
            print(f"  {n:5d}  {f}")

if __name__ == "__main__":
    main()
```

Run:
```bash
python scripts/contributions_from_blame.py > docs/report/contributions_raw.txt
```

Inspect the output. Use it to draft each teammate's paragraph.

- [ ] **Step 2: Draft one paragraph per team member**

Template per teammate (replace bracketed parts with concrete files / issue numbers from `docs/TEAM_PLAN.md` and the blame output):

> **`<name>`** — owned `[component, e.g., executor sandbox + recovery]` (`lab/core/executor.py`, `lab/core/recovery.py`; issues `I-XX`, `I-YY`). Also contributed `[secondary thing]` and `[review responsibilities]`.

Four paragraphs, ~50 words each. No marketing language. **Mark the section as DRAFT (one line at the top of §6) until each teammate confirms.**

- [ ] **Step 3: Circulate for confirmation**

Send the draft to all four teammates; require sign-off before final commit (`git blame` shows what was touched, not why or how — only the author can write that). The DRAFT marker stays in the source until removed in Task 8 final pass.

- [ ] **Step 4: Commit (with DRAFT marker still present)**

```bash
git add docs/report/report.md scripts/contributions_from_blame.py
git commit -m "docs(report): draft individual contributions section (awaiting team sign-off)"
```

---

### Task 7: Title block, abstract, references, video link (0.5 page combined)

**Files:**
- Modify: `docs/report/report.md` — fill abstract + references + video link placeholder

- [ ] **Step 1: Write the abstract (4 lines, ≈100 words)**

Cover: (a) what we built (autonomous research agent driven by a local LLM), (b) Track B / BirdCLEF+ 2026, (c) headline result (best Kaggle public-LB ROC-AUC 0.827; internal val 0.998 reported as a diagnostic), (d) one-sentence key insight ("constraining LLM codegen to a `build_model` block bounded the failure surface and made the search comparable across experiments").

- [ ] **Step 2: Add references**

Minimum (Markdown-friendly numbered list at the end of the document):
1. Chollet, F. *Deep Learning with Python*, 2nd ed. — course reference book.
2. Kaplan et al., 2020 — *Scaling Laws for Neural Language Models* (arXiv:2001.08361).
3. Hoffmann et al., 2022 — *Training Compute-Optimal Large Language Models* (arXiv:2203.15556).
4. Park et al., 2019 — *SpecAugment* (arXiv:1904.08779).
5. Tan & Le, 2019 — *EfficientNet* (arXiv:1905.11946).
6. BirdCLEF+ 2026 Kaggle competition — `https://www.kaggle.com/competitions/birdclef-2026` (retrieved 2026-05-15).
7. Ollama — `https://ollama.com`.

- [ ] **Step 3: Embed the video link as a placeholder**

Add a footnote-style line directly under the abstract:

```markdown
**Video presentation (D3):** TODO — to be replaced with the unlisted YouTube URL before submission. Source file: `videos/demo.mp4` in the repository.
```

The DRAFT marker stays until Task 8 final pass replaces it with the real URL. **Do not commit a final report with a `TODO` here.**

- [ ] **Step 4: Commit**

```bash
git add docs/report/report.md
git commit -m "docs(report): abstract, references, video link placeholder"
```

---

### Task 8: Final pass — fact check, page count, self-review

**Files:**
- Read all: `docs/report/report.md`

- [ ] **Step 1: Re-run `scripts/report_metrics.py` and `scripts/baseline_priors.py`, reconcile every number in the report**

If any number changed (new studies added, new Kaggle submissions logged, etc.) update the report. **Every score, count, and study-id in the report must match the file system at commit time.**

- [ ] **Step 2: Build the docx and check page count**

```bash
cd docs/report && make docx
open report.docx     # macOS; on Linux: xdg-open report.docx
```

Visually check page count in Word. Must be ≤ 10. If over, cut from §1 then §4 prose. **Do not cut §5 — the rubric rewards honesty about limitations.**

- [ ] **Step 3: Replace DRAFT markers + TODOs**

- §6 individual contributions: remove the DRAFT marker once all four teammates have signed off.
- Abstract footnote: replace the video TODO with the real YouTube URL.
- Anywhere else `TODO` / `TBD` appears: replace or remove. **Search before commit:** `grep -E '\b(TODO|TBD|XXX|DRAFT)\b' docs/report/report.md` must return empty.

- [ ] **Step 4: Self-review checklist**

- [ ] Every PDF requirement (six topics + diagram) has a corresponding section that actually addresses it.
- [ ] No `TBD`, `TODO`, `XXX`, `DRAFT`, or placeholder language anywhere in `report.md`.
- [ ] Every cited study-id exists under `experiments/studies/`.
- [ ] Every cited Kaggle score is reproducible from `study.json` `kaggle_scores`.
- [ ] Every cited val score is reproducible from `study.json` `best_score` and flagged as internal-diagnostic-only.
- [ ] Every file path mentioned in the report exists in the repo.
- [ ] Architecture diagram is legible at print size (no 6 pt text) and embedded as a `.png`.
- [ ] Word page count ≤ 10.
- [ ] Video link resolves (open in a private window to confirm).
- [ ] Individual contributions section has explicit sign-off from each teammate (no DRAFT marker).

- [ ] **Step 5: Commit final**

```bash
cd docs/report && make docx
git add docs/report/report.md
# report.docx is gitignored; the build artefact is regenerated on the reviewer's machine.
git commit -m "docs(report): final draft for submission"
```

---

## Self-review of this plan

**Spec coverage:** every PDF-required section (1–6) is its own task; the diagram is a separate step inside Task 1; the video link has a step in Task 7; the run-out-of-the-box requirement is *not* this report's job — it lives in the README. Confirmed.

**Placeholder scan:** the only `TBD` left intentionally is the logreg baseline score in Task 3 Step 4 — flagged with an explicit "do not leave TBD" note. Otherwise no placeholders.

**Type consistency:** "study_id", "primary_score", "macro-AUC", and "family" are used consistently and match the `study.json` schema. The architecture-family names match the actual values in `proposal.family` from the corpus.

**Open decisions before execution starts:**
- Task 3 needs the team to decide which manual baselines to implement before writing the section. Block on this.
- Task 6 needs each teammate's sign-off on their contribution paragraph.
- Task 7 needs the video URL — block §7 final commit on the upload.
