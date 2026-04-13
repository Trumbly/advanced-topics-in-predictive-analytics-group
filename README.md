# 🦜 BirdCLEF+ 2026 — Autonomous Research Agent

**Advanced Predictive Analytics 2025/2026 — Course Project (Track B)**

An AI-powered autonomous research agent that designs, trains, evaluates, and iterates on deep learning models for bird species recognition from audio recordings. The agent uses a locally-hosted LLM (via Ollama) to drive the entire ML experimentation loop — from architecture proposal to code generation, training, evaluation, and error recovery — **without human intervention**.

🏆 **Competition:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

---

## 👥 Team

| Name | GitHub | Role |
|------|--------|------|
| Maximilian Noelle-Wying | [@Trumbly](https://github.com/Trumbly) | Owner |
| Lorena Leicht | [@Lorry171717](https://github.com/Lorry171717) | Collaborator |
| Sebastian Mis | [@SebastianMis23](https://github.com/SebastianMis23) | Collaborator |
| Danish Mujtaba Qureshi | [@danish-m-qureshi](https://github.com/danish-m-qureshi) | Collaborator |

---

## 🚀 Quick Start (for Dummies)

> **Prerequisites:** macOS or Linux, 16 GB+ RAM. Apple Silicon Mac (M1–M4) recommended for GPU training.

### Step 1: 📥 Clone the repo

```bash
git clone git@github.com:Trumbly/advanced-topics-in-predictive-analytics-group.git
cd advanced-topics-in-predictive-analytics-group
```

### Step 2: 🐍 Set up the Python environment

We use **conda** with Python 3.11. This is mandatory because some dependencies (TensorFlow) don't support newer Python versions.

```bash
# Install conda if you don't have it: https://docs.conda.io/en/latest/miniconda.html

conda create -n birdclef python=3.11 -y
conda activate birdclef
pip install -r requirements.txt
```

> 💡 **Troubleshooting:** If you see `ERROR: Could not find a version that satisfies the requirement tensorflow`, your Python is too new. Use conda with `python=3.11` as shown above.

### Step 3: 🤖 Install and start Ollama (the local LLM server)

Ollama lets you run LLMs locally without needing a GPU cloud service.

1. Download from https://ollama.com/download and install it
2. Pull a model:

```bash
# Our recommended model (~10 GB download, strong code generation)
ollama pull gemma4:e4b

# Alternatives (lighter but less reliable code generation):
ollama pull qwen3:9b           # ~6 GB
ollama pull nemotron-3-nano:4b # ~3 GB, smallest
```

Ollama runs as a background service — no need to start it manually after installation.

### Step 4: 📊 Download and preprocess the BirdCLEF data

You need a Kaggle account and an API token at `~/.kaggle/kaggle.json`:

```bash
pip install kaggle
python scripts/download_data.py --dest data/raw
bash scripts/preprocess.sh
```

This creates mel-spectrograms from the audio files. It only needs to run **once**.

### Step 5: 🌐 Launch the Web Dashboard

```bash
conda activate birdclef
python -m agent.cli ui
```

Open http://127.0.0.1:8000/ in your browser. You'll see the **🦜 BirdCLEF Agent Dashboard**.

### Step 6: 🧪 Start your first study

**Option A — From the Dashboard (recommended for beginners):**

1. Click **"+ New study"** on the dashboard
2. Fill in:
   - **Name:** `my first study`
   - **Hypothesis:** `Testing the baseline CNN architecture`
   - **Max experiments:** `5` (start small!)
3. Click **"Launch study"**
4. Watch the live progress on the study detail page 🎬

**Option B — From the Terminal:**

```bash
python -m agent.cli start \
    --study "my first study" \
    --hypothesis "Testing the baseline CNN" \
    --max-experiments 5
```

### Step 7: 👀 Watch the agent work

On the study detail page you'll see:
- 🔵 A **"Agent running" card** with a pipeline progress bar showing which step the agent is on (propose → generate → validate → execute → capture → analyze)
- 📜 **Live stdout** from the training subprocess (click "Show live stdout" to expand)
- 📊 The **experiment table** fills up in real time as experiments complete
- 📈 **Score progression chart** and **failure breakdown chart** update automatically

### Step 8: 📦 Export for Kaggle

Once the study finishes and you have a good score:
1. Click **"📦 Export Kaggle submission"** on the study detail page
2. The agent creates a Kaggle-compatible `.ipynb` notebook with `BIRDCLEF_DEVICE=cpu` pinned
3. Upload the notebook to Kaggle and submit! 🎉

---

## ✨ Features

### 🔄 Autonomous ML Research Loop

The agent runs a fully autonomous loop for each experiment:

```
┌──────────────────────────────────────────────────────────────┐
│                     🧠 ORCHESTRATOR                          │
│                                                              │
│  1. 💡 Propose Architecture (LLM)                            │
│     → "[efficientnet_b0] EfficientNet via TorchvisionAdapter"│
│                                                              │
│  2. 💻 Generate Code (LLM)                                   │
│     → Complete Python training script with __main__ guard     │
│                                                              │
│  3. ✅ Validate Code (static checks)                         │
│     → AST analysis: forbidden imports, hallucinated layers,  │
│       missing __main__ guard, EPOCHS cap                     │
│                                                              │
│  4. 🏋️ Execute Training (sandbox subprocess)                 │
│     → Runs on MPS/CUDA/CPU, live stdout streaming            │
│                                                              │
│  5. 📊 Capture Metrics                                       │
│     → F1 (primary), ROC-AUC, cmap@5, loss, training curves  │
│                                                              │
│  6. 🔍 Analyze Results (LLM)                                 │
│     → "The SE attention helped. Next: try a deeper ResNet."  │
│                                                              │
│  ↻ Repeat for N experiments (configurable budget)            │
└──────────────────────────────────────────────────────────────┘
```

### 🛡️ Error Recovery (two-layer retry system)

Most LLM-generated code has bugs. The agent tries **really hard** to fix them:

- ⚡ **Layer 1 — Codegen retry** (fast, ~10s each): When validation fails, the LLM regenerates the code with the validation error as feedback. Up to 5 retries. No subprocess needed.
- 🔧 **Layer 2 — Error recovery** (slower, ~5-15 min each): When execution crashes, the `error_recovery` prompt gives the LLM the broken code + traceback and asks for a fix. Up to 5 retries.

Total: up to **10 fix attempts** per experiment before giving up. 💪

### 🖥️ Web Dashboard

A full web dashboard at http://127.0.0.1:8000/ with:

| Page | What it shows |
|------|--------------|
| 📋 **Studies** | All studies with best score, status, experiment counts. Click "+ New study" to start one. |
| 📊 **Study Detail** | Experiment table (F1, ROC-AUC, cmap@5), charts, live running card with progress bar + stdout |
| 🔬 **Experiment Detail** | Generated code, task chain (prompts, LLM responses, errors), training curves, logs |
| 🧪 **Prompts** | A/B testing dashboard: create immutable versions, compare scores, set defaults |
| 📝 **Report** | Auto-generated study report with rendered Markdown, charts, and figures |

### 🧪 Prompt A/B Testing

- Prompts are **versioned** (`config/prompts/<task>/v1.yaml`, `v2.yaml`, ...)
- Versions are **immutable** once created — no editing, no score tampering 🔒
- The dashboard shows **mean experiment scores per prompt version**
- When starting a study, you can select which prompt version to use per task
- Each study records which prompts it used in `study.json`

### 🔗 Study Continuation

Build on a previous study instead of starting from scratch:
- Click **"🔄 Continue study"** on any completed study's detail page
- The new study inherits the predecessor's full experiment memory
- The LLM sees all previous architectures, scores, and failures
- No baseline repeat — it starts at experiment N+1 🚀

### ⚡ GPU Support (MPS/CUDA/CPU)

Training runs on the best available device:

| Setting in `config.yaml` | Behavior |
|--------------------------|----------|
| `training.device: auto` | 🍎 MPS on Apple Silicon, 🟢 CUDA on NVIDIA, CPU fallback |
| `training.device: mps` | Force Apple MPS (M1/M2/M3/M4) |
| `training.device: cuda` | Force NVIDIA CUDA |
| `training.device: cpu` | Force CPU (slow but always works) |

The Kaggle submission **always** uses CPU regardless (automatically enforced). ✅

### 📂 Git-Transportable Studies

Study results are committed to Git so the team can share them:
- `experiments/studies/` is tracked (not gitignored)
- Each experiment's `code.py` is copied into the study directory
- `scripts/commit_study.sh <study_id>` commits a study with a descriptive message
- `.gitattributes` prevents merge conflicts on parallel commits

### 📏 Metrics

Three metrics are computed per experiment:

| Metric | Description | Use |
|--------|-------------|-----|
| **🎯 F1 (macro)** | Primary metric. Actual classification performance at threshold 0.5. | Sorting, comparison, "best score" |
| **📈 ROC-AUC (macro)** | Ranking quality. Separates classes regardless of threshold. | Secondary comparison |
| **🏅 cmap@5** | Class-mean average precision at k=5. The real BirdCLEF metric. | Competition alignment |

---

## 📁 Project Structure

```
.
├── agent/                      # 🧠 Agent core modules
│   ├── models.py                   # Pydantic data models
│   ├── orchestrator.py             # Main agent loop + error recovery
│   ├── llm_client.py               # Ollama client
│   ├── prompt_engine.py            # YAML prompt template loader
│   ├── prompt_registry.py          # Versioned prompt management (A/B testing)
│   ├── prompt_scoring.py           # Score aggregation per prompt version
│   ├── context_handler.py          # Assembles LLM prompts
│   ├── memory.py                   # ExperimentMemory (top-k, predecessor seeding)
│   ├── executor.py                 # Sandbox subprocess (live stdout)
│   ├── metrics.py                  # Parses results.json
│   ├── logger.py                   # JSON + Markdown logs
│   ├── submission.py               # Kaggle notebook exporter
│   ├── report.py                   # Auto-generated study report
│   ├── cli.py                      # Click CLI
│   ├── handlers/                   # 🛡️ Predefined task handlers
│   └── ui/                         # 🌐 Web dashboard (FastAPI + Jinja2 + HTMX)
├── pipelines/                  # 🎵 Fixed audio preprocessing
│   ├── audio_pipeline.py
│   ├── data_loader.py              # PyTorch DataLoader + pos_weight
│   └── models.py                   # CnnSmallV1 + TorchvisionAdapter
├── registry/                   # 📦 Model catalog (4 models)
├── config/
│   ├── config.yaml                 # ⚙️ Global config
│   ├── pipelines/                  # Pipeline YAML definitions
│   └── prompts/                    # 🧪 Versioned prompt templates
├── experiments/studies/        # 📊 Study results (git-tracked)
├── sandbox/                    # 🏗️ Temp execution (gitignored)
├── scripts/                    # 🔧 Utility scripts
├── tests/                      # ✅ 324 pytest tests
└── README.md
```

---

## 💻 CLI Reference

```bash
# 🌐 Start the web dashboard
python -m agent.cli ui [--port 8000] [--host 127.0.0.1]

# 🚀 Start a new study
python -m agent.cli start \
    --study "study name" \
    --hypothesis "what we're testing" \
    [--max-experiments 20] \
    [--model gemma4:e4b] \
    [--predecessor study_20260412_...] \
    [--prompt-selection '{"generate_code": "v2"}']

# 📋 Resume / inspect / export
python -m agent.cli resume <study_id>
python -m agent.cli list
python -m agent.cli status [<study_id>]
python -m agent.cli show-best [<study_id>]
python -m agent.cli report [<study_id>]
python -m agent.cli submit [<study_id>]
```

---

## ⚙️ Configuration

All tunable settings live in `config/config.yaml`:

```yaml
llm:
  default_model: gemma4:e4b      # Which Ollama model to use
  temperature: 0.7               # Higher = more creative architectures

compute_budget:
  max_experiments: 20            # Experiments per study
  max_epochs_per_run: 5          # Epochs per experiment
  max_experiment_seconds: 7200   # 2h timeout per experiment

training:
  device: auto                   # auto | cpu | mps | cuda
  batch_size: 512
  num_workers: 8
  persistent_workers: true
```

---

## 🔧 How the Agent Works (Technical)

### 1. 📝 Prompt-driven architecture

Every LLM task is driven by a **YAML prompt template** (in `config/prompts/`). The template has a `system` prompt (rules, skeleton code, pitfalls) and a `template` with `{slot}` placeholders that get filled dynamically with experiment memory, dataset profile, and model registry.

### 2. 🛡️ The validation gauntlet

LLM-generated code passes through multiple static checks **BEFORE** execution:
- 🚫 **Forbidden imports** (subprocess, urllib, requests, socket)
- 🔍 **Hallucinated torch.nn layers** (reflection against `dir(torch.nn)` — suggests correct names!)
- 🔒 **Missing `__main__` guard** (required for multiprocessing DataLoader workers)
- ⏱️ **EPOCHS cap** (rejects literal overrides)
- ⚡ **LazyLinear initialization** (requires dummy forward pass before optimizer)

### 3. 📺 Live training monitoring

The executor writes `stdout.log` **line by line** as the training subprocess produces output. The dashboard's HTMX partial polls every 2 seconds. You see epoch progress, loss values, and "model built" messages in real time — like watching a terminal, but in your browser.

### 4. 🔗 Study continuation

When you "continue" a study, the new study's `ExperimentMemory` is pre-seeded with the predecessor's experiments. The LLM sees this full history in its context, so it doesn't repeat the baseline — it proposes the N+1 architecture based on everything learned so far.

---

## 📋 Changelog

All changes on the `max_development` branch, newest first:

| Date | Change | Impact |
|------|--------|--------|
| Apr 13 | 🎯 **F1 as primary metric**, fallback to ROC-AUC for old studies | Index + detail pages show F1 first |
| Apr 13 | 🔄 **Continue-study button** with modal on study detail page | One-click study continuation |
| Apr 13 | 🧪 **Prompt selection** in new-study form (collapsible) | Choose prompt versions per task |
| Apr 13 | 🧪 **Prompt A/B testing** — versioned prompts, immutable, score dashboard | `config/prompts/<task>/v1.yaml` layout |
| Apr 13 | 📂 **Git-transportable studies** — .gitignore updated, code.py copied | Team can share study results |
| Apr 13 | 🔗 **Study continuation** — predecessor dropdown, memory seeding | Build on previous study's work |
| Apr 13 | 📏 **cmap@5 + macro-F1** added alongside ROC-AUC | Three metrics per experiment |
| Apr 13 | 📺 **Live stdout.log** — written line-by-line during training | Real-time training progress in UI |
| Apr 13 | 💪 **More aggressive retries** (5+5), blacklist instead of whitelist | Higher success rate |
| Apr 13 | 🐛 **Prompt sidebar fix** — filename as label, not YAML name | report.yaml now accessible |
| Apr 13 | 🐛 **Fix stopped study status** — patches study.json on SIGTERM | Immediate UI update on stop |
| Apr 13 | 📝 **Report rendering** with marked.js + figure serving | Markdown + charts in browser |
| Apr 13 | ✏️ **Prompt editor** — view/edit YAML prompts from dashboard | `/prompts` nav link |
| Apr 13 | 📊 **Richer report figures** — score comparison + architecture families charts | 5 figures in report |
| Apr 13 | ⚡ **Codegen retry loop** (Layer 1) — re-generate on validation failure | 5 cheap retries before execution |
| Apr 13 | 🐛 **Fix 234→num_classes** — prompts no longer hardcode 234 | Correct for 206-class dataset |
| Apr 13 | 🛡️ **Fix top 5 runtime errors** — LazyLinear dummy forward, GRU guidance | 20x UninitializedParameter eliminated |
| Apr 12 | 👀 **Live experiment visibility** — pipeline progress bar, agent status | See what the agent is doing |
| Apr 12 | ▶️⏹️ **Start/Stop controls** — process manager, new-study form | Launch studies from the UI |
| Apr 12 | 🔄 **Live monitoring** — HTMX auto-refresh, running detection | Auto-updating stat cards |
| Apr 12 | 📦 **Kaggle submission button** — one-click export + history | Export from the dashboard |
| Apr 12 | 🌐 **Web dashboard** — FastAPI + Jinja2 + HTMX + Chart.js | Full UI at localhost:8000 |
| Apr 12 | 💪 **Agent reliability P0-P2** — retry-on-empty, LazyLinear, torchvision adapter, validation, pos_weight, anti-herding | Success rate 40% → 83% |
| Apr 12 | ⚡ **MPS/CUDA device support** — configurable via config.yaml | GPU training on Apple Silicon |
| Apr 11 | 🐛 **DataLoader pickle fix** — _TorchAdapter at module scope | Fixed every training crash |
| Apr 11 | 🔒 **__main__ guard** — required in skeleton, enforced by validator | Fixed multiprocessing bootstrap error |
| Apr 11 | ⚙️ **Configurable training** — batch_size, workers, epochs via config.yaml | No more hardcoded values |
| Apr 11 | 🏎️ **CPU saturation** — OMP_NUM_THREADS, bigger batches, persistent workers | Full core utilization |
| Apr 11 | 📝 **Auto-generated report** — LLM writes study summary with charts | Score progression, failure breakdown, learning curves |
| Apr 11 | 🔧 **Error recovery loop** — LLM rewrites broken code with traceback context | Resilient experimentation |
| Apr 11 | 📺 **Live subprocess streaming** — stdout/stderr streamed to terminal | See training progress live |
| Apr 10 | 🏗️ **Phases 0–6** — Full agent implementation from scratch | Data models → CLI → orchestrator → executor → submission |

---

## ✅ Running Tests

```bash
conda activate birdclef
pytest tests/ -q              # 324 tests, ~15 seconds
pytest tests/ -v              # verbose output
pytest tests/test_ui.py       # just the dashboard tests
pytest tests/test_orchestrator.py  # end-to-end agent loop tests
```

---

## 📜 License

See [LICENSE](LICENSE).
