# BirdCLEF+ 2026 — Autonomous Research Agent

**Advanced Predictive Analytics 2025/2026 — Course Project (Track B)**

An AI-powered autonomous research agent that designs, trains, evaluates, and iterates on deep learning models for bird species recognition from audio recordings. The agent uses a locally-hosted LLM to drive the ML experimentation loop.

**Competition:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

## Team

| Name | GitHub | Role |
|------|--------|------|
| Maximilian Noelle-Wying | [@Trumbly](https://github.com/Trumbly) | Owner |
| Lorena Leicht | [@Lorry171717](https://github.com/Lorry171717) | Collaborator |
| Sebastian Mis | [@SebastianMis23](https://github.com/SebastianMis23) | Collaborator |
| Danish Mujtaba Qureshi | [@danish-m-qureshi](https://github.com/danish-m-qureshi) | Collaborator |

## Project Structure

```
.
├── agent/                  # Agent loop modules
│   ├── models.py               # Pydantic data models (Study, Experiment, Task, ...)
│   ├── orchestrator.py         # Main loop
│   ├── llm_client.py           # Ollama (OpenAI-compatible) client
│   ├── prompt_engine.py        # YAML prompt template loader
│   ├── context_handler.py      # Assembles final LLM prompts
│   ├── memory.py               # ExperimentMemory (top-k, markdown)
│   ├── executor.py             # Sandbox subprocess runner
│   ├── metrics.py              # Parses results.json
│   ├── logger.py               # JSON + Markdown study/experiment/task logs
│   ├── report.py               # Auto-generated study report
│   ├── submission.py           # Kaggle notebook exporter
│   ├── cli.py                  # Click CLI (start / resume / list / status / submit / ui)
│   ├── main.py                 # python -m agent.main entry point
│   ├── handlers/               # Predefined task handlers
│   │   ├── validate_code.py
│   │   ├── execute_training.py
│   │   ├── capture_metrics.py
│   │   └── generate_submission.py
│   └── ui/                     # Web dashboard (FastAPI + Jinja2/HTMX)
│       ├── app.py                  # Routes, API endpoints
│       ├── loaders.py              # Read-only disk access for studies/experiments
│       ├── process_manager.py      # Background process start/stop
│       └── templates/              # HTML templates (study, experiment, files, prompts)
├── pipelines/              # Fixed audio preprocessing (NOT modified by the agent)
│   ├── audio_pipeline.py       # Mel-spectrogram, windowing, augmentation
│   ├── dataset_profile.py      # DatasetProfile builder
│   └── data_loader.py          # Reads precomputed .npy files into torch datasets
├── registry/               # Pretrained model catalog
│   ├── models.yaml             # LLM picks from this catalog
│   └── registry.py             # Loader + markdown export for prompt injection
├── config/
│   ├── config.yaml             # Global runtime config (MPS/GPU development)
│   ├── config_exam.yaml        # CPU-only exam config (batch=32, 3 epochs, 1hr timeout)
│   ├── pipelines/              # default / exploration / exploitation / exam
│   ├── prompts/                # propose_architecture / generate_code / analyze_results / error_recovery
│   └── tasks/                  # Predefined handler documentation
├── scripts/
│   ├── preprocess.sh           # One-shot download + build_profile
│   ├── download_data.py        # Kaggle CLI wrapper
│   └── build_profile.py        # Runs AudioPipeline + builds DatasetProfile
├── experiments/
│   └── studies/                # Per-study JSON + Markdown logs (gitignored)
├── data/
│   ├── raw/                    # Kaggle download (gitignored)
│   └── processed/              # .npy spectrograms + dataset_profile.json (gitignored)
├── sandbox/                # Subprocess workdir for generated code (gitignored)
├── models/                 # Saved model checkpoints (gitignored)
├── notebooks/
│   └── manual_baseline.ipynb   # Hand-crafted baseline for the report comparison
├── reports/figures/        # Report assets
├── tests/                  # pytest suite (154 tests)
├── Project_Handout.pdf
├── requirements.txt
└── README.md
```

### Architecture Overview

The agent is composed of several decoupled modules that work together in an autonomous experimentation loop:

| Module | Location | Description |
|--------|----------|-------------|
| **Orchestrator** | `agent/` | Main loop controller. Drives the Propose → Generate → Execute → Evaluate → Iterate cycle. Manages stopping criteria (max iterations, score plateau, time budget) and two-phase strategy: broad exploration first, then exploitation of best candidates. |
| **LLM Client** | `agent/` | Model-agnostic interface to locally-hosted LLMs via OpenAI-compatible API (Ollama). Supports swapping between Gemma 4, Qwen 3, DeepSeek-R1, etc. without code changes. |
| **Prompt Engine** | `prompts/` | Situational prompt templates instead of one monolithic prompt. Separate templates for exploration, exploitation, error recovery, and final submission. Slots are filled dynamically with experiment history and metrics. |
| **Context Handler** | `agent/` | Assembles the final LLM prompt from templates + experiment memory. Handles context window management — decides what fits and what gets trimmed. |
| **Experiment Memory** | `experiments/memory/` | Structured JSON registry of all past runs: architecture, hyperparameters, score, duration, errors. Prevents the LLM from repeating failed experiments. Compact summaries are injected into the LLM context. |
| **Experiment Logger** | `experiments/logs/` | Full logs for every run: the prompt sent, code generated, training metrics, and LLM analysis. Human-readable markdown + structured JSON. Essential for the report and debugging. |
| **Code Generator** | `agent/` | Translates LLM output into executable Python code. Only architecture, hyperparameters, and training config are generated — fixed building blocks (audio pipeline, data loader) are referenced, not rewritten. |
| **Code Executor / Sandbox** | `sandbox/` | Runs generated code in isolation. Catches errors (OOM, syntax, shape mismatch, timeouts). Returns structured results (metrics, logs, exit status). Enforces compute budget per run. |
| **Audio Pipeline** | `pipelines/` | Fixed, well-tested code for audio preprocessing: mel-spectrogram generation, 5-second windowing, augmentation (time-shift, noise injection, mixup). Parameters are configurable by the agent, but the code itself is stable. |
| **Model Registry** | `registry/` | Catalog of verified building blocks (CnnSmallV1 baseline, torchvision backbones) with metadata (input shape, output dim, size, suitability) and ready-to-use import snippets. The agent uses the registry as a starting point, but is **not limited to it** — it is free to design custom `nn.Module` architectures inline and combine them with any public `torch` / `torchvision` / `torchaudio` APIs. |
| **Metrics Collector** | `agent/` | Collects ROC-AUC (macro-averaged), loss, learning curves, run duration, and resource usage. Enables comparison across runs. |
| **Submission Exporter** | `agent/` | Exports the best model + inference pipeline as a standalone Kaggle notebook that meets the CPU-only, 90-minute runtime constraint. |
| **Dashboard** | `agent/ui/` | Web-based UI (FastAPI + Jinja2/HTMX) for live monitoring with a real-time terminal, experiment browsing, file browser, prompt editor, and one-click Kaggle export. Launch with `python -m agent.main ui`. |

### Agent Loop

```
┌─────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                        │
│                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐     │
│   │  Prompt   │───>│   LLM    │───>│    Code      │     │
│   │  Engine   │    │  Client  │    │  Generator   │     │
│   └────▲─────┘    └──────────┘    └──────┬───────┘     │
│        │                                  │             │
│        │                                  ▼             │
│   ┌────┴─────┐    ┌──────────┐    ┌──────────────┐     │
│   │ Context  │    │ Metrics  │<───│   Sandbox    │     │
│   │ Handler  │    │ Collector│    │  (Executor)  │     │
│   └────▲─────┘    └────┬─────┘    └──────────────┘     │
│        │               │                                │
│        │               ▼                                │
│   ┌────┴───────────────────────┐                       │
│   │    Experiment Memory       │                       │
│   └────────────────────────────┘                       │
│                                                         │
│   Phase 1: EXPLORATION (broad, fast, few epochs)       │
│   Phase 2: EXPLOITATION (scale best candidates)        │
└─────────────────────────────────────────────────────────┘

Fixed modules (not generated by agent):
  ├── Audio Pipeline (pipelines/)
  └── Model Registry (registry/)
```

### Core Principles

1. **Fixed pipeline, free-form architectures** — The audio pipeline and data loader are stable, tested code (spectrogram generation, windowing, augmentation presets). The agent varies *everything above that*: model architecture (from registry **or** custom `nn.Module` classes written inline), hyperparameters, training config, loss design, ensembling.
2. **Registry = starting points, not a cage** — The Model Registry (`registry/models.yaml`) is a catalog of verified building blocks with ready-to-copy import snippets. The agent uses it when it fits, but it is encouraged to design novel architectures from `torch.nn` primitives when the situation calls for it.
3. **Explore cheap, exploit deep** — Start with small models, few epochs, data subsets. Only invest heavy compute into the most promising candidates (neural scaling laws).
4. **Log everything** — Every experiment is fully logged: prompt, code, metrics, LLM analysis. This powers both the agent's memory and the project report.
5. **Model-agnostic LLM** — The agent works with any locally-hosted LLM via the OpenAI-compatible API. Swap models without changing code.

## Setup

### 1. Clone and install dependencies

We recommend **conda with Python 3.11** because TensorFlow currently only
ships wheels for Python 3.9–3.12, and most ML libraries are best-tested on
3.11. A plain `venv` with Python 3.11/3.12 works too.

```bash
git clone git@github.com:Trumbly/advanced-topics-in-predictive-analytics-group.git
cd advanced-topics-in-predictive-analytics-group

# Option A (recommended): conda
conda create -n birdclef python=3.11 -y
conda activate birdclef
pip install -r requirements.txt

# Option B: venv (only works if your system python is 3.11 or 3.12)
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> If you see `ERROR: Could not find a version that satisfies the requirement tensorflow`,
> your Python is too new for TF. Use conda with `python=3.11` as shown above.

### 2. Install and start a local LLM

We use [Ollama](https://ollama.com) to run LLMs locally.

```bash
# Install Ollama from https://ollama.com/download
# Start the Ollama server (macOS/Windows: it's a background service)
ollama serve &

# Pull a model — any of these work, Gemma 4 E4B is the default
ollama pull gemma4:e4b      # ~10 GB, strong all-rounder
ollama pull qwen3:9b        # ~6 GB, lighter
ollama pull deepseek-r1:8b  # ~5 GB, reasoning-focused
```

You can change the default model in `config/config.yaml` (`llm.default_model`)
or pass `--model` on the command line.

### 3. Download and preprocess the competition data

First-time setup (requires a Kaggle account and an API token at
`~/.kaggle/kaggle.json`):

```bash
pip install kaggle
python scripts/download_data.py --dest data/raw
```

Then run the preprocessing pipeline once. This converts raw audio into
mel-spectrograms, writes a label file, and builds the `DatasetProfile`
the agent reads at runtime:

```bash
# Quick smoke test with a small subset
python scripts/build_profile.py --sample 100

# Full dataset (slow, but only has to run once)
python scripts/build_profile.py
# or
bash scripts/preprocess.sh
```

After this step, `data/processed/` contains:
- `spectrograms/*.npy`       — precomputed mel-spectrograms
- `labels.csv`               — sample_id → class_id mapping
- `dataset_profile.json`     — statistics + train/val split (seed=42)

### 4. Run the agent

```bash
# Show available commands
python -m agent.main --help

# List existing studies
python -m agent.main list

# Start a new study (uses config/pipelines/default_pipeline.yaml by default)
python -m agent.main start \
    --study baseline_run \
    --hypothesis "First end-to-end baseline with small CNN" \
    --max-experiments 5

# Resume an existing study
python -m agent.main resume study_20260410_120000_baseline_run

# Check status of the most recent study
python -m agent.main status

# Pretty-print the best experiment
python -m agent.main show-best

# Export the best experiment as a Kaggle submission notebook
python -m agent.main submit

# Launch the web dashboard (live terminal, file browser, prompts)
python -m agent.main ui
```

#### Exam / CPU-only mode

During the exam (CPU only, limited time), use the exam config:

```bash
python -m agent.main start \
    --config config/config_exam.yaml \
    --pipeline config/pipelines/exam_pipeline.yaml \
    --study exam_run \
    --hypothesis "CPU-optimized run for exam" \
    --max-experiments 5
```

| Setting | Default (MPS/GPU) | Exam (CPU) |
|---------|-------------------|------------|
| `device` | mps | cpu |
| `batch_size` | 128 | 32 |
| `max_epochs` | 7 | 3 |
| `timeout/experiment` | 4 hours | 1 hour |
| `num_workers` | 8 | 4 |

The exam config is only used when you explicitly pass `--config config/config_exam.yaml`. Without it, the default MPS config is used.

Artifacts are written under `experiments/studies/<study_id>/`:
- `study.json` / `study.md`           — study metadata
- `memory.json` / `memory.md`         — experiment memory (what the LLM sees)
- `experiments/<exp_id>/`             — per-experiment JSON + Markdown + tasks
- `submissions/<exp_id>_submission.ipynb`  — exported Kaggle notebook

### 5. Run the test suite

```bash
pytest tests/                     # 154 tests, runs in ~2s
pytest tests/ -v                  # verbose
pytest tests/test_orchestrator.py # just the end-to-end smoke tests
```

### 6. Reviewer quick start (CPU-only machines)

If you are reviewing this project on a CPU-only machine (no GPU/MPS), use the
exam config to verify the agent works end-to-end. This runs with 3 epochs,
batch_size=32, and a 1-hour timeout — enough to complete 1-2 experiments and
see the full autonomous loop in action.

```bash
# 1. Install dependencies + start Ollama (see steps 1-3 above)
# 2. Run the agent with CPU-optimized settings:
python -m agent.main start \
    --config config/config_exam.yaml \
    --pipeline config/pipelines/exam_pipeline.yaml \
    --study reviewer_demo \
    --hypothesis "CPU demo for review" \
    --max-experiments 2
```

This will:
- Ask the LLM to propose an architecture
- Generate executable training code
- Train on CPU (~30-60 min per experiment with 3 epochs)
- Capture metrics (ROC-AUC, cmap@5, F1)
- Feed results back to the LLM for analysis
- Iterate with an improved architecture

**Pre-computed results:** Full training results from our MPS (Apple Silicon GPU)
runs are already saved in `experiments/studies/`. To browse them without
re-training, launch the dashboard:

```bash
python -m agent.main ui
# Open http://127.0.0.1:8000 — browse all studies, experiments, and metrics
```

**Kaggle submission:** The agent exports the best model as a standalone Kaggle
notebook (CPU-only inference, well within the 90-minute runtime limit):

```bash
python -m agent.main submit
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `start --study <name>` | Create a new Study and run the full loop |
| `resume <study_id>` | Continue an existing Study from its saved state |
| `list` | Table of all Studies on disk |
| `status [<study_id>]` | Show one Study's metadata (defaults to most recent) |
| `show-best [<study_id>]` | Pretty-print the best experiment's JSON |
| `submit [<study_id>]` | Export the best experiment as a Kaggle notebook |
| `ui [--port 8000]` | Launch the web dashboard (live terminal, file browser, prompts) |

Global options:
- `--config <path>` — override `config/config.yaml`
- `--log-level {DEBUG,INFO,WARNING,ERROR}`

Per-command options for `start`:
- `--hypothesis <text>` — one-line description of what the study tests
- `--pipeline <yaml>` — override the default pipeline
- `--model <name>` — override the LLM model (e.g. `gemma4:e4b`, `qwen3:9b`)
- `--max-experiments <N>` — override the compute budget
- `--mode {autonomous,interactive}` — interactive mode is a placeholder

## How it works

### Pipelines are data, not code

The agent's execution flow is defined entirely by YAML files under
`config/pipelines/`. Each pipeline is a sequence of steps, and each step is
either an LLM call (with a prompt template) or a predefined Python handler.
To add a new experiment flavor, copy `default_pipeline.yaml`, edit the steps,
and pass `--pipeline path/to/your.yaml` to `start`.

### Two-phase strategy

The agent uses a two-phase strategy with pipeline-specific epoch budgets:

| Pipeline | Max Epochs | Purpose |
|----------|------------|--------|
| `exploration_pipeline.yaml` | 3 | Fast probing of architecture families on the full dataset |
| `exploitation_pipeline.yaml` | 5 | Scale and tune the most promising candidates |
| `default_pipeline.yaml` | 7 | Standard runs with full training budget |
| `exam_pipeline.yaml` | 3 | CPU-only exam mode with 1-hour timeout |

All pipelines use **early stopping** (patience=3 on validation ROC-AUC),
so experiments stop sooner if the model has already converged.

### Memory and prompt construction

`ExperimentMemory` persists every experiment as JSON + regenerated
Markdown. On each new experiment, the `ContextHandler` pulls the top-K
successes and recent failures from memory, combines them with the
`DatasetProfile` + `ModelRegistry`, and fills the prompt template.
If the assembled prompt exceeds the token budget, memory is progressively
trimmed until it fits.

### Training configuration

Key training settings (configured in `config/config.yaml`):

| Setting | Value | Notes |
|---------|-------|-------|
| `max_epochs_per_run` | 7 | Default cap; pipelines can override (exploration=3, exploitation=5) |
| `batch_size` | 128 | Safe for 24 GB Apple Silicon; configurable |
| `max_experiment_seconds` | 14400 | 4-hour timeout per experiment |
| Early stopping patience | 3 | Stops if validation ROC-AUC does not improve for 3 epochs |
| LR scheduler | Cosine annealing | `CosineAnnealingLR(optimizer, T_max=EPOCHS)` |
| First experiment | Pretrained EfficientNet-B0 | Always starts with transfer learning for a strong baseline |
| Device | MPS (Apple Silicon) | Auto-detected; also supports CUDA and CPU |

### Validation before execution

LLM-generated code runs through `agent.handlers.validate_code` before it
ever reaches the sandbox subprocess:
- AST-level rejection of forbidden imports (`subprocess`, `urllib`,
  `requests`, `socket`, ...)
- Substring rejection of dangerous patterns (`os.system`, `eval(`,
  `exec(`, ...)
- Optional allowlist of expected imports (configured in the pipeline YAML)

This catches most bad-code cases before spending sandbox time on them.

## License

See [LICENSE](LICENSE).
