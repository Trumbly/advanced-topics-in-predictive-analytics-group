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
│   ├── submission.py           # Kaggle notebook exporter
│   ├── cli.py                  # Click CLI (start / resume / list / status / submit)
│   ├── main.py                 # python -m agent.main entry point
│   └── handlers/               # Predefined task handlers
│       ├── validate_code.py
│       ├── execute_training.py
│       ├── capture_metrics.py
│       └── generate_submission.py
├── pipelines/              # Fixed audio preprocessing (NOT modified by the agent)
│   ├── audio_pipeline.py       # Mel-spectrogram, windowing, augmentation
│   ├── dataset_profile.py      # DatasetProfile builder
│   └── data_loader.py          # Reads precomputed .npy files into torch datasets
├── registry/               # Pretrained model catalog
│   ├── models.yaml             # LLM picks from this catalog
│   └── registry.py             # Loader + markdown export for prompt injection
├── config/
│   ├── config.yaml             # Global runtime config
│   ├── pipelines/              # default / exploration / exploitation
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
| **Dashboard** *(optional)* | `dashboard/` | Web-based UI (FastAPI + React/HTMX) for live monitoring, experiment browsing, prompt management, and one-click Kaggle export. |

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

```bash
git clone git@github.com:Trumbly/advanced-topics-in-predictive-analytics-group.git
cd advanced-topics-in-predictive-analytics-group
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

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
```

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

## CLI Reference

| Command | Description |
|---------|-------------|
| `start --study <name>` | Create a new Study and run the full loop |
| `resume <study_id>` | Continue an existing Study from its saved state |
| `list` | Table of all Studies on disk |
| `status [<study_id>]` | Show one Study's metadata (defaults to most recent) |
| `show-best [<study_id>]` | Pretty-print the best experiment's JSON |
| `submit [<study_id>]` | Export the best experiment as a Kaggle notebook |

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

`exploration_pipeline.yaml` runs with tiny models, 3 epochs, and 10% of the
data — ideal for the first ~10 experiments to quickly probe architecture
families. Once you identify a promising family, switch to
`exploitation_pipeline.yaml`, which scales the same flow to full data and
longer training.

### Memory and prompt construction

`ExperimentMemory` persists every experiment as JSON + regenerated
Markdown. On each new experiment, the `ContextHandler` pulls the top-K
successes and recent failures from memory, combines them with the
`DatasetProfile` + `ModelRegistry`, and fills the prompt template.
If the assembled prompt exceeds the token budget, memory is progressively
trimmed until it fits.

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
