# BirdCLEF+ 2026 — Autonomous Research Agent

**Advanced Predictive Analytics 2025/2026 — Course Project (Track B)**

An AI-powered autonomous research agent that designs, trains, evaluates, and iterates on deep learning models for bird species recognition from audio recordings. The agent uses a locally-hosted LLM to drive the ML experimentation loop.

**Competition:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

## Team

| Name | Role |
|------|------|
| TBD  | TBD  |
| TBD  | TBD  |
| TBD  | TBD  |
| TBD  | TBD  |

## Project Structure

```
.
├── agent/                  # Core agent modules (orchestrator, LLM client, code generator, executor)
├── pipelines/              # Fixed, tested audio processing code (mel-spectrograms, augmentation)
├── prompts/                # LLM prompt templates (exploration, exploitation, error recovery)
├── config/                 # YAML configuration files (LLM settings, compute budgets, hyperparams)
├── registry/               # Pretrained model catalog (BirdNET, YAMNet, EfficientNet) with metadata
├── experiments/
│   ├── memory/             # Compact experiment memory (JSON) — injected into LLM context
│   └── logs/               # Full experiment logs (generated code, prompts, metrics, markdown)
├── sandbox/                # Temporary execution environment for LLM-generated code
├── data/
│   ├── raw/                # Original Kaggle competition data (not tracked by git)
│   └── processed/          # Preprocessed spectrograms and features (not tracked by git)
├── models/                 # Saved model checkpoints (not tracked by git)
├── notebooks/              # Exploratory Jupyter notebooks
├── dashboard/              # Optional: web-based UI for monitoring and controlling the agent
├── reports/
│   └── figures/            # Figures and charts for the project report
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
| **Model Registry** | `registry/` | Catalog of available pretrained models (BirdNET, YAMNet, EfficientNet) with metadata (input shape, output dim, size, suitability). The agent selects from the catalog rather than hallucinating architectures. |
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

1. **Fixed pipeline, variable architecture** — The audio pipeline and data loader are stable, tested code. The agent only varies model architecture, hyperparameters, and training config.
2. **Explore cheap, exploit deep** — Start with small models, few epochs, data subsets. Only invest heavy compute into the most promising candidates (neural scaling laws).
3. **Log everything** — Every experiment is fully logged: prompt, code, metrics, LLM analysis. This powers both the agent's memory and the project report.
4. **Model-agnostic LLM** — The agent works with any locally-hosted LLM via the OpenAI-compatible API. Swap models without changing code.

## Setup

### 1. Clone and install dependencies

```bash
git clone git@github.com:Trumbly/advanced-topics-in-predictive-analytics-group.git
cd advanced-topics-in-predictive-analytics-group
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install and start a local LLM

We use [Ollama](https://ollama.com) to run LLMs locally.

```bash
# Install Ollama from https://ollama.com
# Then pull a model:
ollama pull gemma4        # ~9.6 GB — recommended default
```

### 3. Download the competition data

```bash
# Option A: Kaggle API
pip install kaggle
kaggle competitions download -c birdclef-2026 -p data/raw/

# Option B: Manual download from https://www.kaggle.com/competitions/birdclef-2026/data
# Place files in data/raw/
```

### 4. Run the agent

```bash
# TBD — will be updated once the agent is implemented
python -m agent.main
```

## License

See [LICENSE](LICENSE).
