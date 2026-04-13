# Task exp_001_task_01_propose_architecture

- **Experiment:** exp_001
- **Type:** llm
- **Name:** propose_architecture
- **Status:** failed
- **Started:** 2026-04-10 13:07:07.805428+00:00
- **Completed:** 2026-04-10 13:07:34.181029+00:00

## Prompt Used
```
[SYSTEM]
You are an ML research agent specializing in audio classification on mel-spectrograms.
You are working on BirdCLEF 2026 — a 234-class MULTI-LABEL bird species classification task.
Key constraints:
  - The final Kaggle submission notebook must run on CPU only, within 90 minutes.
  - Audio has already been preprocessed into mel-spectrograms by a fixed pipeline.
  - You MUST only propose architectures that use models from the provided Model Registry
    or combinations thereof. Do not invent exotic or unsupported components.
  - Output is a vector of 234 independent sigmoid probabilities (NOT a softmax).


[USER]
No previous experiments yet. Start with a simple, fast baseline:
a small CNN (3 Conv2D layers, ~500k parameters) trained for 3 epochs on 128-mel
spectrograms with basic augmentation (time-shift + noise injection). Goal: verify
the pipeline end-to-end. Tuning comes later.

```

## LLM Response
```
This is the perfect starting point. Given the strict constraints (CPU, speed, multi-label output), we must prioritize depth and parameter efficiency over raw capacity.

I propose the following architecture and training protocol to serve as a fast, robust baseline.

---

### 🔬 Baseline Architecture Proposal: Shallow CNN Feature Extractor

The goal is to build a lightweight convolutional stack that extracts local acoustic features efficiently, followed by a classification head suitable for multi-label output.

**Model Name:** `Baseline_3Layer_CNN`

**Conceptual Structure:**
$$\text{Input} \rightarrow \text{Conv}_1 \rightarrow \text{ReLU} \rightarrow \text{Pool} \rightarrow \text{Conv}_2 \rightarrow \text{ReLU} \rightarrow \text{Pool} \rightarrow \text{Conv}_3 \rightarrow \text{ReLU} \rightarrow \text{GAP} \rightarrow \text{Dense}(\text{234})$$

**Detailed Layer Specification:**

1.  **Input Layer:** Mel-Spectrogram ($T \times 128$).
2.  **Convolution Block 1:**
    *   `Conv2D`: Kernel $(3 \times 3)$, Stride $1$.
    *   `Batch Normalization` (BN).
    *   `Activation`: ReLU.
    *   *Filters:* 32
3.  **Pooling 1:**
    *   `MaxPool2D`: Kernel $(2 \times 2)$, Stride $2$ (Downsamples both time and frequency dimensions, reducing computational load).
4.  **Convolution Block 2:**
    *   `Conv2D`: Kernel $(3 \times 3)$, Stride $1$.
    *   `BN` $\rightarrow$ `ReLU`.
    *   *Filters:* 64
5.  **Pooling 2:**
    *   `MaxPool2D`: Kernel $(2 \times 2)$, Stride $2$.
6.  **Convolution Block 3:**
    *   `Conv2D`: Kernel $(3 \times 3)$, Stride $1$.
    *   `BN` $\rightarrow$ `ReLU`.
    *   *Filters:* 128
7.  **Global Feature Aggregation:**
    *   `GlobalAveragePooling2D` (GAP): This is far superior to flattening for spectrograms as it maintains spatial invariance while drastically reducing parameters.
8.  **Classification Head:**
    *   `Dense` Layer: Input size = (Number of features remaining after GAP).
    *   *Output Dimension:* 234
    *   *Activation:* **Sigmoid** (Crucial for independent multi-label predictions).

**Parameter Estimate Check:**
This structure is highly controlled. By using small kernels ($3 \times 3$), decreasing filter counts slightly, and relying heavily on GAP, we estimate the parameter count to be comfortably under the 500k target, ensuring fast CPU inference.

### ⚙️ Training Protocol

*   **Optimization:** Adam or AdamW (Standard recommendation).
*   **Loss Function:** **Binary Cross-Entropy Loss (BCE)**.
    *   *Rationale:* Since we are predicting 234 independent probabilities via Sigmoid, the loss must treat each class as a separate Bernoulli trial.
*   **Metrics:** AUC-ROC (Primary) and overall F1 Score (Secondary).
*   **Epochs:** 3.
*   **Learning Rate Scheduling:** Fixed initial learning rate (e.g., $1e-3$) for simplicity in the baseline, or a very simple linear decay if time permits.

### ➕ Data Augmentation Pipeline

The augmentation strategy must be applied *before* the model forward pass:

1.  **Time-Shift Jitter:** Randomly shift the time axis of the spectrogram by $\pm 10\%$ (ensures robustness to slight temporal variations in bird calls).
2.  **Noise Injection:** Add Gaussian white noise scaled by a small factor (e.g., $\sigma=0.05$) to the spectrogram magnitude bins (prevents overfitting to perfect spectrogram representations).

---

### Summary Action Plan

| Component | Specification | Rationale |
| :--- | :--- | :--- |
| **Model** | `Baseline_3Layer_CNN` (As defined above) | Fast, parameter-efficient, CPU-friendly CNN. |
| **Loss Function** | Binary Cross-Entropy (BCE) | Correct loss formulation for 234 independent Sigmoid outputs. |
| **Activation** | Sigmoid (Output Layer only) | Required for multi-label probability vector output. |
| **Augmentation** | Time-Shift $\pm 10\%$, Noise Injection ($\sigma=0.05$) | Basic robustness testing. |
| **Training** | 3 Epochs | As requested for initial validation. |

This setup is minimal, fast to train, and directly addresses the multi-label output constraint, allowing us to verify the entire pipeline end-to-end before investing time in more complex architectures.
```

## Error
- **type:** ValueError
- **message:** propose_architecture: LLM did not return valid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
