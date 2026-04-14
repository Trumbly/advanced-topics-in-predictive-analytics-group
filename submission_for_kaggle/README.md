# Kaggle Submission Guide

## What goes in this folder

After the agent finishes training, this folder will contain your **ready-to-upload Kaggle notebook**:

```
submission_for_kaggle/
├── README.md                    ← you are here
├── export_best.sh               ← one-command export script
└── <exp_id>_submission.ipynb    ← the notebook you upload to Kaggle (generated)
```

## How it works — step by step

### Step 1: Train at home (already done if agent completed)

The agent trains multiple experiments on your local machine (MPS/GPU).
Results are saved in `experiments/studies/<study_id>/`.

### Step 2: Export the best experiment as a Kaggle notebook

```bash
# Option A: use the helper script (copies to this folder automatically)
bash submission_for_kaggle/export_best.sh

# Option B: use the CLI directly
python -m agent.main submit
# Then copy the notebook from experiments/studies/<study_id>/submissions/
```

### Step 3: Upload to Kaggle

1. Go to https://www.kaggle.com/competitions/birdclef-2026/submit
2. Click **"Submit"** or **"New Notebook Submission"**
3. Upload the `.ipynb` file from this folder
4. Kaggle runs it on their CPU servers (90-minute limit)
5. Your score appears on the leaderboard

## What the notebook does (inference only)

The exported notebook does **NOT train** — it only runs predictions:

1. **Loads** the trained model architecture (from the experiment code)
2. **Loads** the saved model weights (checkpoint from training)
3. **Reads** the secret test audio files (provided by Kaggle)
4. **Converts** audio to mel-spectrograms
5. **Runs** `model.predict()` on each spectrogram → probability per species
6. **Writes** `submission.csv` with columns: `row_id, species1, species2, ..., species206`

**Estimated runtime on Kaggle CPU: 15-20 minutes** (well within the 90-minute limit).

## What the notebook contains

| Cell | Contents |
|------|----------|
| 1 | Markdown header (study ID, experiment ID) |
| 2 | Config: forces `BIRDCLEF_DEVICE=cpu`, disables CUDA |
| 3 | Markdown: explains this is the experiment code |
| 4 | The full experiment code (model definition + inference) |
| 5 | Markdown: reminder to check `submission.csv` output |

## Important notes

- The notebook forces CPU mode (`os.environ["BIRDCLEF_DEVICE"] = "cpu"`)
- No CUDA/GPU calls are allowed (Kaggle CPU kernel constraint)
- The model weights must be attached as a Kaggle dataset or embedded in the notebook
- If the notebook fails on Kaggle, check the logs for missing dependencies
