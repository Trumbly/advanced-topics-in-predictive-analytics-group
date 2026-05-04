# I-11 — Training skeleton (Jinja2)

**Labels:** `track-task`, `p0`
**Milestone:** week-2-loop-and-task
**Owner:** Dev C

## Context
ADR-003. LLM writes only `build_model(num_classes)`. Everything else is in `config/skeletons/audio_multilabel.py.j2`.

## Scope
Ship a complete training script as a Jinja template. Fills env-driven knobs. Writes `results.json` per the locked schema.

## Template contract
Env vars consumed:
- `AGENT_DEVICE` — `"cpu"` only for now.
- `AGENT_BATCH_SIZE` — int.
- `AGENT_EPOCHS` — int, clamped to `max_epochs_per_run` by orchestrator.
- `AGENT_PROCESSED_DIR` — path to precomputed mel shards.
- `AGENT_CHECKPOINT_IN` — optional path to `state_dict.pt` for warm-start.
- `AGENT_CHECKPOINT_OUT` — path to write best `state_dict.pt`.
- `AGENT_SEED` — int for `torch.manual_seed`.
- `AGENT_LR` — float.
- `AGENT_LR_SCHEDULE` — `"constant"|"cosine"|"onecycle"`.

LLM block:
```python
# --- AGENT_BUILD_MODEL_START ---
def build_model(num_classes: int) -> nn.Module:
    ...
# --- AGENT_BUILD_MODEL_END ---
```

Skeleton body:
- Loads mel shards via `load_audio_dataset(processed_dir, split="train")` + `split="val"`.
- Splits 80/20 train/val if no val shard.
- Builds model via `build_model(num_classes=234)`.
- Optional warm-start from `AGENT_CHECKPOINT_IN`.
- `BCEWithLogitsLoss` (multi-label).
- Optimizer: `AdamW(lr=AGENT_LR)`.
- Scheduler per `AGENT_LR_SCHEDULE`.
- Epoch loop with per-epoch train+val metrics logged via `log_event("epoch", ...)`.
- Metrics: `f1_macro`, `roc_auc_macro` (sklearn).
- Early stop on 3 epochs no-improvement.
- Saves best checkpoint to `AGENT_CHECKPOINT_OUT`.
- Writes `results.json`:
```json
{"primary_score": 0.41, "primary_metric": "f1_macro",
 "metrics": {"f1_macro":0.41, "roc_auc_macro":0.86},
 "history": [{"epoch":1,"loss":0.7,"f1_macro":0.3,"roc_auc_macro":0.7}],
 "stopped_early": false, "duration_seconds": 42.1}
```

## Files
- Create `config/skeletons/audio_multilabel.py.j2`
- Create `tests/test_skeleton.py` — renders + runs on 100-sample synthetic shard with identity `build_model`

## Acceptance
- End-to-end run on synthetic shard in ≤60 s CPU.
- `results.json` validates against `ExecutionResult` expected keys.
- Warm-start test: with `AGENT_CHECKPOINT_IN` set, first-epoch loss lower than cold start.

## Depends on
I-10.
