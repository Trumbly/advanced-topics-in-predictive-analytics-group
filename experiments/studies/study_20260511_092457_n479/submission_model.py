"""Track B audio multi-label training skeleton.

Render-time (Jinja) variables filled in by the orchestrator:
- num_classes, input_tensor_shape, primary_metric

Run-time (env) knobs read from os.environ:
- AGENT_DEVICE, AGENT_BATCH_SIZE, AGENT_EPOCHS, AGENT_PROCESSED_DIR,
  AGENT_CHECKPOINT_IN (optional), AGENT_CHECKPOINT_OUT, AGENT_SEED,
  AGENT_LR, AGENT_LR_SCHEDULE.

The LLM owns ONLY the build_model block between the START/END markers.
Everything else is fixed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

# ---------- runtime knobs ----------
NUM_CLASSES = 234
INPUT_SHAPE = tuple([1, 128, 313])
PRIMARY_METRIC = "roc_auc_macro"

DEVICE = os.environ.get("AGENT_DEVICE", "cpu")
BATCH_SIZE = int(os.environ.get("AGENT_BATCH_SIZE", "8"))
NUM_WORKERS = int(os.environ.get("AGENT_NUM_WORKERS", "0"))
EPOCHS = int(os.environ.get("AGENT_EPOCHS", "3"))
PROCESSED_DIR = Path(os.environ.get("AGENT_PROCESSED_DIR", "data/processed/mels"))
CHECKPOINT_IN = os.environ.get("AGENT_CHECKPOINT_IN")
CHECKPOINT_OUT = os.environ.get("AGENT_CHECKPOINT_OUT", "checkpoint.pt")
SEED = int(os.environ.get("AGENT_SEED", "42"))
LR = float(os.environ.get("AGENT_LR", "1e-3"))
LR_SCHEDULE = os.environ.get("AGENT_LR_SCHEDULE", "constant")
WEIGHT_DECAY = float(os.environ.get("AGENT_WEIGHT_DECAY", "0.0"))

torch.manual_seed(SEED)


# ---------- data ----------
class LazyMelDataset(Dataset):
    """Reads (x, y) per sample from .npy files listed in a JSON index.

    Used when the processed dir contains ``<split>_index.json`` (the lazy
    preprocess output) instead of an eager ``<split>.pt`` tensor — lets
    the loop train on hundreds of thousands of mels without holding them
    all in RAM.
    """

    def __init__(self, index_path: Path):
        with index_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.spectrograms_dir = Path(payload["spectrograms_dir"])
        self.num_classes = int(payload["num_classes"])
        self.entries = payload["samples"]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        arr = np.load(self.spectrograms_dir / f"{e['sid']}.npy").astype("float32")
        if arr.ndim == 2:
            arr = arr[None]
        x = torch.from_numpy(arr)
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        # Prefer the multi-label index emitted by build_real_shards; fall back
        # to the legacy single-class field for old indices.
        indices = e.get("class_indices")
        if indices is not None:
            for ci in indices:
                y[int(ci)] = 1.0
        else:
            y[int(e["class_idx"])] = 1.0
        return x, y


def load_audio_dataset(processed_dir: Path, split: str):
    """Return a Dataset for ``split`` ('train' or 'val').

    Preference order:
      1. ``<split>_index.json`` -> LazyMelDataset (recommended; full corpus)
      2. ``<split>.pt`` -> eager TensorDataset (used by the smoke path)
      3. None when neither file is present (caller falls back to split).
    """
    index_path = processed_dir / f"{split}_index.json"
    if index_path.exists():
        return LazyMelDataset(index_path)
    pt_path = processed_dir / f"{split}.pt"
    if not pt_path.exists():
        return None
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    return TensorDataset(blob["x"], blob["y"])


def make_loaders():
    train = load_audio_dataset(PROCESSED_DIR, "train")
    if train is None:
        raise FileNotFoundError(
            f"missing processed shard: {PROCESSED_DIR / 'train.pt'} "
            f"(or {PROCESSED_DIR / 'train_index.json'})"
        )
    val = load_audio_dataset(PROCESSED_DIR, "val")
    if val is None:
        n_train = int(0.8 * len(train))
        train, val = random_split(train, [n_train, len(train) - n_train])
    return (
        DataLoader(
            train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            persistent_workers=NUM_WORKERS > 0,
        ),
        DataLoader(
            val,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=NUM_WORKERS > 0,
        ),
    )


# --- AGENT_BUILD_MODEL_START ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SpecAugmentWrap(nn.Module):
    def __init__(self, inner: nn.Module, freq_mask: int = 16,
                 time_mask: int = 32, num_masks: int = 2):
        super().__init__()
        self.inner = inner
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.num_masks = num_masks

    def _mask(self, x: torch.Tensor, dim: int, max_w: int) -> torch.Tensor:
        if max_w <= 0 or not self.training:
            return x
        B, C, F, T = x.shape
        size = x.shape[dim]
        for _ in range(self.num_masks):
            w = int(torch.randint(0, max_w + 1, (1,)).item())
            if w == 0:
                continue
            start = int(torch.randint(0, size - w + 1, (1,)).item())
            if dim == 2:
                x = x.clone()
                x[:, :, start:start + w, :] = 0.0
            else:
                x = x.clone()
                x[:, :, :, start:start + w] = 0.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self._mask(x, dim=2, max_w=self.freq_mask)
            x = self._mask(x, dim=3, max_w=self.time_mask)
        return self.inner(x)

def build_model(num_classes: int):
    try:
        backbone = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.DEFAULT
        )
    except Exception:
        backbone = models.efficientnet_b1(weights=None)
    
    # Freeze first 4 stages
    for stage_idx, stage in enumerate(backbone.features):
        if stage_idx < 4:
            for p in stage.parameters():
                p.requires_grad = False
    
    # Freeze BatchNorm running stats in frozen stages
    def _freeze_bn(m):
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for stage_idx in range(4):
        backbone.features[stage_idx].apply(_freeze_bn)
    
    # Replace final layer
    in_feats = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, num_classes)
    )
    
    # Wrap with SpecAugment
    model = SpecAugmentWrap(backbone)
    
    # Ensure channel compatibility
    model = _ensure_channel_compat(model, INPUT_SHAPE[0])
    
    return model
# --- AGENT_BUILD_MODEL_END ---


# ---------- metrics ----------
def _binarise(probs: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (probs >= thr).float()


def macro_f1(y_true: torch.Tensor, probs: torch.Tensor) -> float:
    yhat = _binarise(probs)
    eps = 1e-9
    tp = (yhat * y_true).sum(dim=0)
    fp = (yhat * (1 - y_true)).sum(dim=0)
    fn = ((1 - yhat) * y_true).sum(dim=0)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(f1.mean().item())


def macro_roc_auc(y_true: torch.Tensor, probs: torch.Tensor) -> float:
    """Manual ROC-AUC per class, mean over classes with both pos and neg samples."""
    aucs: list[float] = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c].numpy()
        ps = probs[:, c].numpy()
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        order = ps.argsort()[::-1]
        yt_sorted = yt[order]
        tp = 0
        fp = 0
        prev_score = None
        auc = 0.0
        pos = float(yt.sum())
        neg = float(len(yt) - pos)
        for i, _ in enumerate(yt_sorted):
            if yt_sorted[i] == 1:
                tp += 1
            else:
                auc += tp
                fp += 1
        if pos == 0 or neg == 0:
            continue
        aucs.append(auc / (pos * neg))
    if not aucs:
        return 0.0
    return float(sum(aucs) / len(aucs))


# ---------- training ----------
def lr_scheduler(opt, schedule, total_steps):
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    if schedule == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total_steps)
    return None


def _print(msg: str) -> None:
    """Single source of stdout writes — every line is flushed so the UI's
    live tail picks it up the moment the subprocess emits it."""
    print(msg, flush=True)


def _ensure_channel_compat(model: nn.Module, in_chan: int) -> nn.Module:
    """Bridge mel-spectrogram channel count to whatever the model expects.

    Mels are 1-channel. Audio-pretrained backbones almost always have a
    3-channel first conv (ImageNet weights). Without this adapter the
    first forward pass crashes with ``Given groups=1, weight of size [...,
    3, ...], expected input ... to have 3 channels``. Detection: walk the
    model in registration order until we hit the first ``nn.Conv2d``; if
    its ``in_channels != in_chan`` we wrap the model in a tiny
    ``Conv2d(in_chan -> expected, 1x1)`` adapter that the optimizer will
    learn alongside the rest of the network.
    """
    expected = _first_conv_in_channels(model)
    if expected is None or expected == in_chan:
        return model

    _print(
        f"[skeleton] wrapping model with Conv2d({in_chan} -> {expected}, 1x1) "
        f"to match the first conv's expected in_channels"
    )

    class _ChannelAdapter(nn.Module):
        def __init__(self, src: int, dst: int, inner: nn.Module):
            super().__init__()
            self.adapter = nn.Conv2d(src, dst, kernel_size=1, bias=False)
            # Initialise the adapter to "broadcast" the single channel
            # equally across the destination channels (close to the common
            # mel-as-RGB-replication trick) — gives the pretrained network
            # something close to its expected statistics on the first forward.
            with torch.no_grad():
                nn.init.constant_(self.adapter.weight, 1.0 / src)
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(self.adapter(x))

    return _ChannelAdapter(in_chan, expected, model)


def _first_conv_in_channels(model: nn.Module) -> int | None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module.in_channels
    return None


def main():
    train_loader, val_loader = make_loaders()
    _print(
        f"[loader] train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={BATCH_SIZE} num_workers={NUM_WORKERS} device={DEVICE}"
    )
    model = build_model(num_classes=NUM_CLASSES)
    model = _ensure_channel_compat(model, INPUT_SHAPE[0])
    if CHECKPOINT_IN and Path(CHECKPOINT_IN).exists():
        model.load_state_dict(torch.load(CHECKPOINT_IN, map_location="cpu"))
        _print(f"[init] warm-started from {CHECKPOINT_IN}")
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    _print(f"[model] params={n_params:,} num_classes={NUM_CLASSES}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, EPOCHS * len(train_loader))
    sched = lr_scheduler(optimizer, LR_SCHEDULE, total_steps)

    history = []
    best_score = float("-inf")
    no_improve = 0
    t0 = time.time()
    stopped_early = False

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()
        model.train()
        running = 0.0
        n = 0
        log_every = max(1, len(train_loader) // 5)
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.float().to(DEVICE)
            y = y.float().to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if sched is not None:
                sched.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
            if batch_idx % log_every == 0:
                _print(
                    f"[epoch {epoch}/{EPOCHS}] batch {batch_idx}/{len(train_loader)} "
                    f"running_loss={running / max(1, n):.4f}"
                )
        train_loss = running / max(1, n)

        # val
        model.eval()
        ys: list[torch.Tensor] = []
        ps: list[torch.Tensor] = []
        logits_acc: list[torch.Tensor] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.float().to(DEVICE)
                logits = model(x)
                logits_acc.append(logits.cpu())
                probs = torch.sigmoid(logits).cpu()
                ys.append(y.float())
                ps.append(probs)
        y_true = torch.cat(ys)
        probs = torch.cat(ps)
        # val_loss is the same BCE-with-logits we used for training, so the
        # train/val curves can be compared on the same y-scale and the user
        # can see overfitting (train down, val up) directly.
        val_loss = float(criterion(torch.cat(logits_acc), y_true).item())
        f1 = macro_f1(y_true, probs)
        auc = macro_roc_auc(y_true, probs)
        primary = auc if PRIMARY_METRIC == "roc_auc_macro" else f1
        history.append(
            {
                "epoch": epoch,
                # `loss` retained for back-compat with old reports + memory
                # snippets; new code should read `train_loss` / `val_loss`.
                "loss": float(train_loss),
                "train_loss": float(train_loss),
                "val_loss": val_loss,
                "f1_macro": float(f1),
                "roc_auc_macro": float(auc),
            }
        )

        improved = primary > best_score
        if improved:
            best_score = primary
            no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_OUT)
        else:
            no_improve += 1

        epoch_dt = time.time() - epoch_t0
        _print(
            f"[epoch {epoch}/{EPOCHS}] done in {epoch_dt:.1f}s "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"f1_macro={f1:.4f} roc_auc_macro={auc:.4f} "
            f"best={best_score:.4f} {'(NEW BEST)' if improved else f'(no improve {no_improve}/3)'}"
        )

        if not improved and no_improve >= 3:
            stopped_early = True
            _print("[stop] early stopping triggered (3 epochs without improvement)")
            break

    duration = time.time() - t0
    _print(
        f"[done] best {PRIMARY_METRIC}={best_score:.4f} duration={duration:.1f}s "
        f"epochs_run={len(history)}"
    )
    Path("results.json").write_text(
        json.dumps(
            {
                "primary_score": best_score,
                "primary_metric": PRIMARY_METRIC,
                "metrics": {
                    "f1_macro": float(history[-1]["f1_macro"]),
                    "roc_auc_macro": float(history[-1]["roc_auc_macro"]),
                },
                "history": history,
                "stopped_early": stopped_early,
                "duration_seconds": duration,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()