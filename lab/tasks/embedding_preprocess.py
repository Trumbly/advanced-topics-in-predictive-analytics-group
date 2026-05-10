"""Audio embedding extraction for the BirdCLEF task.

Loads a pretrained audio backbone (Perch / BirdNET) and emits one
``.npy`` embedding per 5-second window aligned with the existing
spectrogram cache. The output mirrors the lazy-mel layout so the
embedding skeleton can read it with the same ``train_index.json`` /
``val_index.json`` structure.

Backbones supported:
- ``perch``   Google bird-vocalization-classifier (TF-Hub). 1280-d output.
- ``birdnet`` Cornell BirdNET-Analyzer (TFLite). 1024-d output.

Both load the model once and stream-process the existing audio files.
TensorFlow / TFLite are imported lazily so the rest of the lab does not
depend on a TF install. Install:

    pip install 'lab[embeddings]'

or directly:

    pip install tensorflow-cpu tensorflow-hub librosa

If the backbone weights are not cacheable inside the lab repo, set
``LAB_BACKBONE_CACHE`` to a writable path; the helper resolves to
``~/.cache/lab-backbones`` by default.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from lab.config import Settings

_LOG = logging.getLogger("lab.preprocess.embedding")

_PERCH_TFHUB_URL = "https://tfhub.dev/google/bird-vocalization-classifier/8"
_BIRDNET_TFLITE_URL = (
    "https://github.com/kahst/BirdNET-Analyzer/raw/main/checkpoints/V2.4/"
    "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
)
_DEFAULT_SAMPLE_RATE = 32_000
_WINDOW_SECONDS = 5
_WINDOW_SAMPLES = _DEFAULT_SAMPLE_RATE * _WINDOW_SECONDS

_WINDOW_SUFFIX_RE = re.compile(r"_w(\d+)$")


def _backbone_cache_dir() -> Path:
    raw = os.environ.get("LAB_BACKBONE_CACHE") or "~/.cache/lab-backbones"
    return Path(raw).expanduser()


def extract_embeddings(
    settings: Settings,
    *,
    backbone: str,
    overwrite: bool = False,
    limit: int | None = None,
) -> tuple[Path, Path]:
    """Extract embeddings for every window referenced by the lazy mel index.

    Reads ``train_index.json`` + ``val_index.json`` from the mel processed
    dir, recovers the source audio path per sid, runs the backbone, and
    writes one ``<sid>.npy`` per window into
    ``data/processed/<backbone>_emb/``. Mirrors the index files so the
    embedding skeleton can pick them up.

    Returns the (train_index, val_index) paths in the embedding cache.
    Raises ``FileNotFoundError`` when the source mel cache is absent and
    ``RuntimeError`` when the backbone cannot be loaded.
    """
    backbone = backbone.lower()
    if backbone not in {"perch", "birdnet"}:
        raise ValueError(
            f"unknown backbone '{backbone}'; supported: perch, birdnet"
        )

    mel_dir = Path(settings.task.processed_data_dir)
    mel_train = mel_dir / "train_index.json"
    mel_val = mel_dir / "val_index.json"
    if not mel_train.exists() or not mel_val.exists():
        raise FileNotFoundError(
            f"missing mel lazy indexes at {mel_dir}; run `lab preprocess` "
            "first to materialise the spectrogram cache + train/val split"
        )

    out_dir = mel_dir.parent / f"{backbone}_emb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_dir / "train_index.json"
    out_val = out_dir / "val_index.json"

    extractor = _load_extractor(backbone)
    embedding_dim = extractor.embedding_dim

    audio_root = mel_dir.parent.parent / "raw"

    n_done = 0
    for split_in, split_out in ((mel_train, out_train), (mel_val, out_val)):
        payload = json.loads(split_in.read_text(encoding="utf-8"))
        new_samples: list[dict] = []
        for entry in payload.get("samples", []):
            sid = entry["sid"]
            target = out_dir / f"{sid}.npy"
            if target.exists() and not overwrite:
                new_samples.append(entry)
                n_done += 1
                continue
            audio_path = _resolve_audio_path(audio_root, sid)
            if audio_path is None:
                _LOG.warning("skipping %s: source audio not found", sid)
                continue
            try:
                window = _load_audio_window(audio_path, sid)
            except Exception as exc:  # noqa: BLE001 — backbone IO is varied
                _LOG.warning("skipping %s: audio load failed: %s", sid, exc)
                continue
            try:
                emb = extractor(window)
            except Exception as exc:  # noqa: BLE001 — TF errors aren't typed
                _LOG.warning("skipping %s: backbone forward failed: %s", sid, exc)
                continue
            _save_npy(target, emb)
            new_samples.append(entry)
            n_done += 1
            if limit is not None and n_done >= limit:
                break

        split_out.write_text(
            json.dumps(
                {
                    "embeddings_dir": str(out_dir.resolve()),
                    "embedding_dim": int(embedding_dim),
                    "num_classes": payload.get("num_classes"),
                    "samples": new_samples,
                }
            )
        )
        if limit is not None and n_done >= limit:
            break

    _LOG.info(
        "embedded %d windows with backbone=%s -> %s (dim=%d)",
        n_done,
        backbone,
        out_dir,
        embedding_dim,
    )
    return out_train, out_val


# ---------------------------------------------------------------------------
# extractor implementations
# ---------------------------------------------------------------------------


class _BaseExtractor:
    embedding_dim: int = 0

    def __call__(self, audio):  # pragma: no cover — overridden
        raise NotImplementedError


def _load_extractor(backbone: str) -> _BaseExtractor:
    if backbone == "perch":
        return _PerchExtractor()
    if backbone == "birdnet":
        return _BirdnetExtractor()
    raise ValueError(backbone)


class _PerchExtractor(_BaseExtractor):
    embedding_dim = 1280

    def __init__(self) -> None:
        try:
            import tensorflow as tf  # type: ignore[import-not-found]
            import tensorflow_hub as hub  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — env-dependent
            raise RuntimeError(
                "Perch needs `tensorflow` + `tensorflow_hub`. Install with "
                "`pip install tensorflow-cpu tensorflow-hub`."
            ) from exc

        cache = _backbone_cache_dir() / "tfhub"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TFHUB_CACHE_DIR", str(cache))
        self._tf = tf
        self._model = hub.load(_PERCH_TFHUB_URL)

    def __call__(self, audio):
        tf = self._tf
        x = tf.constant([audio], dtype=tf.float32)
        # Perch returns a dict; the embedding head is keyed
        # 'embedding'. Fall back to the first 1-D output if the schema
        # ever changes.
        out = self._model.infer_tf(x)
        if isinstance(out, dict):
            emb = out.get("embedding") or next(iter(out.values()))
        else:
            emb = out
        emb_np = emb.numpy().reshape(-1)
        return emb_np


class _BirdnetExtractor(_BaseExtractor):
    embedding_dim = 1024

    def __init__(self) -> None:
        try:
            import tensorflow as tf  # type: ignore[import-not-found]
            import urllib.request
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "BirdNET needs `tensorflow` (TFLite runtime). Install with "
                "`pip install tensorflow-cpu`."
            ) from exc

        cache = _backbone_cache_dir() / "birdnet"
        cache.mkdir(parents=True, exist_ok=True)
        tflite_path = cache / "birdnet_v2.4_fp32.tflite"
        if not tflite_path.exists():
            _LOG.info(
                "downloading BirdNET TFLite weights to %s", tflite_path
            )
            urllib.request.urlretrieve(_BIRDNET_TFLITE_URL, tflite_path)

        self._tf = tf
        self._interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self._interpreter.allocate_tensors()
        self._input_idx = self._interpreter.get_input_details()[0]["index"]
        # BirdNET surfaces both logits and the penultimate embedding; we
        # use the embedding tensor (largest non-logit output) as the feature.
        outputs = self._interpreter.get_output_details()
        self._output_idx = max(outputs, key=lambda d: d["shape"][-1])["index"]

    def __call__(self, audio):
        tf = self._tf
        x = tf.constant([audio], dtype=tf.float32).numpy()
        self._interpreter.set_tensor(self._input_idx, x)
        self._interpreter.invoke()
        emb = self._interpreter.get_tensor(self._output_idx)
        return emb.reshape(-1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_audio_path(audio_root: Path, sid: str) -> Path | None:
    """Map a sid back to the source audio file on disk.

    The BirdCLEF cache uses two prefix conventions:

    - ``iNat<id>_w<idx>``   -> ``train_audio/<class>/iNat<id>.ogg``
    - ``BC2026_Train_<id>`` -> ``train_soundscapes/<id>.ogg``
    - ``XC<id>_w<idx>``     -> ``train_audio/<class>/XC<id>.ogg``

    We walk the obvious roots; callers that pre-index audio paths can
    inject a faster lookup.
    """
    base = _WINDOW_SUFFIX_RE.sub("", sid)
    for sub in ("train_audio", "train_soundscapes"):
        root = audio_root / sub
        if not root.exists():
            continue
        # Two-level layout: <root>/<class>/<base>.ogg
        for ext in (".ogg", ".wav", ".flac"):
            for candidate in root.rglob(f"{base}{ext}"):
                return candidate
    return None


def _load_audio_window(audio_path: Path, sid: str):
    """Read the matching 5-second window of the source audio."""
    try:
        import librosa  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "embedding extraction needs `librosa`. Install with "
            "`pip install librosa`."
        ) from exc

    match = _WINDOW_SUFFIX_RE.search(sid)
    window_idx = int(match.group(1)) if match else 0
    start = window_idx * _WINDOW_SECONDS
    audio, _ = librosa.load(
        audio_path,
        sr=_DEFAULT_SAMPLE_RATE,
        offset=start,
        duration=_WINDOW_SECONDS,
        mono=True,
    )
    if audio.shape[0] < _WINDOW_SAMPLES:
        # Pad short tails with zeros (mirrors the mel pipeline).
        import numpy as np

        pad = np.zeros(_WINDOW_SAMPLES - audio.shape[0], dtype=audio.dtype)
        audio = np.concatenate([audio, pad])
    return audio[:_WINDOW_SAMPLES]


def _save_npy(path: Path, arr) -> None:
    import numpy as np

    np.save(path, np.asarray(arr, dtype="float32"))
