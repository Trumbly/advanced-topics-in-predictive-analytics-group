"""Build (and optionally publish) the offline-weights cache.

Reads backbones from ``registry/track_*_models.yaml``, downloads each one's
pretrained weights into a local directory that mirrors ``torch.hub`` /
``huggingface`` cache layouts, and (if ``--publish``) pushes the directory
as a Kaggle dataset so kernels can mount it without internet access.

Usage (from repo root):

    python scripts/build_weights_dataset.py                    # local only
    python scripts/build_weights_dataset.py --publish          # also push to Kaggle

The Kaggle slug is read from ``config/config.yaml`` (``executor.kaggle.weights_dataset``).
Re-running is idempotent: already-cached files are skipped. Use ``--versions``
to publish a new version of an already-created dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lab.config import load_settings  # noqa: E402


def _load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or []
    if isinstance(data, dict):
        data = data.get("models", [])
    return [e for e in data if isinstance(e, dict)]


def _download_torchvision(entry: dict, weights_dir: Path) -> bool:
    """Construct the torchvision backbone once so TORCH_HOME gets populated."""
    backbone = entry.get("backbone") or ""
    if not backbone.startswith("torchvision.models."):
        return False
    fn_name = backbone.rsplit(".", 1)[-1]
    weights_key = entry.get("weights") or "DEFAULT"
    os.environ["TORCH_HOME"] = str(weights_dir)
    try:
        import torch  # noqa: F401
        import torchvision.models as tvm
    except Exception as exc:  # noqa: BLE001
        print(f"  ! torch/torchvision not importable: {exc}", file=sys.stderr)
        return False
    ctor = getattr(tvm, fn_name, None)
    if ctor is None:
        print(f"  ! torchvision has no {fn_name}", file=sys.stderr)
        return False
    try:
        ctor(weights=weights_key)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! download failed for {fn_name}: {exc}", file=sys.stderr)
        return False
    return True


def _download_timm(entry: dict, weights_dir: Path) -> bool:
    name = entry.get("timm")
    if not name:
        return False
    hf_cache = weights_dir / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["TIMM_HOME"] = str(weights_dir)
    try:
        import timm  # noqa: PLC0415
    except Exception:
        # timm is optional — if it's not installed locally we just skip.
        return False
    try:
        timm.create_model(name, pretrained=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! timm download failed for {name}: {exc}", file=sys.stderr)
        return False
    return True


def _populate_weights_cache(registry_paths: list[Path], weights_dir: Path) -> int:
    weights_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for reg_path in registry_paths:
        entries = _load_registry(reg_path)
        for entry in entries:
            if entry.get("pretrained") is False:
                continue
            name = entry.get("name", "?")
            family = entry.get("family", "?")
            print(f"· {name} [{family}]")
            if _download_torchvision(entry, weights_dir):
                total += 1
            if _download_timm(entry, weights_dir):
                total += 1
    return total


def _write_dataset_metadata(slug: str, weights_dir: Path) -> Path:
    meta = {
        "title": slug.split("/", 1)[-1],
        "id": slug,
        "licenses": [{"name": "other"}],
    }
    path = weights_dir / "dataset-metadata.json"
    path.write_text(json.dumps(meta, indent=2))
    return path


def _run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=None,
                        help="Where to cache downloaded weights. "
                             "Default: paths.offline_weights from config.yaml.")
    parser.add_argument("--publish", action="store_true",
                        help="After downloading, `kaggle datasets create` / "
                             "`kaggle datasets version` to push to Kaggle.")
    parser.add_argument("--versions", action="store_true",
                        help="Publish as a new version of an existing dataset.")
    parser.add_argument("--notes", default="refresh offline weights",
                        help="Version notes (used with --versions).")
    args = parser.parse_args()

    settings = load_settings()
    weights_dir = Path(args.weights_dir) if args.weights_dir else settings.abspath(
        settings.paths.offline_weights,
    )
    registry_paths = [
        settings.abspath("registry/track_a_models.yaml"),
        settings.abspath("registry/track_b_models.yaml"),
    ]

    print(f"Populating {weights_dir} from:")
    for p in registry_paths:
        print(f"  - {p}")
    n = _populate_weights_cache(registry_paths, weights_dir)
    print(f"Downloaded {n} backbone(s). Cache size:")
    du = shutil.disk_usage(weights_dir)
    print(f"  free={du.free / 1e9:.1f} GB  total={du.total / 1e9:.1f} GB")

    if not args.publish and not args.versions:
        return 0

    slug = settings.executor.kaggle.weights_dataset
    if not slug:
        print("executor.kaggle.weights_dataset is empty in config.yaml — set it "
              "to `<username>/lab-agent-weights` before publishing.",
              file=sys.stderr)
        return 2

    _write_dataset_metadata(slug, weights_dir)
    cli = shutil.which("kaggle")
    if not cli:
        print("kaggle CLI not on PATH — `pip install kaggle` first.", file=sys.stderr)
        return 2

    if args.versions:
        return _run([cli, "datasets", "version", "-p", str(weights_dir),
                     "-m", args.notes, "--dir-mode", "zip"])
    return _run([cli, "datasets", "create", "-p", str(weights_dir), "--dir-mode", "zip"])


if __name__ == "__main__":
    raise SystemExit(main())
