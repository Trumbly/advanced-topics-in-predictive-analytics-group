"""Download BirdCLEF+ 2026 competition data from Kaggle.

Usage:
    python scripts/download_data.py                        # default: data/raw/
    python scripts/download_data.py --dest data/raw
    python scripts/download_data.py --competition birdclef-2026

Prerequisites:
    - kaggle CLI installed: `pip install kaggle`
    - Kaggle API credentials at ~/.kaggle/kaggle.json
    - User has accepted the BirdCLEF 2026 competition rules on kaggle.com

The script wraps `kaggle competitions download` and then unzips into the
target directory. We do NOT download anything on import — only when run
as a script.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download(competition: str, dest: Path) -> None:
    """Download and unzip the competition data."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Download
    _run(["kaggle", "competitions", "download", "-c", competition, "-p", str(dest)])

    # Unzip the top-level zip file
    zip_files = list(dest.glob(f"{competition}*.zip"))
    if not zip_files:
        raise FileNotFoundError(
            f"No zip file found in {dest} after kaggle download"
        )
    main_zip = zip_files[0]
    print(f"Unzipping {main_zip} -> {dest}")
    with zipfile.ZipFile(main_zip) as zf:
        zf.extractall(dest)

    print(f"Done. Files in {dest}:")
    for item in sorted(dest.iterdir()):
        print(f"  {item.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--competition",
        default="birdclef-2026",
        help="Kaggle competition slug",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw"),
        help="Target directory for the downloaded data",
    )
    args = parser.parse_args()

    if shutil.which("kaggle") is None:
        print(
            "ERROR: kaggle CLI not found. Install with `pip install kaggle` and "
            "place your API token at ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        return 1

    try:
        download(args.competition, args.dest)
    except subprocess.CalledProcessError as e:
        print(f"kaggle CLI failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
