"""Summarise per-author file ownership across lab/, config/, scripts/, tests/, docs/.

Prints, for each known author, the top N files where they wrote the most
lines (per git blame). Used to seed §6 of the report; teammates still
confirm their own paragraph before sign-off.
"""
from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

AUTHORS = ["Trumbly", "danish-m-qureshi", "Lorry171717", "SebastianMis23"]
ROOTS = ["lab", "config", "scripts", "tests", "docs"]
TOP_N = 10

def main() -> None:
    files: list[str] = []
    for root in ROOTS:
        if not Path(root).is_dir():
            continue
        files += [str(p) for p in Path(root).rglob("*") if p.is_file()]
    per_author: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in files:
        try:
            out = subprocess.run(
                ["git", "blame", "--line-porcelain", f],
                capture_output=True, text=True, check=False,
            ).stdout
        except Exception:
            continue
        for line in out.splitlines():
            if line.startswith("author "):
                a = line[len("author "):].strip()
                if a in AUTHORS:
                    per_author[a][f] += 1
    for a in AUTHORS:
        print(f"\n## {a}")
        ranked = sorted(per_author[a].items(), key=lambda kv: -kv[1])[:TOP_N]
        for f, n in ranked:
            print(f"  {n:5d}  {f}")

if __name__ == "__main__":
    main()
