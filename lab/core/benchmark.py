"""Cross-study model benchmarking.

Aggregates the best score per (architecture_family, architecture_name) over
every saved study, so the team can answer 'which architectures actually work
on this task?'.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean

from pydantic import BaseModel

from lab.core.loaders import list_studies, load_many


class BenchmarkRow(BaseModel):
    family: str
    architecture_name: str
    runs: int
    mean_score: float
    best_score: float
    studies: list[str]


def benchmark(experiments_dir: Path) -> list[BenchmarkRow]:
    bucket: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for study in load_many(Path(experiments_dir), list_studies(experiments_dir)):
        for exp in study.experiments:
            if exp.primary_score is None or exp.proposal is None:
                continue
            key = (exp.proposal.family, exp.proposal.architecture_name)
            bucket[key].append((float(exp.primary_score), study.id))

    rows: list[BenchmarkRow] = []
    for (family, arch), entries in bucket.items():
        scores = [s for s, _ in entries]
        rows.append(
            BenchmarkRow(
                family=family,
                architecture_name=arch,
                runs=len(scores),
                mean_score=fmean(scores),
                best_score=max(scores),
                studies=sorted({sid for _, sid in entries}),
            )
        )
    rows.sort(key=lambda r: r.best_score, reverse=True)
    return rows
