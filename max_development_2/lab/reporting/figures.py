"""Matplotlib figures for the study report.

Every axis label uses ``study.primary_metric`` rather than a hardcoded
string — the same function renders ROC-AUC figures for Track B and F1
figures for Track A with no branching.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from lab.core.models import Study


def render_all(study: Study, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: list[Path] = []
    figs.append(_score_progression(study, out_dir / "score_progression.png"))
    figs.append(_failure_breakdown(study, out_dir / "failure_breakdown.png"))
    curve = _best_learning_curve(study, out_dir / "best_learning_curve.png")
    if curve:
        figs.append(curve)
    family = _per_family_box(study, out_dir / "per_family_box.png")
    if family:
        figs.append(family)
    return figs


def _score_progression(study: Study, path: Path) -> Path:
    import matplotlib.pyplot as plt

    metric = study.primary_metric or "score"
    xs = list(range(1, len(study.experiments) + 1))
    ys = [e.primary_score if e.primary_score is not None else float("nan")
          for e in study.experiments]
    colors = ["#4ade80" if e.primary_score is not None else "#f87171"
              for e in study.experiments]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(xs, ys, c=colors, s=60)
    ax.plot(xs, ys, color="#94a3b8", alpha=0.4)
    ax.set_xlabel("experiment #")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per experiment")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _failure_breakdown(study: Study, path: Path) -> Path:
    import matplotlib.pyplot as plt

    counter: Counter[str] = Counter()
    for e in study.experiments:
        if e.error:
            counter[e.error.error_type] += 1
        elif e.primary_score is not None:
            counter["success"] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    items = counter.most_common()
    labels = [k for k, _ in items] or ["no data"]
    values = [v for _, v in items] or [0]
    colors = ["#4ade80" if k == "success" else "#f97316" for k in labels]
    ax.bar(labels, values, color=colors)
    ax.set_title("Experiment outcomes")
    ax.set_ylabel("count")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _best_learning_curve(study: Study, path: Path) -> Path | None:
    if not study.best_experiment_id:
        return None
    best = next((e for e in study.experiments if e.id == study.best_experiment_id), None)
    if best is None or not best.history or len(best.history) < 2:
        return None

    import matplotlib.pyplot as plt

    metric = study.primary_metric
    xs = [h.get("epoch", i + 1) for i, h in enumerate(best.history)]
    ys = [h.get(metric) for h in best.history]
    loss = [h.get("loss") for h in best.history]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(xs, ys, marker="o", color="#60a5fa")
    a1.set_title(f"Best — {metric}")
    a1.set_xlabel("epoch")
    a1.set_ylabel(metric)
    a1.grid(True, alpha=0.2)

    a2.plot(xs, loss, marker="o", color="#f87171")
    a2.set_title("Best — training loss")
    a2.set_xlabel("epoch")
    a2.set_ylabel("loss")
    a2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _per_family_box(study: Study, path: Path) -> Path | None:
    import matplotlib.pyplot as plt

    by_family: dict[str, list[float]] = defaultdict(list)
    for e in study.experiments:
        if e.primary_score is None:
            continue
        fam = e.architecture_family or "unknown"
        by_family[fam].append(e.primary_score)
    if not by_family:
        return None

    labels = sorted(by_family)
    data = [by_family[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, labels=labels, vert=True, patch_artist=True)
    ax.set_title(f"{study.primary_metric} by architecture family")
    ax.set_ylabel(study.primary_metric)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


__all__ = ["render_all"]
