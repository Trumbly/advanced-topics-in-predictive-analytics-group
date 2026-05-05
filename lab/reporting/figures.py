"""Per-study figures: score progression, learning curve, failure breakdown,
family performance, per-class AUC.

Rendered with matplotlib (matches the pyproject dependency). Each figure
returns a path to a PNG written under ``out_dir``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from lab.core.models import Study


def render_figures(study: Study, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["score_progression"] = _score_progression(study, out_dir)

    best = _best_experiment(study)
    if best is not None and best.history:
        paths["best_learning_curve"] = _best_learning_curve(study, best, out_dir)

    paths["failure_breakdown"] = _failure_breakdown(study, out_dir)
    paths["family_performance"] = _family_performance(study, out_dir)

    if best is not None:
        per_class = best.metrics.get("per_class_auc")
        if isinstance(per_class, list) and per_class:
            paths["per_class_auc"] = _per_class_auc(per_class, out_dir)

    return paths


# ---------------------------------------------------------------------------
# individual figures
# ---------------------------------------------------------------------------


def _score_progression(study: Study, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = []
    ys = []
    annotations = []
    for exp in sorted(study.experiments, key=lambda e: e.index):
        if exp.primary_score is None:
            continue
        xs.append(exp.index)
        ys.append(exp.primary_score)
        annotations.append(exp.proposal.family if exp.proposal else "")

    if xs:
        ax.plot(xs, ys, marker="o")
        for x, y, label in zip(xs, ys, annotations):
            ax.annotate(label, (x, y), fontsize=7, alpha=0.7)
    ax.set_xlabel("experiment index")
    ax.set_ylabel(f"primary_score ({study.experiments[0].primary_metric if study.experiments else ''})")
    ax.set_title("Score progression")
    ax.grid(True, alpha=0.3)
    path = out_dir / "score_progression.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _best_learning_curve(study: Study, best, out_dir: Path) -> Path:
    history = best.history or []
    epochs = [h.get("epoch", i) for i, h in enumerate(history, start=1)]
    losses = [h.get("loss") for h in history]
    metric = best.primary_metric
    metric_values = [h.get(metric) for h in history]

    fig, ax_loss = plt.subplots(figsize=(8, 4))
    ax_metric = ax_loss.twinx()

    if any(l is not None for l in losses):
        ax_loss.plot(epochs, losses, "C0-o", label="loss")
    if any(m is not None for m in metric_values):
        ax_metric.plot(epochs, metric_values, "C1-s", label=metric)

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_metric.set_ylabel(metric)
    fig.suptitle(f"Best ({best.id}) learning curve")
    fig.tight_layout()
    path = out_dir / "best_learning_curve.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _failure_breakdown(study: Study, out_dir: Path) -> Path:
    counts: Counter[str] = Counter()
    for exp in study.experiments:
        if exp.primary_score is not None:
            continue
        for task in exp.tasks:
            if task.error is not None:
                counts[task.error.error_type] += 1
                break

    fig, ax = plt.subplots(figsize=(5, 5))
    if counts:
        ax.pie(list(counts.values()), labels=list(counts.keys()), autopct="%.0f%%")
        ax.set_title("Failure breakdown")
    else:
        ax.text(0.5, 0.5, "No failures", ha="center", va="center")
        ax.set_axis_off()
    path = out_dir / "failure_breakdown.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _family_performance(study: Study, out_dir: Path) -> Path:
    by_family: dict[str, list[float]] = defaultdict(list)
    for exp in study.experiments:
        if exp.primary_score is None or not exp.proposal:
            continue
        by_family[exp.proposal.family].append(exp.primary_score)

    fig, ax = plt.subplots(figsize=(8, 4))
    if by_family:
        labels = list(by_family.keys())
        data = [by_family[l] for l in labels]
        ax.boxplot(data, tick_labels=labels)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("primary_score")
    else:
        ax.text(0.5, 0.5, "No completed experiments", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Performance by architecture family")
    fig.tight_layout()
    path = out_dir / "family_performance.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _per_class_auc(values: list[float], out_dir: Path) -> Path:
    sorted_vals = sorted(values, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(sorted_vals)), sorted_vals)
    ax.set_xlabel("class index (sorted)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Per-class AUC (best experiment)")
    fig.tight_layout()
    path = out_dir / "per_class_auc.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _best_experiment(study: Study):
    successes = [e for e in study.experiments if e.primary_score is not None]
    if not successes:
        return None
    return max(successes, key=lambda e: e.primary_score)
