"""Per-study EDA step (ADR-017).

Runs once at study start, reads the adapter's dataset profile, and emits a
compact (<=4 kB) markdown summary that is then injected as the ``{eda_summary}``
slot for every ``propose_architecture`` prompt during the study.
"""

from __future__ import annotations

from lab.config import Settings
from lab.core.models import EDAReport
from lab.tasks.base import TaskAdapter

_LOW_SAMPLE_THRESHOLD = 25
_MAX_MD_BYTES = 4096


def run_eda(adapter: TaskAdapter, settings: Settings) -> EDAReport:
    profile = adapter.profile()

    counts = profile.class_imbalance or {}
    if counts:
        c_max = max(counts.values())
        c_min = max(1, min(counts.values()))  # avoid div-by-zero
        imbalance_ratio = c_max / c_min
        top_class, top_count = max(counts.items(), key=lambda kv: kv[1])
        bottom_class, bottom_count = min(counts.items(), key=lambda kv: kv[1])
        long_tail = [k for k, v in counts.items() if v < _LOW_SAMPLE_THRESHOLD]
    else:
        imbalance_ratio = 1.0
        top_class = bottom_class = ""
        top_count = bottom_count = profile.num_train // max(1, profile.num_classes)
        long_tail = []

    notes: list[str] = []
    if long_tail:
        notes.append(
            f"{len(long_tail)} classes have <{_LOW_SAMPLE_THRESHOLD} samples; "
            "consider augmentation + class-weighted loss."
        )
    if imbalance_ratio > 10.0:
        notes.append(
            f"Class imbalance is severe ({imbalance_ratio:.1f}x); "
            "macro metrics will be dominated by long-tail classes."
        )

    md_lines: list[str] = []
    md_lines.append(f"# EDA - {settings.task_name}")
    md_lines.append(f"- Train samples: {profile.num_train}")
    md_lines.append(f"- Classes: {profile.num_classes}")
    if counts:
        md_lines.append(
            f"- Class imbalance (max/min): {imbalance_ratio:.1f}x  "
            f"(top class: {top_count} samples; bottom: {bottom_count})"
        )
    md_lines.append(f"- Input tensor shape: {tuple(profile.input_tensor_shape)}")
    if notes:
        md_lines.append("- Notable: " + " ".join(notes))

    md = "\n".join(md_lines).rstrip() + "\n"
    if len(md.encode("utf-8")) > _MAX_MD_BYTES:
        md = md.encode("utf-8")[:_MAX_MD_BYTES].decode("utf-8", errors="ignore")

    return EDAReport(
        markdown=md,
        num_classes=profile.num_classes,
        num_train=profile.num_train,
        imbalance_ratio=float(imbalance_ratio),
        input_tensor_shape=tuple(profile.input_tensor_shape),
        notes=notes,
    )
