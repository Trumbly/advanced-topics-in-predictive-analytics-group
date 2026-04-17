"""Per-task exploratory data analysis.

Run once per task via ``python -m lab eda --task <name>``. Produces a
compact markdown report at ``docs/eda/<task>.md`` (checked in) plus an
HTML sibling for the UI. The orchestrator auto-picks up the markdown as
a prompt slot, so the LLM sees dataset-level insights on the first
experiment of every study.

We keep the report dataset-aware but task-shape-agnostic: the adapter
hands us a :class:`DatasetProfile` and a reference to the raw data
directory; we derive class balance, size stats, and light distribution
info from there.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lab.config import Settings
from lab.core.models import DatasetProfile
from lab.tasks.base import TaskAdapter


@dataclass
class EdaReport:
    markdown: str
    html: str
    source_sha: str


def _sha_of_inputs(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        if p.exists():
            stat = p.stat()
            h.update(str(p).encode())
            h.update(str(stat.st_size).encode())
            h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()[:12]


def _eda_for_text(adapter: TaskAdapter, profile: DatasetProfile) -> tuple[str, list[Path]]:
    raw = adapter.task_cfg.get("data", {}).get("raw", {})
    inputs: list[Path] = []
    for v in raw.values():
        p = adapter.settings.abspath(v)
        inputs.append(p)
    lines = ["### Class balance", "", "- Binary (target=1 fraction):"]
    pos = profile.extras.get("positive_fraction")
    if isinstance(pos, (int, float)):
        lines.append(f"  - `{pos:.3f}` positive out of `{profile.num_train_samples}` train samples")
    else:
        lines.append("  - positive_fraction unavailable from profile")
    lines.append("")
    lines.append("### Token lengths")
    avg = profile.extras.get("avg_tokens_per_sample")
    if isinstance(avg, (int, float)):
        lines.append(f"- Average tokens per sample: `{avg:.1f}`")
    max_len = profile.extras.get("max_length")
    if isinstance(max_len, int):
        lines.append(f"- Configured max_length: `{max_len}`")
    return "\n".join(lines), inputs


def _eda_for_audio(adapter: TaskAdapter, profile: DatasetProfile) -> tuple[str, list[Path]]:
    raw = adapter.task_cfg.get("data", {}).get("raw", {})
    inputs = [adapter.settings.abspath(v) for v in raw.values()]
    lines: list[str] = ["### Shape", ""]
    lines.append(f"- Classes: `{profile.num_classes}`")
    lines.append(f"- Train samples: `{profile.num_train_samples}`")
    if profile.num_val_samples:
        lines.append(f"- Val samples: `{profile.num_val_samples}`")
    lines.append("")
    lines.append("### Class distribution")
    dist = profile.extras.get("class_counts") or profile.extras.get("label_counts")
    if isinstance(dist, dict) and dist:
        items = sorted(dist.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:15]
        lines.append("Top 15 most-represented classes:")
        for name, cnt in items:
            lines.append(f"- `{name}` — {cnt}")
        if len(dist) > 15:
            lines.append(f"- … ({len(dist) - 15} more)")
    else:
        lines.append("_class_counts not present in dataset profile (run `lab ...` to build profile)_")
    for k in ("sample_rate", "n_mels", "clip_seconds"):
        v = profile.extras.get(k)
        if v is not None:
            lines.append(f"- {k}: `{v}`")
    return "\n".join(lines), inputs


_EDA_BY_KIND: dict[str, Callable[[TaskAdapter, DatasetProfile], tuple[str, list[Path]]]] = {
    "text_classification_binary": _eda_for_text,
    "audio_multilabel": _eda_for_audio,
}


def build_report(adapter: TaskAdapter) -> EdaReport:
    profile = adapter.load_profile()
    fn = _EDA_BY_KIND.get(adapter.kind) or _eda_for_text
    body, inputs = fn(adapter, profile)
    sha = _sha_of_inputs(inputs)
    md_parts = [
        f"# EDA — {adapter.name} ({adapter.kind})",
        "",
        f"_Source hash: `{sha}`. Regenerate with `python -m lab eda --task {adapter.name}`._",
        "",
        body,
    ]
    md = "\n".join(md_parts).strip() + "\n"
    html = _markdown_to_bare_html(md)
    return EdaReport(markdown=md, html=html, source_sha=sha)


def write_report(settings: Settings, adapter: TaskAdapter) -> tuple[Path, Path]:
    report = build_report(adapter)
    eda_dir = settings.abspath(settings.paths.eda_dir)
    eda_dir.mkdir(parents=True, exist_ok=True)
    md_path = eda_dir / f"{adapter.name}.md"
    html_path = eda_dir / f"{adapter.name}.html"
    md_path.write_text(report.markdown)
    html_path.write_text(report.html)
    return md_path, html_path


def load_summary(settings: Settings, task_name: str) -> str:
    """Return the checked-in EDA markdown for a task, or an empty string."""
    md_path = settings.abspath(settings.paths.eda_dir) / f"{task_name}.md"
    if md_path.is_file():
        return md_path.read_text()
    return ""


def _markdown_to_bare_html(md: str) -> str:
    """Dumb escape-and-wrap — just enough for the UI preview page."""
    from html import escape
    buf = io.StringIO()
    buf.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>EDA</title>"
              "<style>body{font-family:ui-sans-serif,system-ui,sans-serif;max-width:860px;"
              "margin:2rem auto;padding:0 1rem;line-height:1.5} code{background:#eee;"
              "padding:.1em .3em;border-radius:3px} pre{background:#111;color:#eee;"
              "padding:1rem;border-radius:4px;overflow:auto}</style></head><body>")
    for line in md.splitlines():
        if line.startswith("### "):
            buf.write(f"<h3>{escape(line[4:])}</h3>")
        elif line.startswith("## "):
            buf.write(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("# "):
            buf.write(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("- "):
            buf.write(f"<li>{escape(line[2:])}</li>")
        elif not line.strip():
            buf.write("<br>")
        else:
            buf.write(f"<p>{escape(line)}</p>")
    buf.write("</body></html>")
    return buf.getvalue()


__all__ = ["EdaReport", "build_report", "write_report", "load_summary"]
