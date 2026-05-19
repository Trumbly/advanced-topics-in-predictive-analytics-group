"""Report generator: figures + Jinja2 markdown + HTML."""

from __future__ import annotations

import base64
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape

from lab.config import Settings
from lab.core.loaders import list_studies, load_many
from lab.core.models import Study
from lab.reporting.figures import render_figures
from lab.ui.display import build_study_ordinals, exp_label, study_label

_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Inline stylesheet shipped with every report.html so the file stays
# self-contained (viewable directly from disk, mailable, archive-friendly).
_REPORT_CSS = """
:root { color-scheme: light dark; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  max-width: 900px;
  margin: 32px auto;
  padding: 0 24px 64px;
  line-height: 1.55;
  color: #1d2433;
  background: #ffffff;
}
h1, h2, h3 { line-height: 1.25; }
h1 { border-bottom: 1px solid #e3e6eb; padding-bottom: 0.3em; }
h2 { border-bottom: 1px solid #eceef2; padding-bottom: 0.2em; margin-top: 2em; }
h3 { margin-top: 1.5em; }
code, pre { font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 0.92em; }
pre {
  background: #f4f6fa;
  padding: 12px 14px;
  border-radius: 6px;
  overflow-x: auto;
}
table {
  border-collapse: collapse;
  margin: 12px 0 18px;
  width: 100%;
  font-size: 0.95em;
}
th, td {
  border: 1px solid #d8dde3;
  padding: 6px 10px;
  text-align: left;
}
th { background: #f4f6fa; }
tbody tr:nth-child(even) td { background: #fbfcfe; }
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 12px 0;
  border: 1px solid #e3e6eb;
  border-radius: 4px;
}
ul, ol { padding-left: 1.4em; }
li { margin: 3px 0; }
blockquote {
  border-left: 4px solid #3a6cd6;
  margin: 0;
  padding: 4px 14px;
  color: #4a5266;
  background: #f4f6fa;
}
@media (prefers-color-scheme: dark) {
  body { background: #11151b; color: #e4e8ef; }
  h1 { border-color: #2a313b; }
  h2 { border-color: #232932; }
  th, td { border-color: #2a313b; }
  th { background: #1a1f27; }
  tbody tr:nth-child(even) td { background: #161b22; }
  pre { background: #1a1f27; }
  blockquote { background: #1a1f27; color: #b8c0cd; border-color: #3a6cd6; }
  img { border-color: #2a313b; }
}
"""


def generate_report(study: Study, settings: Settings) -> Path:
    """Render the full study report (markdown + HTML) and return the markdown path."""
    out_dir = Path(settings.paths.experiments_dir) / study.id
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = render_figures(study, out_dir)
    figures_relative = {k: v.name for k, v in figures.items()}

    experiments_root = Path(settings.paths.experiments_dir)
    all_studies = load_many(experiments_root, list_studies(experiments_root))
    ordinals = build_study_ordinals(all_studies)
    ord_n = ordinals.get(study.id, len(ordinals) + 1)

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape([]),
        trim_blocks=False,
        lstrip_blocks=False,
    )
    template = env.get_template("report.md.j2")

    md = template.render(
        study=study,
        study_label=study_label(study.id, ord_n),
        exp_label_for=exp_label,
        figures=figures_relative,
        top_experiments=_top_experiments(study, n=5),
        failure_breakdown=_failure_breakdown(study),
        executive_summary=_summary(study),
    )

    md_path = out_dir / "report.md"
    md_path.write_text(md, encoding="utf-8")

    html_path = out_dir / "report.html"
    try:
        from markdown_it import MarkdownIt  # type: ignore[import-not-found]

        # `commonmark` (the default profile) does not render pipe tables.
        # Enable the `table` extension so the "Top experiments" block in
        # report.md.j2 turns into a real <table>.
        body = MarkdownIt("commonmark").enable("table").render(md)
    except ImportError:  # pragma: no cover - optional dep
        body = f"<pre>{md}</pre>"

    # Inline the generated figures as base64 data URIs so the report.html is
    # a single self-contained file (mail it, archive it, open it from disk
    # without a web server). Avoids 404s that would otherwise happen because
    # the UI route /reports/<study_id> serves a single resource and has no
    # accompanying static-files mount for the per-study figure pngs.
    body = _inline_images(body, out_dir)

    html_path.write_text(_wrap_html(body, study.id), encoding="utf-8")

    return md_path


# ---------------------------------------------------------------------------
# html helpers
# ---------------------------------------------------------------------------


_IMG_SRC_RE = re.compile(r'<img\s+([^>]*?)src="([^"]+)"', re.IGNORECASE)
_EXT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "svg": "image/svg+xml",
    "webp": "image/webp",
}


def _inline_images(html: str, base_dir: Path) -> str:
    """Replace `<img src="some-file.png">` with a base64 data URI so the
    rendered HTML stays self-contained when served from any URL."""

    def _repl(match: re.Match[str]) -> str:
        attrs = match.group(1)
        src = match.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        target = (base_dir / src).resolve()
        if not target.exists():
            return match.group(0)
        ext = target.suffix.lower().lstrip(".")
        mime = _EXT_TO_MIME.get(ext)
        if mime is None:
            return match.group(0)
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
        return f'<img {attrs}src="data:{mime};base64,{encoded}"'

    return _IMG_SRC_RE.sub(_repl, html)


def _wrap_html(body: str, study_id: str) -> str:
    """Wrap the markdown body in a full HTML document with inline CSS so
    the file renders cleanly when opened directly (no web-server CSS load,
    no missing-stylesheet flash)."""
    return (
        "<!doctype html>\n"
        f"<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>Study report — {study_id}</title>"
        f"<style>{_REPORT_CSS}</style>"
        f"</head><body>\n{body}\n</body></html>\n"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _top_experiments(study: Study, *, n: int = 5) -> list:
    successes = [e for e in study.experiments if e.primary_score is not None]
    successes.sort(key=lambda e: e.primary_score or 0.0, reverse=True)
    return successes[:n]


def _failure_breakdown(study: Study) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for exp in study.experiments:
        if exp.primary_score is not None:
            continue
        for task in exp.tasks:
            if task.error is not None:
                counts[task.error.error_type] += 1
                break
    return dict(counts)


def _summary(study: Study) -> str:
    if not study.experiments:
        return "No experiments were recorded for this study."

    completed = [e for e in study.experiments if e.primary_score is not None]
    if not completed:
        return f"Study completed {len(study.experiments)} experiments; none produced a primary score."

    best = max(completed, key=lambda e: e.primary_score or 0.0)
    family = best.proposal.family if best.proposal else "unknown"
    return (
        f"Study completed {len(study.experiments)} experiments "
        f"({len(completed)} scored). Best so far: {best.id} "
        f"({family}) at {best.primary_metric}={best.primary_score:.4f}."
    )
