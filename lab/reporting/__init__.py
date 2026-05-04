"""Reporting: figures + Jinja2 markdown + optional HTML conversion."""
from lab.reporting.figures import render_figures
from lab.reporting.generator import generate_report

__all__ = ["render_figures", "generate_report"]
