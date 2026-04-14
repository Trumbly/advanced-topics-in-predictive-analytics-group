"""Shared test fixtures.

Most of these tests were scaffolded with AI assistance, then reviewed and
pared down to what actually guards against regressions. See README for
the project policy on AI-assisted tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `lab` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
