"""Writes a results.json missing required keys."""
import json
from pathlib import Path

Path("results.json").write_text(json.dumps({"foo": 1}))
