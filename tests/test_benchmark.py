"""#41 acceptance: benchmark aggregates by (family, architecture)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab.config import load_settings
from lab.core.benchmark import benchmark
from lab.core.models import Experiment, Proposal, Study
from lab.ui.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


def _study(sid: str, scores: list[tuple[str, str, float]]) -> Study:
    exps: list[Experiment] = []
    for i, (family, arch, score) in enumerate(scores):
        exps.append(
            Experiment(
                id=f"exp_{i}",
                index=i,
                status="JUDGED",
                proposal=Proposal(
                    architecture_name=arch,
                    family=family,
                    lr=1e-3,
                    lr_schedule="cosine",
                    epochs=2,
                ),
                primary_metric="roc_auc_macro",
                primary_score=score,
            )
        )
    return Study(
        id=sid,
        task_name="track_b",
        status="COMPLETED",
        personality="exploratory",
        agent_memory_enabled=False,
        experiments=exps,
        created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )


def test_benchmark_aggregates_across_studies(tmp_path):
    _study("s1", [("cnn_scratch", "Tiny", 0.4), ("efficientnet_pretrained", "B0", 0.7)]).save(tmp_path)
    _study("s2", [("cnn_scratch", "Tiny", 0.5), ("efficientnet_pretrained", "B0", 0.65)]).save(tmp_path)
    rows = benchmark(tmp_path)
    by_arch = {(r.family, r.architecture_name): r for r in rows}
    cnn = by_arch[("cnn_scratch", "Tiny")]
    eff = by_arch[("efficientnet_pretrained", "B0")]
    assert cnn.runs == 2 and cnn.best_score == pytest.approx(0.5)
    assert eff.best_score == pytest.approx(0.7)
    # Sorted by best_score desc
    assert rows[0].family == "efficientnet_pretrained"


def test_cli_benchmark_runs(monkeypatch, capsys, tmp_path):
    """CLI calls benchmark; smoke that exit 0 with no studies returns the empty hint."""
    from lab.cli import main

    monkeypatch.chdir(REPO_ROOT)
    rc = main(["benchmark"])
    assert rc == 0


def test_ui_benchmark_renders(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(update={"experiments_dir": str(tmp_path / "studies")})
    s = s.model_copy(update={"paths": new_paths})
    Path(new_paths.experiments_dir).mkdir(parents=True)
    _study("s1", [("efficientnet_pretrained", "B0", 0.7)]).save(Path(new_paths.experiments_dir))

    client = TestClient(create_app(s))
    r = client.get("/benchmark")
    assert r.status_code == 200
    assert "B0" in r.text
    assert "efficientnet_pretrained" in r.text

    r = client.get("/api/benchmark")
    assert r.status_code == 200
    assert r.json()[0]["architecture_name"] == "B0"
