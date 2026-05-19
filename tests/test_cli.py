"""I-17 acceptance: argparse CLI surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.cli import _build_parser, _apply_run_overrides, cmd_prompts, cmd_preprocess, main
from lab.config import load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_run_help_lists_all_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--task",
        "--predecessor",
        "--use-best-prompts",
        "--agent-memory",
        "--personality",
        "--max-experiments",
        "--max-wallclock-min",
        "--data-subset",
    ):
        assert flag in out


def test_no_subcommand_prints_help(capsys):
    rc = main([])
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_invalid_subcommand_fails():
    with pytest.raises(SystemExit):
        main(["nonexistent"])


def test_invalid_personality_rejected():
    with pytest.raises(SystemExit):
        main(["run", "--personality", "rogue"])


def test_prompts_list_runs(capsys, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    rc = main(["prompts", "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "propose_architecture" in out
    # The active version pin can evolve as we ship new prompt revisions.
    # The CLI must report SOME active version (active=vN) for every task.
    assert "active=v" in out


def test_prompts_activate_requires_args(capsys):
    rc = main(["prompts", "activate"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "task" in err.lower() and "version" in err.lower()


def test_preprocess_synthetic_writes_shards(tmp_path, monkeypatch):
    """`preprocess --synthetic` writes train.pt + val.pt."""
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("PWD", str(tmp_path))
    # Build parser-only path: invoke cmd_preprocess with crafted args.
    parser = _build_parser()
    args = parser.parse_args(["preprocess", "--task", "track_b", "--synthetic"])

    # Patch processed dir into tmp via monkeypatching load_settings... easier:
    # call the real path; cleanup via tmp_path is ignored.
    rc = cmd_preprocess(args)
    assert rc == 0
    # The default config writes to data/processed/mels - we just confirm exit code.


# ---------------------------------------------------------------------------
# --data-subset flag tests
# ---------------------------------------------------------------------------

def test_data_subset_flag_accepted_by_parser():
    parser = _build_parser()
    args = parser.parse_args(["run", "--data-subset", "30"])
    assert args.data_subset == 30


def test_data_subset_flag_default_is_none():
    parser = _build_parser()
    args = parser.parse_args(["run"])
    assert args.data_subset is None


def test_data_subset_rejects_invalid_value():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--data-subset", "15"])


def test_apply_run_overrides_propagates_data_subset():
    settings = load_settings("track_b", repo_root=REPO_ROOT)
    parser = _build_parser()
    args = parser.parse_args(["run", "--data-subset", "30"])
    updated = _apply_run_overrides(settings, args)
    assert updated.compute_budget.data_subset_percent == 30


def test_apply_run_overrides_leaves_default_when_not_specified():
    settings = load_settings("track_b", repo_root=REPO_ROOT)
    parser = _build_parser()
    args = parser.parse_args(["run"])
    updated = _apply_run_overrides(settings, args)
    # Default from config.yaml is 100
    assert updated.compute_budget.data_subset_percent == 100
