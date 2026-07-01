"""Tests for the multi-seed stability report (n_stability_runs / stability_seeds).

Covers config plumbing, validation, CLI wiring, MCP schema, and end-to-end
pipeline behavior: stability_score and stability_votes columns appear and have
the correct shape/values when n_stability_runs > 0.
"""

import pytest
from conftest import make_config

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.exceptions import ConfigError
from treeshaker.mcp_server import _build_config, list_config_schema
from treeshaker.pipeline import FeatureSelectionPipeline


def test_stability_config_defaults():
    cfg = Config(input_file="x.csv", target_column="y", task="binary")
    assert cfg.n_stability_runs == 0
    assert cfg.stability_seeds is None


def test_stability_runs_validation():
    with pytest.raises(ConfigError, match="n_stability_runs must be >= 0"):
        Config(
            input_file="x.csv",
            target_column="y",
            task="binary",
            n_stability_runs=-1,
        )


def test_stability_seeds_without_runs():
    with pytest.raises(ConfigError, match="stability_seeds has no effect"):
        Config(
            input_file="x.csv",
            target_column="y",
            task="binary",
            n_stability_runs=0,
            stability_seeds=[1, 2],
        )


def test_stability_seeds_wrong_length():
    with pytest.raises(ConfigError, match="must have exactly n_stability_runs=2"):
        Config(
            input_file="x.csv",
            target_column="y",
            task="binary",
            n_stability_runs=2,
            stability_seeds=[1],
        )


def test_stability_seeds_correct_length():
    cfg = Config(
        input_file="x.csv",
        target_column="y",
        task="binary",
        n_stability_runs=2,
        stability_seeds=[10, 20],
    )
    assert cfg.stability_seeds == [10, 20]


def test_cli_n_stability_runs():
    args = parse_arguments.__wrapped__(
        [
            "--input", "a.csv", "--target", "t", "--task", "binary",
            "--n-stability-runs", "3",
        ]
    ) if hasattr(parse_arguments, "__wrapped__") else None
    # parse_arguments reads sys.argv — use monkeypatching via pytest
    import sys
    old = sys.argv
    sys.argv = [
        "treeshaker",
        "--input", "a.csv",
        "--target", "t",
        "--task", "binary",
        "--n-stability-runs", "3",
        "--stability-seeds", "1", "2", "3",
    ]
    try:
        args = parse_arguments()
        assert args.n_stability_runs == 3
        assert args.stability_seeds == [1, 2, 3]
    finally:
        sys.argv = old


def test_mcp_schema_has_n_stability_runs():
    schema = list_config_schema()
    names = {f["name"] for f in schema["fields"]}
    assert "n_stability_runs" in names
    assert "stability_seeds" in names


def test_mcp_build_config_n_stability_runs():
    cfg = _build_config(
        input_path="x.csv",
        target_column="y",
        task="binary",
        output_path="out",
        n_stability_runs=2,
    )
    assert cfg.n_stability_runs == 2


def test_stability_columns_absent_when_disabled(binary_csv, tmp_path):
    cfg = make_config(binary_csv, "binary", tmp_path, skip_tuning=True)
    results = FeatureSelectionPipeline(cfg).run()
    assert "stability_score" not in results.columns
    assert "stability_votes" not in results.columns


def test_stability_columns_present_when_enabled(binary_csv, tmp_path):
    cfg = make_config(
        binary_csv,
        "binary",
        tmp_path,
        skip_tuning=True,
        n_stability_runs=2,
        stability_seeds=[100, 200],
    )
    results = FeatureSelectionPipeline(cfg).run()

    assert "stability_score" in results.columns
    assert "stability_votes" in results.columns
    assert len(results) > 0

    # stability_score is a float in [0, 1]
    assert results["stability_score"].between(0.0, 1.0).all()

    # stability_votes is a non-empty string like "Keep:2,Drop:1"
    for votes in results["stability_votes"]:
        assert isinstance(votes, str)
        assert ":" in votes

    # With n_stability_runs=2, total votes per feature = 3 (main + 2 extra).
    import re
    for votes in results["stability_votes"]:
        total = sum(int(n) for n in re.findall(r":(\d+)", votes))
        assert total == 3


def test_stability_score_unanimous_when_seeds_identical(binary_csv, tmp_path):
    """If all seeds yield the same recommendation, stability_score should be 1.0."""
    # Use the same explicit seed twice — deterministic runs should agree.
    cfg = make_config(
        binary_csv,
        "binary",
        tmp_path,
        skip_tuning=True,
        n_stability_runs=2,
        stability_seeds=[42, 42],
    )
    results = FeatureSelectionPipeline(cfg).run()
    # All three runs (main at seed 42, extra 42, extra 42) are deterministic
    # → every feature should be unanimous.
    assert (results["stability_score"] == 1.0).all()
