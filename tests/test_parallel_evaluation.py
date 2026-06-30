"""Tests for opt-in parallel feature evaluation configuration and wiring."""

from unittest.mock import patch

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.mcp_server import _build_config, list_config_schema


def test_parallel_feature_evaluation_config_defaults():
    """Verify parallel_feature_evaluation/n_jobs default to off/-1 in Config."""
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.parallel_feature_evaluation is False
    assert config.n_jobs == -1


def test_cli_parallel_feature_evaluation_flags():
    """Verify --parallel-feature-evaluation and --n-jobs parse correctly."""
    with patch(
        "sys.argv",
        [
            "treeshaker",
            "--input",
            "data.csv",
            "--target",
            "y",
            "--task",
            "binary",
            "--parallel-feature-evaluation",
            "--n-jobs",
            "4",
        ],
    ):
        args = parse_arguments()
        assert args.parallel_feature_evaluation is True
        assert args.n_jobs == 4


def test_cli_parallel_feature_evaluation_default_off():
    """Verify the flag is off and n_jobs is -1 when not passed."""
    with patch(
        "sys.argv",
        ["treeshaker", "--input", "data.csv", "--target", "y", "--task", "binary"],
    ):
        args = parse_arguments()
        assert args.parallel_feature_evaluation is False
        assert args.n_jobs == -1


def test_mcp_server_parallel_feature_evaluation():
    """Verify MCP server schema and _build_config support the new fields."""
    schema = list_config_schema()
    field_names = [f["name"] for f in schema["fields"]]
    assert "parallel_feature_evaluation" in field_names
    assert "n_jobs" in field_names

    cfg = _build_config(
        input_path="data.csv",
        target_column="y",
        task="binary",
        output_path="out.csv",
        parallel_feature_evaluation=True,
        n_jobs=2,
    )
    assert cfg.parallel_feature_evaluation is True
    assert cfg.n_jobs == 2
