"""Tests for large-dataset sampling strategy configuration and data loading."""

from unittest.mock import patch

import pandas as pd
import pytest

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.exceptions import ConfigError
from treeshaker.mcp_server import _build_config, list_config_schema
from treeshaker.utils import load_csv


def test_large_dataset_sampling_config_defaults():
    """Verify large_dataset_sample_* fields default correctly in Config."""
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.large_dataset_sample_threshold is None
    assert config.large_dataset_sample_frac == 0.5


def test_large_dataset_sampling_config_validation():
    """Verify Config validation for sampling threshold and fraction bounds."""
    with pytest.raises(
        ConfigError, match="large_dataset_sample_threshold must be >= 1"
    ):
        Config(
            input_file="test.csv",
            target_column="target",
            task="binary",
            large_dataset_sample_threshold=0,
        )

    with pytest.raises(ConfigError, match="large_dataset_sample_frac must be in"):
        Config(
            input_file="test.csv",
            target_column="target",
            task="binary",
            large_dataset_sample_frac=0.0,
        )


def test_cli_large_dataset_sampling_flags():
    """Verify CLI flags set sampling threshold and fraction in parsed args."""
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
            "--large-dataset-sample-threshold",
            "50",
            "--large-dataset-sample-frac",
            "0.4",
        ],
    ):
        args = parse_arguments()
        assert args.large_dataset_sample_threshold == 50
        assert args.large_dataset_sample_frac == 0.4


def test_mcp_server_large_dataset_sampling():
    """Verify MCP server schema and _build_config support sampling parameters."""
    schema = list_config_schema()
    field_names = [f["name"] for f in schema["fields"]]
    assert "large_dataset_sample_threshold" in field_names
    assert "large_dataset_sample_frac" in field_names

    cfg = _build_config(
        input_path="data.csv",
        target_column="y",
        task="binary",
        output_path="out.csv",
        large_dataset_sample_threshold=100,
        large_dataset_sample_frac=0.3,
    )
    assert cfg.large_dataset_sample_threshold == 100
    assert cfg.large_dataset_sample_frac == 0.3


def test_load_csv_sampling_triggered(tmp_path):
    """Verify load_csv performs auto-stratified sampling when threshold exceeded."""
    csv_file = tmp_path / "large_data.csv"
    # Create 100 rows (80 zeros, 20 ones)
    df = pd.DataFrame(
        {
            "feature": list(range(100)),
            "target": [0] * 80 + [1] * 20,
        }
    )
    df.to_csv(csv_file, index=False)

    # With threshold 50 and frac 0.5, total sampled rows = 50.
    train_df, val_df, test_df = load_csv(
        filepath=str(csv_file),
        target_column="target",
        task="binary",
        test_size=0.2,
        val_size=0.2,
        random_seed=42,
        large_dataset_sample_threshold=50,
        large_dataset_sample_frac=0.5,
    )

    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows == 50
