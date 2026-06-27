"""Tests for the CLI interface."""

from unittest.mock import patch

import pandas as pd

from treeshaker.cli import main, parse_arguments


def test_parse_arguments_required():
    with patch(
        "sys.argv",
        [
            "treeshaker",
            "--input",
            "f.csv",
            "--target",
            "y",
            "--task",
            "binary",
        ],
    ):
        args = parse_arguments()
    assert args.input == "f.csv"
    assert args.target == "y"
    assert args.task == "binary"


def test_parse_arguments_defaults():
    with patch(
        "sys.argv",
        [
            "treeshaker",
            "--input",
            "f.csv",
            "--target",
            "y",
            "--task",
            "regression",
        ],
    ):
        args = parse_arguments()
    assert args.output == "treeshaker_results.csv"
    assert args.output_format == "csv"
    assert args.optuna_trials == 50
    assert args.shap_threshold == 0.01
    assert args.perm_threshold == 0.05
    assert args.test_size == 0.2
    assert args.val_size == 0.2
    assert args.random_seed == 42
    assert not args.verbose


def test_main_invalid_json_xgboost_params(tmp_path):
    with patch(
        "sys.argv",
        [
            "treeshaker",
            "--input",
            "f.csv",
            "--target",
            "y",
            "--task",
            "binary",
            "--xgboost-params",
            "not-json",
        ],
    ):
        result = main()
    assert result == 1


def test_main_end_to_end(binary_csv, tmp_path):
    output = str(tmp_path / "results.csv")
    with patch(
        "sys.argv",
        [
            "treeshaker",
            "--input",
            binary_csv,
            "--target",
            "target",
            "--task",
            "binary",
            "--output",
            output,
            "--optuna-trials",
            "3",
            "--optuna-timeout",
            "30",
        ],
    ):
        result = main()
    assert result == 0
    df = pd.read_csv(output)
    assert "feature_name" in df.columns
    assert "recommendation" in df.columns
    assert len(df) == 10
