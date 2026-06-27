"""Tests for the CLI interface."""

from unittest.mock import patch

import pandas as pd
import pytest

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
    assert args.n_permutations == 5
    assert args.permutation_metric is None


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


def test_parse_arguments_n_permutations_and_permutation_metric():
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
            "--n-permutations",
            "3",
            "--permutation-metric",
            "wape",
        ],
    ):
        args = parse_arguments()
    assert args.n_permutations == 3
    assert args.permutation_metric == "wape"


def test_parse_arguments_invalid_permutation_metric_rejected():
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
            "--permutation-metric",
            "bogus",
        ],
    ):
        with pytest.raises(SystemExit):
            parse_arguments()


def test_main_end_to_end_with_n_permutations_and_metric(binary_csv, tmp_path):
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
            "--n-permutations",
            "2",
            "--permutation-metric",
            "roc_auc",
        ],
    ):
        result = main()
    assert result == 0
    df = pd.read_csv(output)
    assert "permutation_importance_std" in df.columns


def test_parse_arguments_shap_sample_size_default_none():
    with patch(
        "sys.argv",
        ["treeshaker", "--input", "f.csv", "--target", "y", "--task", "regression"],
    ):
        args = parse_arguments()
    assert args.shap_sample_size is None


def test_parse_arguments_shap_sample_size():
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
            "--shap-sample-size",
            "100",
        ],
    ):
        args = parse_arguments()
    assert args.shap_sample_size == 100


def test_main_end_to_end_with_shap_sample_size(binary_csv, tmp_path):
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
            "--shap-sample-size",
            "5",
        ],
    ):
        result = main()
    assert result == 0
    df = pd.read_csv(output)
    assert "shap_importance" in df.columns
    assert len(df) == 10
