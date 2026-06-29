"""Tests for MLflow experiment-tracking export."""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.pipeline import log_run_to_mlflow


def test_use_mlflow_config_default():
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.use_mlflow is False
    assert config.mlflow_experiment_name == "treeshaker"


def test_use_mlflow_cli_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "treeshaker",
            "--input",
            "test.csv",
            "--target",
            "target",
            "--task",
            "binary",
            "--use-mlflow",
            "--mlflow-experiment-name",
            "my-experiment",
        ],
    )
    args = parse_arguments()
    assert args.use_mlflow is True
    assert args.mlflow_experiment_name == "my-experiment"


def test_log_run_to_mlflow_mocked(monkeypatch):
    mock_mlflow = MagicMock()
    run_ctx = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = run_ctx
    monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)

    config = Config(
        input_file="test.csv", target_column="target", task="binary", use_mlflow=True
    )
    results = pd.DataFrame(
        {
            "feature_name": ["feat1", "feat2"],
            "shap_importance": [0.5, 0.1],
            "recommendation": ["Keep", "Drop"],
        }
    )

    log_run_to_mlflow(config=config, results=results, experiment_name="treeshaker")

    mock_mlflow.set_experiment.assert_called_once_with("treeshaker")
    mock_mlflow.log_params.assert_called_once()
    logged_params = mock_mlflow.log_params.call_args.args[0]
    assert logged_params["task"] == "binary"
    assert logged_params["use_mlflow"] is True
    # Nested dataclass fields (e.g. optuna_config) shouldn't be passed as
    # flat MLflow params -- they aren't scalar/JSON-friendly.
    assert "optuna_config" not in logged_params

    assert mock_mlflow.log_metric.call_count == 2
    logged_metric_names = {
        call.args[0] for call in mock_mlflow.log_metric.call_args_list
    }
    assert logged_metric_names == {
        "shap_importance__feat1",
        "shap_importance__feat2",
    }

    mock_mlflow.log_artifact.assert_called_once()


def test_log_run_to_mlflow_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)

    config = Config(
        input_file="test.csv", target_column="target", task="binary", use_mlflow=True
    )
    results = pd.DataFrame({"feature_name": ["feat1"], "shap_importance": [0.5]})

    with pytest.raises(ImportError, match="The 'mlflow' package is required"):
        log_run_to_mlflow(config=config, results=results)
