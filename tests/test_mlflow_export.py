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


def test_log_run_to_mlflow_real_tracking_store(tmp_path):
    """Exercises log_run_to_mlflow against a real (local, no server) MLflow
    tracking store, not just a mock -- confirms params/metrics/artifact
    actually get accepted and persisted, which pure call-assertion mocks
    can't catch (e.g. a value MLflow's real client would reject on
    serialization). Uses a sqlite tracking store: MLflow's plain filesystem
    store is deprecated/blocked by default in the installed version."""
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    experiment_name = "treeshaker-real-store-test"

    config = Config(
        input_file="test.csv",
        target_column="target",
        task="binary",
        use_mlflow=True,
        shap_importance_threshold=0.02,
    )
    results = pd.DataFrame(
        {
            "feature_name": ["feat1", "feat2"],
            "shap_importance": [0.5, 0.1],
            "recommendation": ["Keep", "Drop"],
        }
    )

    log_run_to_mlflow(config=config, results=results, experiment_name=experiment_name)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    # Params actually persisted and readable back (MLflow stores all param
    # values as strings).
    assert run.data.params["task"] == "binary"
    assert run.data.params["use_mlflow"] == "True"
    assert run.data.params["shap_importance_threshold"] == "0.02"
    assert "optuna_config" not in run.data.params

    # Metrics actually persisted with the expected per-feature naming and
    # values.
    assert run.data.metrics["shap_importance__feat1"] == 0.5
    assert run.data.metrics["shap_importance__feat2"] == 0.1

    # Artifact actually uploaded and readable back with the same content.
    artifacts = client.list_artifacts(run.info.run_id)
    assert any(a.path == "results.csv" for a in artifacts)
    local_path = client.download_artifacts(run.info.run_id, "results.csv")
    downloaded = pd.read_csv(local_path)
    assert list(downloaded["feature_name"]) == ["feat1", "feat2"]
    assert list(downloaded["recommendation"]) == ["Keep", "Drop"]
