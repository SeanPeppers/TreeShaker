"""Tests for Random Forest baseline importance integration."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.pipeline import FeatureSelectionPipeline, run_rf_baseline_importance


def test_use_rf_baseline_config_default():
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.use_rf_baseline is False


def test_use_rf_baseline_cli_flag(monkeypatch):
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
            "--use-rf-baseline",
        ],
    )
    args = parse_arguments()
    assert args.use_rf_baseline is True


def test_run_rf_baseline_importance_mocked(monkeypatch):
    mock_rf_cls = MagicMock()
    mock_rf = MagicMock()
    mock_rf.feature_importances_ = np.array([0.1, 0.4, 0.5])
    mock_rf_cls.return_value = mock_rf

    monkeypatch.setattr("sklearn.ensemble.RandomForestClassifier", mock_rf_cls)

    df = pd.DataFrame(
        {"feat1": [1, 2], "feat2": [3, 4], "feat3": [5, 6], "target": [0, 1]}
    )
    importances = run_rf_baseline_importance(
        data=df,
        target_column="target",
        feature_names=["feat1", "feat2", "feat3"],
        task="binary",
    )
    assert pytest.approx(importances) == [0.1, 0.4, 0.5]


def test_pipeline_with_rf_baseline(tmp_path):
    """The use_rf_baseline Config flag actually reaches the pipeline's
    run(), not just run_rf_baseline_importance() in isolation."""
    csv_path = tmp_path / "data.csv"
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
            "target": np.random.choice([0, 1], size=50),
        }
    )
    df.to_csv(csv_path, index=False)

    config = Config(
        input_file=str(csv_path),
        target_column="target",
        task="binary",
        use_rf_baseline=True,
        skip_tuning=True,
    )
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    assert "rf_importance" in results.columns
    assert len(results["rf_importance"]) == 2
