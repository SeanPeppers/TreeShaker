"""Tests for LightGBM backend integration."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.pipeline import FeatureSelectionPipeline, run_lightgbm_importance


def test_use_lightgbm_config_default():
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.use_lightgbm is False


def test_use_lightgbm_cli_flag(monkeypatch):
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
            "--use-lightgbm",
        ],
    )
    args = parse_arguments()
    assert args.use_lightgbm is True


def test_run_lightgbm_importance_mocked(monkeypatch):
    mock_lgb_cls = MagicMock()
    mock_lgb = MagicMock()
    mock_lgb.feature_importances_ = np.array([10.0, 30.0, 60.0])
    mock_lgb_cls.return_value = mock_lgb

    mock_module = MagicMock()
    mock_module.LGBMClassifier = mock_lgb_cls
    import sys

    monkeypatch.setitem(sys.modules, "lightgbm", mock_module)

    df = pd.DataFrame(
        {"feat1": [1, 2], "feat2": [3, 4], "feat3": [5, 6], "target": [0, 1]}
    )
    importances = run_lightgbm_importance(
        data=df,
        target_column="target",
        feature_names=["feat1", "feat2", "feat3"],
        task="binary",
    )
    assert pytest.approx(importances) == [0.1, 0.3, 0.6]


def test_run_lightgbm_importance_import_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "lightgbm", None)

    df = pd.DataFrame({"feat1": [1, 2], "target": [0, 1]})
    with pytest.raises(ImportError, match="The 'lightgbm' package is required"):
        run_lightgbm_importance(
            data=df,
            target_column="target",
            feature_names=["feat1"],
            task="binary",
        )


def test_pipeline_with_lightgbm(monkeypatch, tmp_path):
    """The use_lightgbm Config flag actually reaches the pipeline's run(),
    not just run_lightgbm_importance() in isolation. lightgbm is mocked
    (same as test_run_lightgbm_importance_mocked above) since it's an
    optional extra not installed in the base test environment."""
    mock_lgb = MagicMock()
    mock_lgb.feature_importances_ = np.array([40.0, 60.0])
    mock_lgb_cls = MagicMock(return_value=mock_lgb)

    mock_module = MagicMock()
    mock_module.LGBMClassifier = mock_lgb_cls
    import sys

    monkeypatch.setitem(sys.modules, "lightgbm", mock_module)

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
        use_lightgbm=True,
        skip_tuning=True,
    )
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    assert "lightgbm_importance" in results.columns
    assert len(results["lightgbm_importance"]) == 2
