"""Tests for CatBoost backend integration."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.pipeline import FeatureSelectionPipeline, run_catboost_importance


def test_use_catboost_config_default():
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.use_catboost is False


def test_use_catboost_cli_flag(monkeypatch):
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
            "--use-catboost",
        ],
    )
    args = parse_arguments()
    assert args.use_catboost is True


def test_run_catboost_importance_mocked(monkeypatch):
    mock_cb_cls = MagicMock()
    mock_cb = MagicMock()
    mock_cb.get_feature_importance.return_value = np.array([20.0, 30.0, 50.0])
    mock_cb_cls.return_value = mock_cb

    mock_module = MagicMock()
    mock_module.CatBoostClassifier = mock_cb_cls
    import sys

    monkeypatch.setitem(sys.modules, "catboost", mock_module)

    df = pd.DataFrame(
        {"feat1": [1, 2], "feat2": ["a", "b"], "feat3": [5, 6], "target": [0, 1]}
    )
    importances = run_catboost_importance(
        data=df,
        target_column="target",
        feature_names=["feat1", "feat2", "feat3"],
        task="binary",
    )
    assert pytest.approx(importances) == [0.2, 0.3, 0.5]


def test_run_catboost_importance_import_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "catboost", None)

    df = pd.DataFrame({"feat1": [1, 2], "target": [0, 1]})
    with pytest.raises(ImportError, match="The 'catboost' package is required"):
        run_catboost_importance(
            data=df,
            target_column="target",
            feature_names=["feat1"],
            task="binary",
        )


def test_pipeline_with_catboost(monkeypatch, tmp_path):
    """The use_catboost Config flag actually reaches the pipeline's run(),
    not just run_catboost_importance() in isolation. Also a regression
    guard for the bug where this call site used self.test_data instead of
    train+val (it fits a fresh model internally). catboost is mocked (same
    as test_run_catboost_importance_mocked above) since it's an optional
    extra not installed in the base test environment."""
    mock_cb = MagicMock()
    mock_cb.get_feature_importance.return_value = np.array([40.0, 60.0])
    mock_cb_cls = MagicMock(return_value=mock_cb)

    mock_module = MagicMock()
    mock_module.CatBoostClassifier = mock_cb_cls
    import sys

    monkeypatch.setitem(sys.modules, "catboost", mock_module)

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
        use_catboost=True,
        skip_tuning=True,
    )
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    assert "catboost_importance" in results.columns
    assert len(results["catboost_importance"]) == 2
    # Model must have been fit on train+val, never on the held-out test
    # split (the bug this test guards against).
    fit_call = mock_cb.fit.call_args
    fit_rows = len(fit_call.args[0]) if fit_call.args else len(fit_call.kwargs["X"])
    assert fit_rows == len(pipeline.train_data) + len(pipeline.val_data)
