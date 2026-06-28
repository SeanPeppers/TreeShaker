"""Tests for CatBoost backend integration."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.pipeline import run_catboost_importance


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
