"""Tests for model-agnostic sklearn permutation importance integration."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.pipeline import run_sklearn_permutation_importance


def test_use_sklearn_permutation_config_default():
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.use_sklearn_permutation_importance is False


def test_use_sklearn_permutation_cli_flag(monkeypatch):
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
            "--use-sklearn-permutation-importance",
        ],
    )
    args = parse_arguments()
    assert args.use_sklearn_permutation_importance is True


def test_run_sklearn_permutation_importance_mocked(monkeypatch):
    mock_res = MagicMock()
    mock_res.importances_mean = np.array([0.05, 0.12, 0.01])

    monkeypatch.setattr(
        "sklearn.inspection.permutation_importance",
        lambda *args, **kwargs: mock_res,
    )

    df = pd.DataFrame(
        {"feat1": [1, 2], "feat2": [3, 4], "feat3": [5, 6], "target": [0, 1]}
    )
    importances = run_sklearn_permutation_importance(
        model=MagicMock(),
        data=df,
        target_column="target",
        feature_names=["feat1", "feat2", "feat3"],
        task="binary",
    )
    assert pytest.approx(importances) == [0.05, 0.12, 0.01]
