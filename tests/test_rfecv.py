"""Tests for RFECV integration."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.pipeline import run_rfecv


def test_use_rfecv_config_default():
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.use_rfecv is False


def test_use_rfecv_cli_flag(monkeypatch):
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
            "--use-rfecv",
        ],
    )
    args = parse_arguments()
    assert args.use_rfecv is True


def test_run_rfecv_mocked(monkeypatch):
    mock_rfecv_cls = MagicMock()
    mock_rfecv = MagicMock()
    mock_rfecv.ranking_ = np.array([1, 2, 1])
    mock_rfecv_cls.return_value = mock_rfecv

    monkeypatch.setattr("sklearn.feature_selection.RFECV", mock_rfecv_cls)

    df = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": [2, 3, 4, 5, 6],
            "feat3": [3, 4, 5, 6, 7],
            "target": [0, 1, 0, 1, 0],
        }
    )
    ranking = run_rfecv(
        model=MagicMock(),
        data=df,
        target_column="target",
        feature_names=["feat1", "feat2", "feat3"],
        task="binary",
    )
    assert ranking == [1, 2, 1]
