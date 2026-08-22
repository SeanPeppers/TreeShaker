"""Tests for RFECV integration."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.pipeline import FeatureSelectionPipeline, run_rfecv


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


def test_pipeline_with_rfecv(tmp_path):
    """The use_rfecv Config flag actually reaches the pipeline's run(), not
    just run_rfecv() in isolation -- regression guard for the call site in
    run(), not just the underlying function."""
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
        use_rfecv=True,
        skip_tuning=True,
    )
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    assert "rfecv_ranking" in results.columns
    assert len(results["rfecv_ranking"]) == 2
