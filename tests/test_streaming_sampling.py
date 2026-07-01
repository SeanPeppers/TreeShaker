"""Tests for streaming sample-based evaluation (Stage 3).

Covers:
- streaming_eval_sample_size Config defaults and validation
- custom_model + streaming ConfigError
- CLI --streaming-eval-sample-size flag
- MCP schema exposure
- _eval_sample() passthrough / sampling behaviour
- End-to-end pipeline with a small streaming_eval_sample_size
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.exceptions import ConfigError
from treeshaker.mcp_server import _build_config, list_config_schema
from treeshaker.pipeline import FeatureSelectionPipeline

# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_streaming_eval_sample_size_default():
    cfg = Config(input_file="x.csv", target_column="y", task="binary")
    assert cfg.streaming_eval_sample_size == 50_000


def test_streaming_eval_sample_size_validation():
    with pytest.raises(ConfigError, match="streaming_eval_sample_size must be >= 1"):
        Config(
            input_file="x.csv",
            target_column="y",
            task="binary",
            streaming_eval_sample_size=0,
        )


# ---------------------------------------------------------------------------
# custom_model + streaming incompatibility
# ---------------------------------------------------------------------------


def test_custom_model_streaming_raises():
    """Combining streaming=True with custom_model must raise ConfigError."""
    dummy = MagicMock()
    dummy.fit = MagicMock()
    dummy.predict = MagicMock()
    dummy.predict_proba = MagicMock()
    with pytest.raises(
        ConfigError, match="streaming=True is incompatible with custom_model"
    ):
        Config(
            input_file="x.csv",
            target_column="y",
            task="binary",
            streaming=True,
            custom_model=dummy,
        )


def test_custom_model_without_streaming_ok():
    """custom_model without streaming must not raise."""
    dummy = MagicMock()
    dummy.fit = MagicMock()
    dummy.predict = MagicMock()
    dummy.predict_proba = MagicMock()
    cfg = Config(
        input_file="x.csv",
        target_column="y",
        task="binary",
        custom_model=dummy,
    )
    assert cfg.custom_model is dummy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_streaming_eval_sample_size_flag():
    with patch(
        "sys.argv",
        [
            "treeshaker",
            "--input", "data.csv",
            "--target", "y",
            "--task", "binary",
            "--streaming",
            "--streaming-eval-sample-size", "1000",
        ],
    ):
        args = parse_arguments()
        assert args.streaming_eval_sample_size == 1000


def test_cli_streaming_eval_sample_size_default():
    with patch(
        "sys.argv",
        ["treeshaker", "--input", "d.csv", "--target", "y", "--task", "binary"],
    ):
        args = parse_arguments()
        assert args.streaming_eval_sample_size == 50_000


# ---------------------------------------------------------------------------
# MCP schema
# ---------------------------------------------------------------------------


def test_mcp_schema_exposes_eval_sample_size():
    schema = list_config_schema()
    names = [f["name"] for f in schema["fields"]]
    assert "streaming_eval_sample_size" in names


def test_mcp_build_config_eval_sample_size():
    cfg = _build_config(
        input_path="data.csv",
        target_column="y",
        task="binary",
        output_path="out.csv",
        streaming=True,
        streaming_eval_sample_size=2000,
    )
    assert cfg.streaming_eval_sample_size == 2000


# ---------------------------------------------------------------------------
# _eval_sample() behaviour
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_path, **extra_config_kwargs):
    """Create a pipeline whose Config points at a dummy CSV."""
    csv_file = tmp_path / "dummy.csv"
    df = pd.DataFrame({"a": range(10), "target": [0, 1] * 5})
    df.to_csv(csv_file, index=False)
    cfg = Config(
        input_file=str(csv_file),
        target_column="target",
        task="binary",
        **extra_config_kwargs,
    )
    return FeatureSelectionPipeline(cfg)


def test_eval_sample_passthrough_when_not_streaming(tmp_path):
    """_eval_sample must return original df unchanged when streaming=False."""
    pipeline = _make_pipeline(tmp_path, streaming=False)
    df = pd.DataFrame({"x": range(200)})
    result = pipeline._eval_sample(df)
    # Exact same object (identity) — no copy/sample taken.
    assert result is df


def test_eval_sample_passthrough_when_small(tmp_path):
    """_eval_sample must return original df when len(df) <= cap."""
    pipeline = _make_pipeline(
        tmp_path,
        streaming=True,
        streaming_eval_sample_size=500,
    )
    df = pd.DataFrame({"x": range(100)})
    result = pipeline._eval_sample(df)
    assert len(result) == 100


def test_eval_sample_truncates_when_large(tmp_path):
    """_eval_sample must return a random sample of exactly cap rows."""
    cap = 50
    pipeline = _make_pipeline(
        tmp_path,
        streaming=True,
        streaming_eval_sample_size=cap,
    )
    df = pd.DataFrame({"x": range(500)})
    result = pipeline._eval_sample(df)
    assert len(result) == cap
    # All returned rows come from the original df.
    assert set(result["x"]).issubset(set(df["x"]))


def test_eval_sample_reproducible(tmp_path):
    """_eval_sample with the same random_seed produces same sample."""
    seed = 7
    cap = 30
    pipeline = _make_pipeline(
        tmp_path,
        streaming=True,
        streaming_eval_sample_size=cap,
        random_seed=seed,
    )
    df = pd.DataFrame({"x": np.arange(300)})
    r1 = pipeline._eval_sample(df)
    r2 = pipeline._eval_sample(df)
    pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# End-to-end: small streaming_eval_sample_size
# ---------------------------------------------------------------------------


def test_pipeline_runs_with_tiny_eval_sample(tmp_path):
    """Full pipeline must complete without error with a tiny eval sample."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "target": rng.integers(0, 2, n),
        }
    )
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)

    cfg = Config(
        input_file=str(csv_file),
        target_column="target",
        task="binary",
        streaming=True,
        streaming_chunk_size=100,
        streaming_eval_sample_size=50,  # much smaller than dataset
        skip_tuning=True,
        n_permutations=1,
        test_size=0.2,
        val_size=0.2,
    )
    pipeline = FeatureSelectionPipeline(cfg)
    results = pipeline.run()
    assert results is not None
    assert len(results) == 2  # f1 and f2
    assert "recommendation" in results.columns
