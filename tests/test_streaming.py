"""Tests for the streaming / out-of-core data path (Stage 1: chunked read).

Stage 1 wires the opt-in `streaming` / `streaming_chunk_size` config surface
across Config/CLI/MCP and adds a chunked-CSV read that downcasts numeric
dtypes per chunk. These tests cover the config plumbing and verify the
chunked read produces the same rows as a normal read while narrowing dtypes.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.exceptions import ConfigError
from treeshaker.mcp_server import _build_config, list_config_schema
from treeshaker.utils import _downcast_numeric, _read_csv_streaming, load_csv


def test_streaming_config_defaults():
    """Verify streaming fields default to off / 100k in Config."""
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.streaming is False
    assert config.streaming_chunk_size == 100_000


def test_streaming_chunk_size_validation():
    """Verify Config rejects a non-positive streaming_chunk_size."""
    with pytest.raises(ConfigError, match="streaming_chunk_size must be >= 1"):
        Config(
            input_file="test.csv",
            target_column="target",
            task="binary",
            streaming_chunk_size=0,
        )


def test_cli_streaming_flags():
    """Verify CLI exposes --streaming and --streaming-chunk-size."""
    with patch(
        "sys.argv",
        [
            "treeshaker",
            "--input",
            "data.csv",
            "--target",
            "y",
            "--task",
            "binary",
            "--streaming",
            "--streaming-chunk-size",
            "5000",
        ],
    ):
        args = parse_arguments()
        assert args.streaming is True
        assert args.streaming_chunk_size == 5000


def test_cli_streaming_defaults_off():
    """Verify --streaming defaults off and chunk size defaults to 100000."""
    with patch(
        "sys.argv",
        ["treeshaker", "--input", "d.csv", "--target", "y", "--task", "binary"],
    ):
        args = parse_arguments()
        assert args.streaming is False
        assert args.streaming_chunk_size == 100_000


def test_mcp_server_streaming():
    """Verify MCP schema and _build_config expose streaming parameters."""
    schema = list_config_schema()
    field_names = [f["name"] for f in schema["fields"]]
    assert "streaming" in field_names
    assert "streaming_chunk_size" in field_names

    cfg = _build_config(
        input_path="data.csv",
        target_column="y",
        task="binary",
        output_path="out.csv",
        streaming=True,
        streaming_chunk_size=2000,
    )
    assert cfg.streaming is True
    assert cfg.streaming_chunk_size == 2000


def test_downcast_numeric_narrows_dtypes():
    """Verify _downcast_numeric narrows int/float and preserves values."""
    df = pd.DataFrame(
        {
            "small_int": np.array([1, 2, 3], dtype=np.int64),
            "flt": np.array([1.5, 2.5, 3.5], dtype=np.float64),
            "text": ["a", "b", "c"],
        }
    )
    out = _downcast_numeric(df)
    # Integers narrowed below int64, floats below float64, text untouched.
    assert out["small_int"].dtype.itemsize < 8
    assert out["flt"].dtype == np.float32
    assert out["text"].dtype == object
    # Values preserved.
    assert out["small_int"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(out["flt"].to_numpy(), [1.5, 2.5, 3.5])


def test_read_csv_streaming_matches_plain_read(tmp_path):
    """Verify chunked read returns same rows/columns as a single read."""
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "a": list(range(50)),
            "b": [float(i) / 2 for i in range(50)],
            "target": [0, 1] * 25,
        }
    )
    df.to_csv(csv_file, index=False)

    streamed = _read_csv_streaming(str(csv_file), chunk_size=7)
    plain = pd.read_csv(csv_file)

    assert list(streamed.columns) == list(plain.columns)
    assert len(streamed) == len(plain)
    # Same values regardless of dtype width.
    pd.testing.assert_frame_equal(
        streamed.astype(plain.dtypes.to_dict()), plain, check_like=True
    )


def test_load_csv_streaming_equivalent_split(tmp_path):
    """Verify load_csv(streaming=True) yields the same split sizes as off."""
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "feature": list(range(100)),
            "target": [0] * 50 + [1] * 50,
        }
    )
    df.to_csv(csv_file, index=False)

    kwargs = dict(
        filepath=str(csv_file),
        target_column="target",
        task="binary",
        test_size=0.2,
        val_size=0.2,
        random_seed=42,
    )
    tr_s, va_s, te_s = load_csv(streaming=True, streaming_chunk_size=13, **kwargs)
    tr, va, te = load_csv(**kwargs)

    assert (len(tr_s), len(va_s), len(te_s)) == (len(tr), len(va), len(te))
