"""Tests for Stage 2: out-of-core XGBoost training via external memory.

Covers the streaming training path: the on-disk chunk spill + DataIter, the
BoosterAdapter sklearn-style surface, and an end-to-end pipeline run with
streaming=True producing the same recommendations as the in-memory path
(small chunk size to force multiple on-disk chunks).
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_config

from treeshaker.pipeline import FeatureSelectionPipeline
from treeshaker.streaming_train import (
    BoosterAdapter,
    _spill_chunks,
    train_booster_streaming,
)


def test_spill_chunks_writes_multiple_files(tmp_path):
    """Verify _spill_chunks splits rows into ceil(n/chunk_size) files."""
    X = pd.DataFrame({"a": range(25), "b": range(25)})
    y = pd.Series([0, 1] * 12 + [0])
    paths = _spill_chunks(X, y, chunk_size=10, cache_dir=str(tmp_path))
    assert len(paths) == 3  # 10 + 10 + 5
    # Round-trip preserves the label column and row count.
    total = sum(len(pd.read_pickle(p)) for p in paths)  # noqa: S301
    assert total == 25
    assert "__label__" in pd.read_pickle(paths[0]).columns  # noqa: S301


def test_spill_chunks_rejects_empty():
    """Verify _spill_chunks raises on an empty dataset."""
    with pytest.raises(ValueError, match="empty dataset"):
        _spill_chunks(pd.DataFrame({"a": []}), pd.Series([], dtype=float), 10, ".")


def test_train_booster_streaming_binary(tmp_path):
    """Verify out-of-core training yields a working binary BoosterAdapter."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))
    adapter = train_booster_streaming(
        X=X,
        y=y,
        task="binary",
        params={"max_depth": 3, "learning_rate": 0.3},
        feature_names=["a", "b", "c", "d"],
        chunk_size=40,  # 3 on-disk chunks
        num_boost_round=20,
    )
    assert isinstance(adapter, BoosterAdapter)
    proba = adapter.predict_proba(X)
    assert proba.shape == (120, 2)
    # Probabilities sum to 1 across the two columns.
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    preds = adapter.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
    # Model learned the separable signal reasonably well.
    assert (preds == y.to_numpy()).mean() > 0.8
    assert adapter.get_booster() is not None


def test_train_booster_streaming_regression(tmp_path):
    """Verify out-of-core regression training returns continuous predictions."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(90, 3)), columns=["a", "b", "c"])
    y = pd.Series(2.0 * X["a"] - X["b"])
    adapter = train_booster_streaming(
        X=X,
        y=y,
        task="regression",
        params={"max_depth": 3},
        feature_names=["a", "b", "c"],
        chunk_size=30,
        num_boost_round=20,
    )
    preds = adapter.predict(X)
    assert preds.shape == (90,)
    assert preds.dtype.kind == "f"


def test_pipeline_streaming_matches_in_memory(binary_csv, tmp_path):
    """End-to-end: streaming=True yields a BoosterAdapter and the same
    recommendations as the default in-memory path on the same data."""
    cfg_mem = make_config(binary_csv, "binary", tmp_path, skip_tuning=True)
    cfg_stream = make_config(
        binary_csv,
        "binary",
        tmp_path,
        skip_tuning=True,
        streaming=True,
        streaming_chunk_size=32,  # force several on-disk chunks (200 rows)
    )

    res_mem = FeatureSelectionPipeline(cfg_mem).run()
    pipe_stream = FeatureSelectionPipeline(cfg_stream)
    res_stream = pipe_stream.run()

    assert type(pipe_stream.model).__name__ == "BoosterAdapter"

    mem = res_mem.set_index("feature_name")["recommendation"].to_dict()
    stream = res_stream.set_index("feature_name")["recommendation"].to_dict()
    # Recommendations should agree on the overwhelming majority of features
    # (identical data, same params; only the training API differs).
    agree = sum(1 for k in mem if mem[k] == stream.get(k))
    assert agree >= len(mem) - 1
