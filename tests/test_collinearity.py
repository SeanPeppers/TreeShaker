"""Tests for collinearity-aware grouped feature selection."""

import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import make_config

from treeshaker.collinearity import (
    cluster_correlated_features,
    select_cluster_representative,
)
from treeshaker.config import Config
from treeshaker.exceptions import ConfigError
from treeshaker.pipeline import FeatureSelectionPipeline


def test_constant_column_no_runtime_warning():
    """A zero-variance column must not leak numpy's divide-by-zero warning.

    Correlating a constant column returns NaN (handled as 'not collinear'),
    but numpy emits a RuntimeWarning while dividing by its zero std. Regression
    guard: constant columns are common in messy real data.
    """
    X = pd.DataFrame(
        {
            "const": np.ones(50),
            "a": np.arange(50, dtype=float),
            "b": np.arange(50, dtype=float) * 2.0,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        clusters = cluster_correlated_features(
            X, ["const", "a", "b"], threshold=0.9, method="pearson"
        )
    # const is zero-variance -> its own singleton; a and b are collinear.
    assert ["const"] in clusters
    assert any(set(c) == {"a", "b"} for c in clusters)


def test_threshold_zero_no_column(binary_csv, tmp_path):
    """1. threshold=0 produces no collinearity_cluster column"""
    cfg = make_config(
        binary_csv,
        "binary",
        tmp_path,
        skip_tuning=True,
        collinearity_threshold=0.0,
    )
    pipeline = FeatureSelectionPipeline(cfg)
    results = pipeline.run()
    assert "collinearity_cluster" not in results.columns


def test_two_perfectly_correlated_features_one_cluster():
    """2. two perfectly correlated features form one cluster"""
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=100)
    x2 = x1.copy()
    x3 = rng.normal(size=100)
    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})

    clusters = cluster_correlated_features(
        df,
        ["X1", "X2", "X3"],
        threshold=0.99,
        method="spearman",
    )

    cluster_sets = [set(c) for c in clusters]
    assert {"X1", "X2"} in cluster_sets
    assert {"X3"} in cluster_sets
    assert len(clusters) == 2


def test_representative_is_highest_shap():
    """3. representative is highest SHAP feature"""
    cluster = ["feat_a", "feat_b", "feat_c"]
    shap_map = {"feat_a": 0.1, "feat_b": 0.5, "feat_c": 0.3}
    rep = select_cluster_representative(cluster, shap_map)
    assert rep == "feat_b"


def test_collinear_duplicate_recommendation(tmp_path):
    """4. CollinearDuplicate recommendation appears for non-rep"""
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=200)
    x2 = x1.copy()
    x3 = rng.normal(size=200)
    target = x1 * 2 + rng.normal(scale=0.1, size=200)

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "target": target})
    csv_path = tmp_path / "collinear_dup.csv"
    df.to_csv(csv_path, index=False)

    cfg = make_config(
        str(csv_path),
        "regression",
        tmp_path,
        skip_tuning=True,
        collinearity_threshold=0.95,
        collinearity_method="pearson",
    )
    pipeline = FeatureSelectionPipeline(cfg)
    results = pipeline.run()

    x1_row = results[results["feature_name"] == "X1"].iloc[0]
    x2_row = results[results["feature_name"] == "X2"].iloc[0]

    if x1_row["shap_importance"] > x2_row["shap_importance"]:
        assert x2_row["recommendation"] == "CollinearDuplicate"
        assert x1_row["recommendation"] != "CollinearDuplicate"
    else:
        assert x1_row["recommendation"] == "CollinearDuplicate"
        assert x2_row["recommendation"] != "CollinearDuplicate"


def test_collinearity_cluster_column_present(tmp_path):
    """5. collinearity_cluster column present in results when enabled"""
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=200)
    x2 = x1.copy()
    x3 = rng.normal(size=200)
    target = x1 * 2 + rng.normal(scale=0.1, size=200)

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "target": target})
    csv_path = tmp_path / "collinear_pres.csv"
    df.to_csv(csv_path, index=False)

    cfg = make_config(
        str(csv_path),
        "regression",
        tmp_path,
        skip_tuning=True,
        collinearity_threshold=0.95,
        collinearity_method="pearson",
    )
    pipeline = FeatureSelectionPipeline(cfg)
    results = pipeline.run()

    assert "collinearity_cluster" in results.columns
    cluster_map = dict(
        zip(results["feature_name"], results["collinearity_cluster"], strict=True)
    )
    assert cluster_map["X1"] == cluster_map["X2"]
    assert cluster_map["X1"] != "singleton"
    assert cluster_map["X3"] == "singleton"


def test_config_raises_on_invalid_method():
    """6. config raises on invalid method"""
    with pytest.raises(ConfigError, match="collinearity_method must be"):
        Config(
            input_file="x.csv",
            target_column="y",
            task="binary",
            collinearity_threshold=0.5,
            collinearity_method="invalid_method",
        )


def _make_gating_pipeline(
    collinearity_threshold: float = 0.9,
) -> FeatureSelectionPipeline:
    """A pipeline instance with train_data/feature_names set directly, so
    _apply_collinearity_gating can be tested as a pure function without
    running the full pipeline (deterministic -- no dependence on how XGBoost
    happens to split importance between correlated features)."""
    config = Config(
        input_file="x.csv",
        target_column="target",
        task="binary",
        collinearity_threshold=collinearity_threshold,
        collinearity_method="pearson",
    )
    pipeline = FeatureSelectionPipeline(config)
    rng = np.random.default_rng(42)
    base = rng.normal(size=100)
    pipeline.train_data = pd.DataFrame(
        {
            # ~collinear with kept
            "redundant": base + rng.normal(scale=0.001, size=100),
            "kept": base,
            "unrelated": rng.normal(size=100),
            "target": rng.integers(0, 2, size=100),
        }
    )
    pipeline.feature_names = ["redundant", "kept", "unrelated"]
    return pipeline


def test_collinearity_gating_downgrades_drop_correlated_with_keep():
    """A Drop feature collinear with a Keep feature is downgraded to Caution."""
    pipeline = _make_gating_pipeline()
    recommendations = ["Drop", "Keep", "Drop"]
    reasons = ["low importance", "", "low importance"]

    updated_recs, updated_reasons = pipeline._apply_collinearity_gating(
        recommendations, reasons
    )

    # "redundant" is collinear with "kept" (a Keep) -> demoted.
    assert updated_recs[0] == "Caution"
    assert "kept" in updated_reasons[0]
    assert "low importance" in updated_reasons[0]  # original reason preserved
    # "unrelated" has no collinear Keep partner -> stays Drop, untouched.
    assert updated_recs[2] == "Drop"
    assert updated_reasons[2] == "low importance"
    # Keep itself is never touched.
    assert updated_recs[1] == "Keep"


def test_collinearity_gating_disabled_is_noop():
    """collinearity_threshold=0 (default) leaves recommendations untouched."""
    pipeline = _make_gating_pipeline(collinearity_threshold=0.0)
    recommendations = ["Drop", "Keep", "Drop"]
    reasons = ["low importance", "", "low importance"]

    updated_recs, updated_reasons = pipeline._apply_collinearity_gating(
        recommendations, reasons
    )

    assert updated_recs == recommendations
    assert updated_reasons == reasons


def test_collinearity_gating_leaves_drop_without_keep_partner_alone():
    """A Drop feature whose cluster-mates are also Drop (no Keep) stays Drop."""
    pipeline = _make_gating_pipeline()
    # Both "redundant" and "kept" are Drop this time -- no Keep in the cluster.
    recommendations = ["Drop", "Drop", "Keep"]
    reasons = ["low importance", "low importance", ""]

    updated_recs, updated_reasons = pipeline._apply_collinearity_gating(
        recommendations, reasons
    )

    assert updated_recs == recommendations
    assert updated_reasons == reasons
