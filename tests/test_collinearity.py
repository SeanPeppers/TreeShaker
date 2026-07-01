"""Tests for collinearity-aware grouped feature selection."""

import pandas as pd
import numpy as np
import pytest
from conftest import make_config

from treeshaker.config import Config
from treeshaker.exceptions import ConfigError
from treeshaker.pipeline import FeatureSelectionPipeline
from treeshaker.collinearity import (
    cluster_correlated_features,
    select_cluster_representative,
)


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
    cluster_map = dict(zip(results["feature_name"], results["collinearity_cluster"]))
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
