"""Tests for shadow probe noise-floor detection."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_config as make_fast_config

from treeshaker.config import Config
from treeshaker.exceptions import ConfigError
from treeshaker.pipeline import FeatureSelectionPipeline
from treeshaker.shadow_probes import (
    add_shadow_probe_warnings,
    compute_probe_threshold,
    generate_shadow_probes,
)


def test_shadow_probes_logic_functions():
    # Test generate_shadow_probes
    df = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": [10, 20, 30, 40, 50],
        }
    )
    probes = generate_shadow_probes(df, n_probes=3, random_seed=42)
    assert isinstance(probes, pd.DataFrame)
    assert probes.shape == (5, 3)
    assert list(probes.columns) == [
        "__shadow_probe_0__",
        "__shadow_probe_1__",
        "__shadow_probe_2__",
    ]

    # Test compute_probe_threshold
    importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert compute_probe_threshold(importances, 100.0) == 0.5
    assert compute_probe_threshold(importances, 0.0) == 0.1
    assert compute_probe_threshold(importances, 50.0) == 0.3
    assert compute_probe_threshold(np.array([]), 100.0) == 0.0

    # Test add_shadow_probe_warnings
    results = pd.DataFrame(
        {
            "feature_name": ["feat1", "feat2"],
            "shap_importance": [0.05, 0.15],
        }
    )
    res_flagged = add_shadow_probe_warnings(results, 0.10)
    assert list(res_flagged["shadow_probe_flag"]) == [True, False]


def test_config_validation():
    # Valid configs should not raise
    Config(
        input_file="f.csv",
        target_column="y",
        task="binary",
        n_shadow_probes=5,
        shadow_probe_percentile=95.0,
    )

    # Invalid n_shadow_probes
    with pytest.raises(ConfigError, match="n_shadow_probes"):
        Config(input_file="f.csv", target_column="y", task="binary", n_shadow_probes=-1)

    # Invalid shadow_probe_percentile
    with pytest.raises(ConfigError, match="shadow_probe_percentile"):
        Config(
            input_file="f.csv",
            target_column="y",
            task="binary",
            shadow_probe_percentile=-0.1,
        )
    with pytest.raises(ConfigError, match="shadow_probe_percentile"):
        Config(
            input_file="f.csv",
            target_column="y",
            task="binary",
            shadow_probe_percentile=100.1,
        )


def test_pipeline_shadow_probes_stripped_and_flagged(binary_csv, tmp_path):
    # Run pipeline with shadow probes enabled
    config = make_fast_config(
        binary_csv,
        "binary",
        tmp_path,
        n_shadow_probes=5,
        shadow_probe_percentile=100.0,
        skip_tuning=True,
    )
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    # Verify that results DataFrame and feature_names contain no shadow probes
    assert not any(f.startswith("__shadow_probe_") for f in results["feature_name"])
    assert not any(f.startswith("__shadow_probe_") for f in pipeline.feature_names)

    # Verify that shadow_probe_flag column exists
    assert "shadow_probe_flag" in results.columns
    assert results["shadow_probe_flag"].dtype == bool


def test_pipeline_shadow_probes_disabled(binary_csv, tmp_path):
    # Run pipeline with shadow probes disabled (n_shadow_probes=0)
    config = make_fast_config(
        binary_csv,
        "binary",
        tmp_path,
        n_shadow_probes=0,
        skip_tuning=True,
    )
    results = FeatureSelectionPipeline(config).run()

    # Verify that shadow_probe_flag column is NOT present
    assert "shadow_probe_flag" not in results.columns


def test_pure_noise_dataset(tmp_path):
    # Create a dataset where the target and features are entirely random noise
    rng = np.random.default_rng(seed=42)
    n_samples = 150
    n_features = 8

    X = rng.normal(size=(n_samples, n_features))
    y = rng.choice([0, 1], size=n_samples)

    df = pd.DataFrame(X, columns=[f"noise_feat_{i}" for i in range(n_features)])
    df["target"] = y

    noise_csv = tmp_path / "noise.csv"
    df.to_csv(noise_csv, index=False)

    # Run pipeline on pure noise dataset with shadow probes enabled
    config = make_fast_config(
        str(noise_csv),
        "binary",
        tmp_path,
        n_shadow_probes=10,
        shadow_probe_percentile=50.0,  # 50th percentile threshold
        skip_tuning=True,
    )
    results = FeatureSelectionPipeline(config).run()

    # Since the dataset is pure noise, we expect most features to fall
    # below/around the shadow probes baseline and get flagged. We assert
    # at least 2 are flagged (a very safe lower bound).
    flagged_count = results["shadow_probe_flag"].sum()
    assert flagged_count >= 2
