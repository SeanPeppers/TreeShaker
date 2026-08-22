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


def _make_gating_pipeline(n_shadow_probes: int = 5) -> FeatureSelectionPipeline:
    """A pipeline instance with just enough state set for
    _apply_shadow_probe_gating to be tested as a pure function against a
    hand-crafted results DataFrame, without running the full pipeline
    (deterministic -- no dependence on real training producing a feature
    that happens to land on either side of the noise floor)."""
    config = Config(
        input_file="x.csv",
        target_column="target",
        task="binary",
        n_shadow_probes=n_shadow_probes,
        shadow_probe_percentile=90.0,
    )
    return FeatureSelectionPipeline(config)


def test_shadow_probe_gating_downgrades_disputed_drop():
    """A Drop feature whose importance clears the noise floor
    (shadow_probe_flag=False) is downgraded to Caution."""
    pipeline = _make_gating_pipeline()
    results = pd.DataFrame(
        {
            "feature_name": ["genuinely_noisy", "disputed", "kept"],
            "shap_importance": [0.001, 0.03, 0.5],
            "recommendation": ["Drop", "Drop", "Keep"],
            "reason": ["low importance", "low importance", ""],
            "shadow_probe_flag": [True, False, False],
        }
    )

    gated = pipeline._apply_shadow_probe_gating(results)

    # genuinely_noisy: Drop AND doesn't clear the floor -> stays Drop.
    assert gated.loc[0, "recommendation"] == "Drop"
    assert gated.loc[0, "reason"] == "low importance"
    # disputed: Drop but DOES clear the floor -> downgraded to Caution,
    # original reason preserved, noise-floor note appended.
    assert gated.loc[1, "recommendation"] == "Caution"
    assert "low importance" in gated.loc[1, "reason"]
    assert "shadow-probe noise floor" in gated.loc[1, "reason"]
    # Keep is never touched regardless of its shadow_probe_flag value.
    assert gated.loc[2, "recommendation"] == "Keep"
    assert gated.loc[2, "reason"] == ""


def test_pipeline_shadow_probe_gating_disabled_by_default(binary_csv, tmp_path):
    """n_shadow_probes=0 (default) never reaches _apply_shadow_probe_gating
    at all -- no shadow_probe_flag column, recommendations driven purely by
    the fixed thresholds, same as before this gating existed."""
    config = make_fast_config(binary_csv, "binary", tmp_path, skip_tuning=True)
    assert config.n_shadow_probes == 0
    results = FeatureSelectionPipeline(config).run()
    assert "shadow_probe_flag" not in results.columns
    assert not any("shadow-probe noise floor" in str(r) for r in results["reason"] if r)


def test_severe_imbalance_noise_floor_gating_promotes_to_drop():
    """A Keep/Caution feature at or below the noise floor is demoted to
    Drop when severe class imbalance is in play -- unlike
    _apply_shadow_probe_gating, this one creates new Drops, since under
    severe imbalance nothing gets flagged Drop by the fixed thresholds in
    the first place (verified empirically against the benchmark script's
    imbalanced scenario -- see the pipeline-level test below)."""
    pipeline = _make_gating_pipeline()
    results = pd.DataFrame(
        {
            "feature_name": ["noisy_but_kept", "noisy_but_caution", "real_signal"],
            "shap_importance": [0.03, 0.05, 0.9],
            "recommendation": ["Keep", "Caution", "Keep"],
            "reason": ["", "some other note", ""],
            "shadow_probe_flag": [True, True, False],
        }
    )

    gated = pipeline._apply_severe_imbalance_noise_floor_gating(results)

    # At the noise floor and Keep -> demoted to Drop.
    assert gated.loc[0, "recommendation"] == "Drop"
    assert "Severe class imbalance" in gated.loc[0, "reason"]
    assert "'Keep'" in gated.loc[0, "reason"]
    # At the noise floor and Caution -> demoted to Drop too, existing
    # reason preserved.
    assert gated.loc[1, "recommendation"] == "Drop"
    assert "some other note" in gated.loc[1, "reason"]
    assert "'Caution'" in gated.loc[1, "reason"]
    # Clears the noise floor -> untouched regardless of this method running.
    assert gated.loc[2, "recommendation"] == "Keep"
    assert gated.loc[2, "reason"] == ""


def test_severe_imbalance_noise_floor_gating_never_touches_existing_drop():
    """A feature already Drop is left alone by this method (it only
    demotes Keep/Caution -- _apply_shadow_probe_gating owns the
    Drop-to-Caution direction, and the two never overlap in what they
    touch)."""
    pipeline = _make_gating_pipeline()
    results = pd.DataFrame(
        {
            "feature_name": ["already_dropped"],
            "shap_importance": [0.001],
            "recommendation": ["Drop"],
            "reason": ["low importance"],
            "shadow_probe_flag": [True],
        }
    )

    gated = pipeline._apply_severe_imbalance_noise_floor_gating(results)

    assert gated.loc[0, "recommendation"] == "Drop"
    assert gated.loc[0, "reason"] == "low importance"


def test_pipeline_severe_imbalance_auto_enables_noise_floor_gating(tmp_path):
    """The exact severe-imbalance scenario the benchmark script found
    (0.000 drop-recall) is now fixed at default Config -- no
    n_shadow_probes set by the caller, auto-enabled because
    check_class_imbalance fires. Regression guard for the actual bug."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=3000,
        n_features=30,
        n_informative=6,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=[0.93, 0.07],
        class_sep=1.5,
        shuffle=False,
        random_state=0,
    )
    cols = [f"f{i}" for i in range(30)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    csv_path = tmp_path / "imbalanced.csv"
    df.to_csv(csv_path, index=False)

    config = make_fast_config(str(csv_path), "binary", tmp_path, skip_tuning=True)
    assert config.n_shadow_probes == 0  # not set by the caller

    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    assert pipeline.severe_class_imbalance is True
    assert pipeline._effective_n_shadow_probes == 30
    assert "shadow_probe_flag" in results.columns

    results = results.set_index("feature_name").loc[cols]
    n_signal = 10  # 6 informative + 4 redundant
    is_noise = np.array([i >= n_signal for i in range(30)])
    is_drop = (results["recommendation"] == "Drop").to_numpy()
    tp = int((is_drop & is_noise).sum())
    fn = int((~is_drop & is_noise).sum())
    drop_recall = tp / (tp + fn) if (tp + fn) else 1.0

    # Was exactly 0.000 before this fix; empirically ~1.0 at these
    # settings across seeds -- assert a high but not knife's-edge bound so
    # this isn't flaky on the RNG's own variance.
    assert drop_recall >= 0.8, (
        f"drop_recall={drop_recall:.3f} -- severe-imbalance noise floor"
        f" gating regressed"
    )
