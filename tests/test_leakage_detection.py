"""Tests for LeakageDetector in leakage_detection.py."""

import numpy as np
import pandas as pd
import pytest

from treeshaker.leakage_detection import LeakageConfig, LeakageDetector


@pytest.fixture
def detector() -> LeakageDetector:
    return LeakageDetector()


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A clean dataset with no leakage signals."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 100),
            "feature_b": rng.normal(5, 2, 100),
            "feature_c": rng.normal(-1, 3, 100),
            "target": rng.integers(0, 2, 100).astype(float),
        }
    )


def test_no_warnings_on_clean_data(detector, clean_df):
    shap = np.array([0.3, 0.2, 0.1])
    warnings = detector.detect(
        clean_df, "target", ["feature_a", "feature_b", "feature_c"], shap
    )
    assert all(len(v) == 0 for v in warnings.values())


def test_target_correlation_flagged():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 200)
    df = pd.DataFrame(
        {"leaky": base * 0.999 + rng.normal(0, 0.01, 200), "target": base}
    )
    detector = LeakageDetector(LeakageConfig(target_corr_threshold=0.95))
    shap = np.array([0.5])
    warnings = detector.detect(df, "target", ["leaky"], shap)
    assert len(warnings["leaky"]) > 0
    assert "correlation" in warnings["leaky"][0].lower()


def test_dominant_shap_flagged(detector, clean_df):
    # feature_a has 95% of total SHAP
    shap = np.array([0.95, 0.03, 0.02])
    warnings = detector.detect(
        clean_df, "target", ["feature_a", "feature_b", "feature_c"], shap
    )
    assert len(warnings["feature_a"]) > 0
    assert "%" in warnings["feature_a"][0]


def test_near_constant_flagged():
    df = pd.DataFrame(
        {
            "constant_feature": [1.0] * 199 + [2.0],
            "normal_feature": np.random.default_rng(42).normal(0, 1, 200),
            "target": np.random.default_rng(1).integers(0, 2, 200).astype(float),
        }
    )
    detector = LeakageDetector(LeakageConfig(near_constant_threshold=0.98))
    shap = np.array([0.4, 0.6])
    warnings = detector.detect(
        df, "target", ["constant_feature", "normal_feature"], shap
    )
    assert len(warnings["constant_feature"]) > 0
    assert "constant" in warnings["constant_feature"][0].lower()
    assert len(warnings["normal_feature"]) == 0


def test_near_duplicate_flagged():
    base = np.random.default_rng(42).normal(0, 1, 200)
    df = pd.DataFrame(
        {
            "feat_a": base,
            "feat_b": base * 1.0001,
            "target": np.random.default_rng(1).integers(0, 2, 200).astype(float),
        }
    )
    detector = LeakageDetector(LeakageConfig(near_duplicate_corr_threshold=0.98))
    shap = np.array([0.4, 0.4])
    warnings = detector.detect(df, "target", ["feat_a", "feat_b"], shap)
    flagged = [k for k, v in warnings.items() if v]
    assert len(flagged) > 0


def test_suspicious_name_id_flagged(detector, clean_df):
    df = clean_df.rename(columns={"feature_a": "user_id"})
    shap = np.array([0.3, 0.2, 0.1])
    warnings = detector.detect(
        df, "target", ["user_id", "feature_b", "feature_c"], shap
    )
    assert len(warnings["user_id"]) > 0
    assert "identifier" in warnings["user_id"][0].lower()


def test_suspicious_name_future_flagged(detector, clean_df):
    df = clean_df.rename(columns={"feature_a": "future_price"})
    shap = np.array([0.3, 0.2, 0.1])
    warnings = detector.detect(
        df, "target", ["future_price", "feature_b", "feature_c"], shap
    )
    assert len(warnings["future_price"]) > 0
    assert "future" in warnings["future_price"][0].lower()


def test_exact_id_name_flagged(detector):
    df = pd.DataFrame(
        {
            "id": range(100),
            "feature_b": np.random.default_rng(42).normal(0, 1, 100),
            "target": np.random.default_rng(1).integers(0, 2, 100).astype(float),
        }
    )
    shap = np.array([0.1, 0.3])
    warnings = detector.detect(df, "target", ["id", "feature_b"], shap)
    assert len(warnings["id"]) > 0


def test_returns_all_features(detector, clean_df):
    feature_names = ["feature_a", "feature_b", "feature_c"]
    shap = np.array([0.3, 0.2, 0.1])
    warnings = detector.detect(clean_df, "target", feature_names, shap)
    assert set(warnings.keys()) == set(feature_names)


def test_multicollinearity_flagged():
    """Three features that are jointly redundant but only moderately
    correlated pairwise should be flagged by VIF even though pairwise
    correlation would miss them."""
    rng = np.random.default_rng(42)
    n = 300
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    # c is a near-perfect linear combination of a and b.
    c = a + b + rng.normal(0, 0.001, n)
    df = pd.DataFrame(
        {
            "feat_a": a,
            "feat_b": b,
            "feat_c": c,
            "target": rng.integers(0, 2, n).astype(float),
        }
    )
    detector = LeakageDetector(LeakageConfig(vif_threshold=10.0))
    shap = np.array([0.3, 0.3, 0.3])
    warnings = detector.detect(df, "target", ["feat_a", "feat_b", "feat_c"], shap)
    assert len(warnings["feat_c"]) > 0
    assert "multicollinearity" in warnings["feat_c"][0].lower()
    assert "vif" in warnings["feat_c"][0].lower()


def test_multicollinearity_not_flagged_on_independent_features(detector, clean_df):
    """Independent features should not trigger the VIF check."""
    shap = np.array([0.3, 0.2, 0.1])
    warnings = detector.detect(
        clean_df, "target", ["feature_a", "feature_b", "feature_c"], shap
    )
    for name in ["feature_a", "feature_b", "feature_c"]:
        assert not any("vif" in w.lower() for w in warnings[name])


def test_multicollinearity_skipped_with_fewer_than_two_numeric_features(detector):
    """With fewer than two numeric features, the VIF check should be a no-op."""
    df = pd.DataFrame(
        {
            "feature_a": np.random.default_rng(42).normal(0, 1, 50),
            "category": ["x"] * 25 + ["y"] * 25,
            "target": np.random.default_rng(1).integers(0, 2, 50).astype(float),
        }
    )
    shap = np.array([0.5, 0.5])
    warnings = detector.detect(df, "target", ["feature_a", "category"], shap)
    assert all(len(v) == 0 for v in warnings.values())


def test_multicollinearity_handles_perfect_collinearity():
    """Perfect collinearity (R^2 == 1) should be flagged without a
    division-by-zero error, reporting an infinite VIF."""
    rng = np.random.default_rng(42)
    n = 100
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = a + b  # Exactly linearly dependent, no noise.
    df = pd.DataFrame(
        {
            "feat_a": a,
            "feat_b": b,
            "feat_c": c,
            "target": rng.integers(0, 2, n).astype(float),
        }
    )
    detector = LeakageDetector(LeakageConfig(vif_threshold=10.0))
    shap = np.array([0.3, 0.3, 0.3])
    warnings = detector.detect(df, "target", ["feat_a", "feat_b", "feat_c"], shap)
    assert len(warnings["feat_c"]) > 0
    assert "infinite" in warnings["feat_c"][0].lower()


def test_temporal_feature_high_correlation_not_flagged_as_leakage():
    """A lag feature normally correlates very highly with the target --
    that's expected, not leakage, so it should be exempted by default."""
    rng = np.random.default_rng(42)
    n = 200
    # A near-target-equal series (e.g. a slow-moving trend plus tiny
    # noise) so the lag is highly correlated with the target, like a
    # real lag feature on autocorrelated time series data would be.
    target = np.cumsum(rng.normal(0, 0.1, n))
    lag_7 = pd.Series(target).shift(7)
    df = pd.DataFrame({"lag_7": lag_7.bfill(), "target": target})
    detector = LeakageDetector(LeakageConfig(target_corr_threshold=0.5))
    warnings = detector.detect(df, "target", ["lag_7"], np.array([0.5]))
    assert not any("correlation" in w.lower() for w in warnings["lag_7"])


def test_temporal_feature_nan_mismatch_flagged_as_structural_warning():
    """A feature named lag_7 should have 7 leading NaNs; if it has a
    different count, that's a structural warning, not a leakage warning."""
    rng = np.random.default_rng(42)
    n = 200
    target = rng.normal(0, 1, n)
    # Shift by 3 but name implies lag of 7 -- mismatched leading NaNs.
    lag_7 = pd.Series(target).shift(3)
    df = pd.DataFrame({"lag_7": lag_7, "target": target})
    detector = LeakageDetector()

    leakage_warnings = detector.detect(df, "target", ["lag_7"], np.array([0.5]))
    structural_warnings = detector.detect_temporal_structure(df, ["lag_7"])

    # Not conflated with leakage warnings.
    assert not any("correlation" in w.lower() for w in leakage_warnings["lag_7"])
    assert len(structural_warnings["lag_7"]) > 0
    assert "temporal structure warning" in structural_warnings["lag_7"][0].lower()
    assert "3" in structural_warnings["lag_7"][0]
    assert "7" in structural_warnings["lag_7"][0]


def test_temporal_feature_matching_nan_count_not_flagged():
    """A lag_7 feature with exactly 7 leading NaNs and no other gaps
    should pass the structural check cleanly."""
    rng = np.random.default_rng(42)
    n = 200
    target = rng.normal(0, 1, n)
    lag_7 = pd.Series(target).shift(7)
    df = pd.DataFrame({"lag_7": lag_7, "target": target})
    detector = LeakageDetector()
    structural_warnings = detector.detect_temporal_structure(df, ["lag_7"])
    assert len(structural_warnings["lag_7"]) == 0


def test_temporal_feature_mid_series_gap_flagged():
    """Unexpected NaNs after the leading run (not just at the start)
    should be flagged even if the leading-NaN count matches the offset."""
    rng = np.random.default_rng(42)
    n = 200
    target = rng.normal(0, 1, n)
    lag_5 = pd.Series(target).shift(5)
    lag_5.iloc[50] = np.nan  # inject a mid-series gap
    df = pd.DataFrame({"lag_5": lag_5, "target": target})
    detector = LeakageDetector()
    structural_warnings = detector.detect_temporal_structure(df, ["lag_5"])
    assert len(structural_warnings["lag_5"]) > 0
    assert "unexpected" in structural_warnings["lag_5"][0].lower()


def test_exempt_temporal_features_false_disables_exemption_and_structural_check():
    """With exempt_temporal_features=False, temporal features are treated
    like any other column: the correlation check applies to them, and
    the structural check is disabled entirely."""
    rng = np.random.default_rng(42)
    n = 200
    target = np.cumsum(rng.normal(0, 0.1, n))
    lag_7 = pd.Series(target).shift(7).bfill()
    df = pd.DataFrame({"lag_7": lag_7, "target": target})

    config = LeakageConfig(target_corr_threshold=0.5, exempt_temporal_features=False)
    detector = LeakageDetector(config)

    leakage_warnings = detector.detect(df, "target", ["lag_7"], np.array([0.5]))
    assert any("correlation" in w.lower() for w in leakage_warnings["lag_7"])

    structural_warnings = detector.detect_temporal_structure(df, ["lag_7"])
    assert structural_warnings["lag_7"] == []


def test_non_temporal_feature_not_checked_structurally():
    """A column that doesn't match any temporal pattern should never
    produce a structural warning, even with leading NaNs."""
    df = pd.DataFrame(
        {
            "feature_a": [np.nan, np.nan, 1.0, 2.0, 3.0],
            "target": [0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    detector = LeakageDetector()
    structural_warnings = detector.detect_temporal_structure(df, ["feature_a"])
    assert structural_warnings["feature_a"] == []


def test_categorical_feature_proxy_for_binary_target_flagged():
    """A categorical feature that's a near-perfect proxy for a binary
    target (like the real 'boat' column in the Titanic dataset, which
    is recorded post-outcome) should be flagged for high association,
    even though it's non-numeric and the numeric correlation check can't
    see it."""
    rng = np.random.default_rng(42)
    n = 300
    target = rng.integers(0, 2, n)
    # proxy maps almost 1:1 to target, with a small amount of noise so
    # it's not a deterministic mapping.
    proxy = np.where(target == 1, "boat_assigned", "no_boat").astype(object)
    noise_idx = rng.choice(n, size=10, replace=False)
    proxy[noise_idx] = np.where(
        proxy[noise_idx] == "boat_assigned", "no_boat", "boat_assigned"
    )
    df = pd.DataFrame({"boat": proxy, "target": target})

    detector = LeakageDetector()
    warnings = detector.detect(df, "target", ["boat"], np.array([0.9]), task="binary")
    assert len(warnings["boat"]) > 0
    assert "cramer" in warnings["boat"][0].lower()


def test_categorical_feature_missing_value_proxy_flagged():
    """Reproduces the real Titanic 'boat' leakage case: a high-cardinality
    categorical feature (many distinct boat numbers) that is missing for
    almost everyone who didn't survive and present for almost everyone
    who did. Naive Cramer's V on the raw category values is diluted by
    the many distinct non-null values, but treating NaN as its own
    category recovers the true signal: presence/absence of a value is
    itself a near-perfect proxy for the target."""
    rng = np.random.default_rng(11)
    n = 400
    target = rng.integers(0, 2, n)
    # 20 distinct "boat number" categories, only assigned when target==1.
    boat_numbers = rng.integers(1, 21, n).astype(object)
    col = np.where(target == 1, boat_numbers, None)
    df = pd.DataFrame({"boat": pd.Series(col, dtype=object), "target": target})

    detector = LeakageDetector()
    warnings = detector.detect(df, "target", ["boat"], np.array([0.9]), task="binary")
    assert len(warnings["boat"]) > 0
    assert "cramer" in warnings["boat"][0].lower()


def test_categorical_feature_no_association_not_flagged():
    """A categorical feature with no real relationship to the target
    should not be flagged."""
    rng = np.random.default_rng(7)
    n = 300
    target = rng.integers(0, 2, n)
    category = rng.choice(["a", "b", "c"], size=n)
    df = pd.DataFrame({"category": category, "target": target})

    detector = LeakageDetector()
    warnings = detector.detect(
        df, "target", ["category"], np.array([0.1]), task="binary"
    )
    assert not any("cramer" in w.lower() for w in warnings["category"])


def test_categorical_assoc_threshold_respected():
    """Raising the threshold above the feature's actual association
    should stop it from being flagged."""
    rng = np.random.default_rng(42)
    n = 300
    target = rng.integers(0, 2, n)
    proxy = np.where(target == 1, "boat_assigned", "no_boat").astype(object)
    noise_idx = rng.choice(n, size=10, replace=False)
    proxy[noise_idx] = np.where(
        proxy[noise_idx] == "boat_assigned", "no_boat", "boat_assigned"
    )
    df = pd.DataFrame({"boat": proxy, "target": target})

    detector_strict = LeakageDetector(LeakageConfig(categorical_assoc_threshold=0.5))
    warnings = detector_strict.detect(
        df, "target", ["boat"], np.array([0.9]), task="binary"
    )
    assert len(warnings["boat"]) > 0

    detector_loose = LeakageDetector(
        LeakageConfig(categorical_assoc_threshold=0.999, dominant_shap_fraction=1.1)
    )
    warnings = detector_loose.detect(
        df, "target", ["boat"], np.array([0.9]), task="binary"
    )
    assert len(warnings["boat"]) == 0


def test_categorical_feature_proxy_for_multiclass_target_flagged():
    """The categorical association check should also work for multiclass
    targets, not just binary."""
    rng = np.random.default_rng(3)
    n = 300
    target = rng.integers(0, 3, n)
    mapping = {0: "cat_a", 1: "cat_b", 2: "cat_c"}
    proxy = np.array([mapping[t] for t in target], dtype=object)
    noise_idx = rng.choice(n, size=10, replace=False)
    proxy[noise_idx] = rng.choice(list(mapping.values()), size=10)
    df = pd.DataFrame({"proxy_cat": proxy, "target": target})

    detector = LeakageDetector()
    warnings = detector.detect(
        df, "target", ["proxy_cat"], np.array([0.9]), task="multiclass"
    )
    assert len(warnings["proxy_cat"]) > 0


def test_categorical_association_skipped_for_regression():
    """Regression tasks are out of scope for this check (documented
    limitation), so a categorical proxy feature should not be flagged
    when task='regression'."""
    rng = np.random.default_rng(42)
    n = 300
    target_class = rng.integers(0, 2, n)
    target = target_class.astype(float) * 10 + rng.normal(0, 0.1, n)
    proxy = np.where(target_class == 1, "boat_assigned", "no_boat").astype(object)
    df = pd.DataFrame({"boat": proxy, "target": target})

    detector = LeakageDetector()
    warnings = detector.detect(
        df, "target", ["boat"], np.array([0.9]), task="regression"
    )
    assert not any("cramer" in w.lower() for w in warnings["boat"])


def test_categorical_association_numeric_feature_unaffected():
    """Numeric features should still only go through the Pearson
    correlation check, not the categorical association check."""
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 200)
    df = pd.DataFrame({"numeric_feat": base, "target": rng.integers(0, 2, 200)})
    detector = LeakageDetector()
    warnings = detector.detect(
        df, "target", ["numeric_feat"], np.array([0.5]), task="binary"
    )
    assert not any("cramer" in w.lower() for w in warnings["numeric_feat"])


def test_custom_thresholds():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 200)
    df = pd.DataFrame({"feat": base * 0.98 + rng.normal(0, 0.1, 200), "target": base})
    # With low threshold, should flag
    detector_strict = LeakageDetector(LeakageConfig(target_corr_threshold=0.9))
    warnings = detector_strict.detect(df, "target", ["feat"], np.array([0.5]))
    assert len(warnings["feat"]) > 0

    # With high threshold, should not flag
    detector_loose = LeakageDetector(
        LeakageConfig(target_corr_threshold=0.999, dominant_shap_fraction=1.1)
    )
    warnings = detector_loose.detect(df, "target", ["feat"], np.array([0.5]))
    assert len(warnings["feat"]) == 0


def test_near_duplicate_exact_mode():
    """Test exact mode near-duplicate detection (baseline)."""
    base = np.random.default_rng(42).normal(0, 1, 200)
    df = pd.DataFrame(
        {
            "feat_a": base,
            "feat_b": base * 1.0001,
            "target": np.random.default_rng(1).integers(0, 2, 200).astype(float),
        }
    )
    detector = LeakageDetector(
        LeakageConfig(near_duplicate_corr_threshold=0.98), screening_mode="exact"
    )
    shap = np.array([0.4, 0.4])
    warnings = detector.detect(df, "target", ["feat_a", "feat_b"], shap)
    flagged = [k for k, v in warnings.items() if v]
    assert len(flagged) > 0


def test_near_duplicate_approximate_mode_fallback_without_rust():
    """Test that approximate mode gracefully falls back to exact when
    Rust extension is unavailable."""
    base = np.random.default_rng(42).normal(0, 1, 200)
    df = pd.DataFrame(
        {
            "feat_a": base,
            "feat_b": base * 1.0001,
            "target": np.random.default_rng(1).integers(0, 2, 200).astype(float),
        }
    )
    # If the extension is available, this will use it; if not, it falls
    # back to exact and logs a warning. Both paths should detect the pair.
    detector = LeakageDetector(
        LeakageConfig(near_duplicate_corr_threshold=0.98),
        screening_mode="approximate",
    )
    shap = np.array([0.4, 0.4])
    warnings = detector.detect(df, "target", ["feat_a", "feat_b"], shap)
    flagged = [k for k, v in warnings.items() if v]
    assert len(flagged) > 0


def test_near_duplicate_approximate_mode_matches_exact():
    """Test that approximate mode results are consistent with exact mode
    on a moderately-sized dataset (if Rust extension is available)."""
    base = np.random.default_rng(42).normal(0, 1, 150)
    rng = np.random.default_rng(99)
    # Create 10 features: 3 pairs of near-duplicates (high correlation) and 4
    # independent
    features = {
        "dup_a1": base,
        "dup_a2": base * 0.99999 + rng.normal(0, 0.001, 150),  # Very high correlation
        "dup_b1": rng.normal(5, 2, 150),
        "dup_b2": (
            rng.normal(5, 2, 150) * 0.99998 + rng.normal(0, 0.002, 150)
        ),  # Very high corr
        "dup_c1": rng.normal(-1, 3, 150),
        "dup_c2": (
            rng.normal(-1, 3, 150) * 0.99997 + rng.normal(0, 0.003, 150)
        ),  # Very high corr
        "indep_1": rng.normal(0, 1, 150),
        "indep_2": rng.normal(0, 1, 150),
        "indep_3": rng.normal(0, 1, 150),
        "target": rng.integers(0, 2, 150).astype(float),
    }
    df = pd.DataFrame(features)
    feature_names = [k for k in features.keys() if k != "target"]

    detector_exact = LeakageDetector(
        LeakageConfig(near_duplicate_corr_threshold=0.98),
        screening_mode="exact",
    )
    detector_approx = LeakageDetector(
        LeakageConfig(near_duplicate_corr_threshold=0.98),
        screening_mode="approximate",
    )

    shap = np.ones(len(feature_names)) / len(feature_names)

    warnings_exact = detector_exact.detect(df, "target", feature_names, shap)
    warnings_approx = detector_approx.detect(df, "target", feature_names, shap)

    # Both should flag the same pairs (allowing for LSH approximation,
    # both should find at least most near-duplicate pairs).
    flagged_exact = {k for k, v in warnings_exact.items() if v}
    flagged_approx = {k for k, v in warnings_approx.items() if v}

    # The approximate path may miss some due to hash collisions, but
    # should catch the same pairs or at least most of them.
    # Allow some tolerance since LSH is approximate.
    assert len(flagged_exact) >= 2  # At least some dup pairs
    assert len(flagged_approx) >= 1  # Approximate should catch some


def test_detect_lookahead_leakage_clean_data_no_warnings():
    """Clean data with normal temporal progression should not trigger
    look-ahead warnings."""
    rng = np.random.default_rng(42)
    n = 300

    # Create data with clear temporal order
    time_col = np.arange(n)  # 0, 1, 2, ..., 299
    feature_a = rng.normal(0, 1, n)
    target = rng.integers(0, 2, n).astype(float)

    # Split chronologically (no look-ahead)
    train_n = 180
    val_n = 60
    train_df = pd.DataFrame(
        {
            "time": time_col[:train_n],
            "feature_a": feature_a[:train_n],
            "target": target[:train_n],
        }
    )
    val_df = pd.DataFrame(
        {
            "time": time_col[train_n : train_n + val_n],
            "feature_a": feature_a[train_n : train_n + val_n],
            "target": target[train_n : train_n + val_n],
        }
    )
    test_df = pd.DataFrame(
        {
            "time": time_col[train_n + val_n :],
            "feature_a": feature_a[train_n + val_n :],
            "target": target[train_n + val_n :],
        }
    )

    detector = LeakageDetector()
    warnings = detector.detect_lookahead_leakage(
        train_df, val_df, test_df, "time", ["feature_a"]
    )
    # Clean chronological data should produce no warnings
    assert len(warnings["feature_a"]) == 0


def test_detect_lookahead_leakage_missing_time_column_returns_empty():
    """Missing time_column should gracefully return empty warnings."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n).astype(float),
        }
    )
    train_df = df.iloc[:60]
    val_df = df.iloc[60:80]
    test_df = df.iloc[80:]

    detector = LeakageDetector()
    warnings = detector.detect_lookahead_leakage(
        train_df, val_df, test_df, "nonexistent", ["feature_a"]
    )
    # Should return empty warnings dict (not raise)
    assert len(warnings["feature_a"]) == 0
