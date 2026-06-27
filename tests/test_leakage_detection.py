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
