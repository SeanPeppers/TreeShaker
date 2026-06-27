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
