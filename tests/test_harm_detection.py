"""Tests for HarmDetector in harm_detection.py."""

import numpy as np
import pytest

from treeshaker.harm_detection import HarmDetector


@pytest.fixture
def detector():
    return HarmDetector(shap_importance_threshold=0.01, perm_importance_threshold=0.05)


def test_no_harmful_features(detector):
    shap = np.array([0.5, 0.3, 0.2])
    perm = np.array([0.4, 0.2, 0.1])
    is_harmful, reasons = detector.flag_harmful_features(shap, perm)
    assert not is_harmful.any()
    assert all(r == "" for r in reasons)


def test_low_shap_flagged(detector):
    shap = np.array([0.5, 0.005, 0.2])
    perm = np.array([0.4, 0.2, 0.1])
    is_harmful, reasons = detector.flag_harmful_features(shap, perm)
    assert is_harmful[1]
    assert "almost no weight" in reasons[1]
    assert not is_harmful[0]
    assert not is_harmful[2]


def test_low_perm_flagged(detector):
    shap = np.array([0.5, 0.3, 0.2])
    perm = np.array([0.4, 0.01, 0.1])
    is_harmful, reasons = detector.flag_harmful_features(shap, perm)
    assert is_harmful[1]
    assert "Shuffling" in reasons[1]


def test_both_signals_flagged(detector):
    shap = np.array([0.5, 0.001])
    perm = np.array([0.4, 0.001])
    is_harmful, reasons = detector.flag_harmful_features(shap, perm)
    assert is_harmful[1]
    assert "ignores" in reasons[1]


def test_custom_thresholds():
    detector = HarmDetector(
        shap_importance_threshold=0.1, perm_importance_threshold=0.2
    )
    shap = np.array([0.05])
    perm = np.array([0.3])
    is_harmful, reasons = detector.flag_harmful_features(shap, perm)
    assert is_harmful[0]
    assert "almost no weight" in reasons[0]


def test_output_lengths_match():
    detector = HarmDetector()
    shap = np.array([0.1, 0.2, 0.3, 0.4])
    perm = np.array([0.1, 0.2, 0.3, 0.4])
    is_harmful, reasons = detector.flag_harmful_features(shap, perm)
    assert len(is_harmful) == 4
    assert len(reasons) == 4


def test_low_perm_not_flagged_under_severe_imbalance(detector):
    # Regression test for the fraud-detection bug: under severe class
    # imbalance, permutation importance collapses toward zero for every
    # feature even when SHAP shows the model relies heavily on it (e.g.
    # V4 in the Credit Card Fraud Detection dataset). A high-SHAP,
    # low-perm feature must be KEPT, not dropped, when severe_class_imbalance
    # is True.
    shap = np.array([0.816, 0.3])
    perm = np.array([0.001, 0.001])
    is_harmful, reasons = detector.flag_harmful_features(
        shap, perm, severe_class_imbalance=True
    )
    assert not is_harmful[0]
    assert reasons[0] == ""
    assert not is_harmful[1]


def test_low_perm_still_flagged_when_imbalance_false(detector):
    # Default behavior (severe_class_imbalance=False) must be unchanged
    # for balanced datasets -- low perm importance alone still flags harm.
    shap = np.array([0.5])
    perm = np.array([0.01])
    is_harmful, reasons = detector.flag_harmful_features(
        shap, perm, severe_class_imbalance=False
    )
    assert is_harmful[0]
    assert "Shuffling" in reasons[0]


def test_both_low_still_flagged_under_severe_imbalance(detector):
    # Even under severe imbalance, a feature with BOTH low SHAP and low
    # perm importance is still harmful -- imbalance only protects features
    # SHAP says are meaningful.
    shap = np.array([0.001])
    perm = np.array([0.001])
    is_harmful, reasons = detector.flag_harmful_features(
        shap, perm, severe_class_imbalance=True
    )
    assert is_harmful[0]
    assert "ignores" in reasons[0]


def test_low_shap_still_flagged_under_severe_imbalance(detector):
    # Low SHAP alone is still sufficient to flag a feature as harmful
    # regardless of class imbalance, since SHAP remains reliable.
    shap = np.array([0.005])
    perm = np.array([0.4])
    is_harmful, reasons = detector.flag_harmful_features(
        shap, perm, severe_class_imbalance=True
    )
    assert is_harmful[0]
    assert "almost no weight" in reasons[0]
