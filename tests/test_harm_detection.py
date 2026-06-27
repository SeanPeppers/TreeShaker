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
    assert "SHAP" in reasons[1]
    assert not is_harmful[0]
    assert not is_harmful[2]


def test_low_perm_flagged(detector):
    shap = np.array([0.5, 0.3, 0.2])
    perm = np.array([0.4, 0.01, 0.1])
    is_harmful, reasons = detector.flag_harmful_features(shap, perm)
    assert is_harmful[1]
    assert "permutation" in reasons[1]


def test_both_signals_flagged(detector):
    shap = np.array([0.5, 0.001])
    perm = np.array([0.4, 0.001])
    is_harmful, reasons = detector.flag_harmful_features(shap, perm)
    assert is_harmful[1]
    assert "SHAP" in reasons[1]
    assert "permutation" in reasons[1]


def test_custom_thresholds():
    detector = HarmDetector(
        shap_importance_threshold=0.1, perm_importance_threshold=0.2
    )
    shap = np.array([0.05])
    perm = np.array([0.3])
    is_harmful, reasons = detector.flag_harmful_features(shap, perm)
    assert is_harmful[0]
    assert "SHAP" in reasons[0]


def test_output_lengths_match():
    detector = HarmDetector()
    shap = np.array([0.1, 0.2, 0.3, 0.4])
    perm = np.array([0.1, 0.2, 0.3, 0.4])
    is_harmful, reasons = detector.flag_harmful_features(shap, perm)
    assert len(is_harmful) == 4
    assert len(reasons) == 4
