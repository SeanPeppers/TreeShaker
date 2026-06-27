"""Tests for utility functions in utils.py."""

import pandas as pd

from treeshaker.utils import check_class_imbalance


def test_balanced_binary_returns_none():
    df = pd.DataFrame({"target": [0, 1] * 50})
    assert check_class_imbalance(df, "target", "binary") is None


def test_imbalanced_binary_returns_warning():
    df = pd.DataFrame({"target": [0] * 99 + [1] * 1})
    warning = check_class_imbalance(df, "target", "binary")
    assert warning is not None
    assert "imbalance" in warning.lower()
    assert "1.0%" in warning


def test_borderline_binary_at_threshold_is_balanced():
    # Exactly at the default 10% threshold should not warn (>= threshold).
    df = pd.DataFrame({"target": [0] * 90 + [1] * 10})
    assert check_class_imbalance(df, "target", "binary") is None


def test_borderline_binary_just_below_threshold_warns():
    df = pd.DataFrame({"target": [0] * 91 + [1] * 9})
    assert check_class_imbalance(df, "target", "binary") is not None


def test_balanced_multiclass_returns_none():
    df = pd.DataFrame({"target": [0, 1, 2] * 50})
    assert check_class_imbalance(df, "target", "multiclass") is None


def test_imbalanced_multiclass_returns_warning():
    df = pd.DataFrame({"target": [0] * 90 + [1] * 8 + [2] * 2})
    warning = check_class_imbalance(df, "target", "multiclass")
    assert warning is not None
    assert "imbalance" in warning.lower()


def test_regression_always_returns_none():
    df = pd.DataFrame({"target": [0] * 99 + [1] * 1})
    assert check_class_imbalance(df, "target", "regression") is None


def test_custom_threshold():
    df = pd.DataFrame({"target": [0] * 80 + [1] * 20})
    # 20% minority is fine under default 10% threshold.
    assert check_class_imbalance(df, "target", "binary") is None
    # But should warn under a stricter custom threshold.
    warning = check_class_imbalance(
        df, "target", "binary", imbalance_warning_threshold=0.25
    )
    assert warning is not None
