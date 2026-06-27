"""Tests for SHAP and permutation importance functions."""

import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression

from treeshaker.importance import (
    compute_permutation_importance,
    compute_shap_importance,
)
from treeshaker.utils import get_metric_fn


@pytest.fixture
def fitted_binary_model():
    """Return a fitted XGBClassifier and matching data."""
    X, y = make_classification(
        n_samples=150, n_features=8, n_informative=5, random_state=42
    )
    feature_names = [f"f{i}" for i in range(8)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    model = xgb.XGBClassifier(n_estimators=20, random_state=42, verbosity=0)
    model.fit(df[feature_names], df["target"])
    return model, df, feature_names


@pytest.fixture
def fitted_regression_model():
    """Return a fitted XGBRegressor and matching data."""
    X, y = make_regression(
        n_samples=150, n_features=8, n_informative=5, random_state=42
    )
    feature_names = [f"f{i}" for i in range(8)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    model = xgb.XGBRegressor(n_estimators=20, random_state=42, verbosity=0)
    model.fit(df[feature_names], df["target"])
    return model, df, feature_names


def test_shap_importance_shape(fitted_binary_model):
    model, df, feature_names = fitted_binary_model
    importance = compute_shap_importance(model, df, feature_names)
    assert importance.shape == (8,)


def test_shap_importance_non_negative(fitted_binary_model):
    model, df, feature_names = fitted_binary_model
    importance = compute_shap_importance(model, df, feature_names)
    assert (importance >= 0).all()


def test_shap_importance_regression(fitted_regression_model):
    model, df, feature_names = fitted_regression_model
    importance = compute_shap_importance(model, df, feature_names)
    assert importance.shape == (8,)
    assert (importance >= 0).all()


def test_permutation_importance_shape(fitted_binary_model):
    model, df, feature_names = fitted_binary_model
    metric_fn = get_metric_fn("binary")
    importance = compute_permutation_importance(
        model, df, "target", feature_names, metric_fn
    )
    assert importance.shape == (8,)


def test_permutation_importance_informative_features_positive(
    fitted_regression_model,
):
    """Informative features should have positive permutation importance."""
    model, df, feature_names = fitted_regression_model
    metric_fn = get_metric_fn("regression")
    importance = compute_permutation_importance(
        model, df, "target", feature_names, metric_fn
    )
    # At least some features should have positive importance
    assert (importance > 0).any()
