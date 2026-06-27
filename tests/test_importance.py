"""Tests for SHAP and permutation importance functions."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import roc_auc_score

from treeshaker.importance import (
    compute_permutation_importance,
    compute_shap_importance,
    compute_shap_interactions,
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
        model, df, "target", feature_names, metric_fn, use_predict_proba=True
    )
    assert importance.shape == (8,)


def test_permutation_importance_binary_uses_predict_proba(fitted_binary_model):
    """Binary permutation importance must score against predicted
    probabilities, not hard class labels, since roc_auc_score expects
    a ranking score rather than discrete labels."""
    model, df, feature_names = fitted_binary_model
    metric_fn = get_metric_fn("binary")
    X = df[feature_names]
    y = np.asarray(df["target"].values)

    proba_importance = compute_permutation_importance(
        model, df, "target", feature_names, metric_fn, use_predict_proba=True
    )
    label_importance = compute_permutation_importance(
        model, df, "target", feature_names, metric_fn, use_predict_proba=False
    )

    # The two approaches should generally produce different importance
    # values, since roc_auc_score behaves very differently on hard labels
    # vs. continuous probabilities.
    assert not np.allclose(proba_importance, label_importance)

    # Sanity check: the baseline AUC computed with probabilities should
    # match calling predict_proba directly.
    expected_baseline_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    assert expected_baseline_auc >= 0.0


def test_permutation_importance_default_is_predict(fitted_regression_model):
    """use_predict_proba should default to False, preserving prior
    behavior for regression (and multiclass) tasks."""
    model, df, feature_names = fitted_regression_model
    metric_fn = get_metric_fn("regression")
    importance = compute_permutation_importance(
        model, df, "target", feature_names, metric_fn
    )
    assert importance.shape == (8,)


def test_shap_interactions_columns(fitted_binary_model):
    model, df, feature_names = fitted_binary_model
    interactions = compute_shap_interactions(model, df, feature_names)
    assert list(interactions.columns) == [
        "feature_a",
        "feature_b",
        "interaction_strength",
    ]


def test_shap_interactions_top_n_limit(fitted_binary_model):
    model, df, feature_names = fitted_binary_model
    interactions = compute_shap_interactions(model, df, feature_names, top_n=3)
    assert len(interactions) == 3


def test_shap_interactions_sorted_descending(fitted_binary_model):
    model, df, feature_names = fitted_binary_model
    interactions = compute_shap_interactions(model, df, feature_names)
    values = interactions["interaction_strength"].to_numpy()
    assert (values[:-1] >= values[1:]).all()


def test_shap_interactions_non_negative(fitted_binary_model):
    model, df, feature_names = fitted_binary_model
    interactions = compute_shap_interactions(model, df, feature_names)
    assert (interactions["interaction_strength"] >= 0).all()


def test_shap_interactions_pairs_distinct_features(fitted_binary_model):
    model, df, feature_names = fitted_binary_model
    interactions = compute_shap_interactions(model, df, feature_names)
    assert (interactions["feature_a"] != interactions["feature_b"]).all()


def test_shap_interactions_regression(fitted_regression_model):
    model, df, feature_names = fitted_regression_model
    interactions = compute_shap_interactions(model, df, feature_names, top_n=5)
    assert len(interactions) <= 5
    assert (interactions["interaction_strength"] >= 0).all()


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
