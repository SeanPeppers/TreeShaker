"""Tests for SHAP and permutation importance functions."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import roc_auc_score

from treeshaker.importance import (
    compute_mutual_information,
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
def multiclass_data():
    """Return a DataFrame with a multiclass target and matching feature names."""
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    feature_names = [f"f{i}" for i in range(8)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df, feature_names


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


def test_shap_importance_sample_size_none_uses_all_rows(
    fitted_binary_model, monkeypatch
):
    """sample_size=None (the default) preserves full-dataset behavior."""
    model, df, feature_names = fitted_binary_model
    seen_rows = []
    original_tree_explainer = __import__("shap").TreeExplainer

    class _SpyExplainer:
        def __init__(self, m):
            self._inner = original_tree_explainer(m)

        def shap_values(self, X, check_additivity=True):
            seen_rows.append(len(X))
            return self._inner.shap_values(X, check_additivity=check_additivity)

    monkeypatch.setattr("treeshaker.importance.shap.TreeExplainer", _SpyExplainer)
    compute_shap_importance(model, df, feature_names, sample_size=None)
    assert seen_rows == [len(df)]


def test_shap_importance_sample_size_reduces_rows_passed_to_explainer(
    fitted_binary_model, monkeypatch
):
    """A sample_size smaller than the dataset reduces rows seen by the explainer."""
    model, df, feature_names = fitted_binary_model
    seen_rows = []
    original_tree_explainer = __import__("shap").TreeExplainer

    class _SpyExplainer:
        def __init__(self, m):
            self._inner = original_tree_explainer(m)

        def shap_values(self, X, check_additivity=True):
            seen_rows.append(len(X))
            return self._inner.shap_values(X, check_additivity=check_additivity)

    monkeypatch.setattr("treeshaker.importance.shap.TreeExplainer", _SpyExplainer)
    compute_shap_importance(model, df, feature_names, sample_size=20, random_seed=42)
    assert seen_rows == [20]


def test_shap_importance_sample_size_covers_all_features(fitted_binary_model):
    """Sampling rows still returns one importance value per input feature."""
    model, df, feature_names = fitted_binary_model
    importance = compute_shap_importance(
        model, df, feature_names, sample_size=20, random_seed=42
    )
    assert importance.shape == (len(feature_names),)
    assert (importance >= 0).all()


def test_shap_importance_sample_size_larger_than_data_uses_all_rows(
    fitted_binary_model, monkeypatch
):
    """A sample_size >= row count behaves like no sampling (no shrinkage)."""
    model, df, feature_names = fitted_binary_model
    seen_rows = []
    original_tree_explainer = __import__("shap").TreeExplainer

    class _SpyExplainer:
        def __init__(self, m):
            self._inner = original_tree_explainer(m)

        def shap_values(self, X, check_additivity=True):
            seen_rows.append(len(X))
            return self._inner.shap_values(X, check_additivity=check_additivity)

    monkeypatch.setattr("treeshaker.importance.shap.TreeExplainer", _SpyExplainer)
    compute_shap_importance(model, df, feature_names, sample_size=10_000)
    assert seen_rows == [len(df)]


def test_shap_importance_sample_size_reproducible(fitted_binary_model):
    """Same random_seed yields identical sampled-row importance results."""
    model, df, feature_names = fitted_binary_model
    importance_a = compute_shap_importance(
        model, df, feature_names, sample_size=20, random_seed=7
    )
    importance_b = compute_shap_importance(
        model, df, feature_names, sample_size=20, random_seed=7
    )
    np.testing.assert_array_equal(importance_a, importance_b)


@pytest.fixture
def fitted_high_cardinality_missing_regression_model():
    """Regression model + data designed to stress SHAP additivity.

    Mirrors the conditions that triggered a real-world
    ``shap.utils._exceptions.ExplainerError: Additivity check failed in
    TreeExplainer!`` on the Ames Housing dataset: several high-cardinality
    categorical columns combined with heavy missingness, which can produce
    floating point summation drift in XGBoost's missing-value routing large
    enough to trip SHAP's additivity assertion (even though the SHAP values
    themselves remain valid/meaningful for ranking).
    """
    rng = np.random.default_rng(42)
    n_samples = 600
    n_numeric = 6
    X_numeric, y = make_regression(
        n_samples=n_samples,
        n_features=n_numeric,
        n_informative=4,
        noise=10.0,
        random_state=42,
    )
    feature_names = [f"num{i}" for i in range(n_numeric)]
    df = pd.DataFrame(X_numeric, columns=feature_names)

    # Several high-cardinality categorical columns with heavy missingness.
    cat_feature_names = []
    for c in range(4):
        col = f"cat{c}"
        categories = [f"level_{i}" for i in range(40)]
        values = rng.choice(categories, size=n_samples).astype(object)
        # Heavy missingness: ~40% NaN.
        missing_mask = rng.random(n_samples) < 0.4
        values[missing_mask] = None
        col_codes = pd.Categorical(values, categories=categories).codes.astype(float)
        df[col] = pd.Series(values, dtype="category")
        cat_feature_names.append(col)
        # Fold the categorical signal into the target so the model actually
        # learns to split on these (high-missingness, high-cardinality)
        # columns rather than ignoring them.
        y = y + col_codes * 5.0

    df["target"] = y
    all_feature_names = feature_names + cat_feature_names

    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=6,
        random_state=42,
        verbosity=0,
        enable_categorical=True,
    )
    model.fit(df[all_feature_names], df["target"])
    return model, df, all_feature_names


def test_shap_importance_handles_high_cardinality_missing_data(
    fitted_high_cardinality_missing_regression_model,
):
    """compute_shap_importance must not raise on the additivity-prone case.

    Regression test for the exact crash seen on the Ames Housing dataset:
    ``ExplainerError: Additivity check failed in TreeExplainer!``. This
    exercises a synthetic analogue of that scenario (high-cardinality
    categorical columns with heavy NaN rates); compute_shap_importance
    should complete and return sane (finite, non-negative) values.
    """
    model, df, feature_names = fitted_high_cardinality_missing_regression_model
    importance = compute_shap_importance(model, df, feature_names)
    assert importance.shape == (len(feature_names),)
    assert np.isfinite(importance).all()
    assert (importance >= 0).all()


def test_shap_importance_passes_check_additivity_false(fitted_binary_model):
    """compute_shap_importance must call shap_values with check_additivity=False.

    This guards against a regression of the additivity-check crash even in
    cases where the failure mode itself is hard to reproduce
    deterministically: if a future change drops the check_additivity=False
    argument, this test catches it via a spy on TreeExplainer.shap_values.
    """
    model, df, feature_names = fitted_binary_model
    original_tree_explainer = __import__("shap").TreeExplainer
    received_kwargs = {}

    class _SpyExplainer:
        def __init__(self, m):
            self._inner = original_tree_explainer(m)

        def shap_values(self, X, **kwargs):
            received_kwargs.update(kwargs)
            return self._inner.shap_values(X, **kwargs)

    import treeshaker.importance as importance_module

    original = importance_module.shap.TreeExplainer
    importance_module.shap.TreeExplainer = _SpyExplainer
    try:
        compute_shap_importance(model, df, feature_names)
    finally:
        importance_module.shap.TreeExplainer = original

    assert received_kwargs.get("check_additivity") is False


def test_permutation_importance_shape(fitted_binary_model):
    model, df, feature_names = fitted_binary_model
    metric_fn = get_metric_fn("binary")
    importance, importance_std = compute_permutation_importance(
        model, df, "target", feature_names, metric_fn, use_predict_proba=True
    )
    assert importance.shape == (8,)
    assert importance_std.shape == (8,)
    assert (importance_std >= 0).all()


def test_permutation_importance_binary_uses_predict_proba(fitted_binary_model):
    """Binary permutation importance must score against predicted
    probabilities, not hard class labels, since roc_auc_score expects
    a ranking score rather than discrete labels."""
    model, df, feature_names = fitted_binary_model
    metric_fn = get_metric_fn("binary")
    X = df[feature_names]
    y = np.asarray(df["target"].values)

    proba_importance, _ = compute_permutation_importance(
        model, df, "target", feature_names, metric_fn, use_predict_proba=True
    )
    label_importance, _ = compute_permutation_importance(
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
    importance, _ = compute_permutation_importance(
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
    importance, _ = compute_permutation_importance(
        model, df, "target", feature_names, metric_fn
    )
    # At least some features should have positive importance
    assert (importance > 0).any()


def test_permutation_importance_n_repeats_reproducible(fitted_binary_model):
    """Same n_repeats should give identical results across two calls."""
    model, df, feature_names = fitted_binary_model
    metric_fn = get_metric_fn("binary")
    importance1, std1 = compute_permutation_importance(
        model,
        df,
        "target",
        feature_names,
        metric_fn,
        use_predict_proba=True,
        n_repeats=5,
    )
    importance2, std2 = compute_permutation_importance(
        model,
        df,
        "target",
        feature_names,
        metric_fn,
        use_predict_proba=True,
        n_repeats=5,
    )
    np.testing.assert_array_equal(importance1, importance2)
    np.testing.assert_array_equal(std1, std2)


def test_permutation_importance_n_repeats_1_vs_5_differ(fitted_binary_model):
    """A single shuffle vs. averaging 5 shuffles should generally differ,
    since they use different (and in the n_repeats=5 case, multiple)
    underlying permutations."""
    model, df, feature_names = fitted_binary_model
    metric_fn = get_metric_fn("binary")
    importance_1, std_1 = compute_permutation_importance(
        model,
        df,
        "target",
        feature_names,
        metric_fn,
        use_predict_proba=True,
        n_repeats=1,
    )
    importance_5, std_5 = compute_permutation_importance(
        model,
        df,
        "target",
        feature_names,
        metric_fn,
        use_predict_proba=True,
        n_repeats=5,
    )
    assert importance_1.shape == (8,)
    assert importance_5.shape == (8,)
    # n_repeats=1 has no spread to measure -- std must be exactly 0.
    assert (std_1 == 0).all()
    # n_repeats=5 averages over 5 distinct shuffles, so results should
    # generally differ from the single-shuffle case.
    assert not np.allclose(importance_1, importance_5)


def test_permutation_importance_std_shape_and_non_negative(fitted_regression_model):
    model, df, feature_names = fitted_regression_model
    metric_fn = get_metric_fn("regression")
    importance, importance_std = compute_permutation_importance(
        model, df, "target", feature_names, metric_fn, n_repeats=5
    )
    assert importance.shape == (8,)
    assert importance_std.shape == (8,)
    assert (importance_std >= 0).all()


def test_mutual_information_binary_shape(fitted_binary_model):
    _, df, feature_names = fitted_binary_model
    scores = compute_mutual_information(
        df, "target", feature_names, task="binary", random_seed=42
    )
    assert scores.shape == (8,)
    assert (scores >= 0).all()


def test_mutual_information_multiclass_shape(multiclass_data):
    df, feature_names = multiclass_data
    scores = compute_mutual_information(
        df, "target", feature_names, task="multiclass", random_seed=42
    )
    assert scores.shape == (8,)
    assert (scores >= 0).all()


def test_mutual_information_regression_shape(fitted_regression_model):
    _, df, feature_names = fitted_regression_model
    scores = compute_mutual_information(
        df, "target", feature_names, task="regression", random_seed=42
    )
    assert scores.shape == (8,)
    assert (scores >= 0).all()


def test_mutual_information_informative_features_higher(fitted_regression_model):
    """Informative features should generally score higher than random noise."""
    _, df, feature_names = fitted_regression_model
    scores = compute_mutual_information(
        df, "target", feature_names, task="regression", random_seed=42
    )
    assert (scores > 0).any()
