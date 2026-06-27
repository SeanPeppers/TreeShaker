"""SHAP and permutation importance computation."""

import logging

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_shap_importance(
    model,
    data: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """
    Compute mean absolute SHAP values for each feature.

    Uses TreeExplainer for efficient computation with tree-based models.
    For multiclass problems, SHAP values are averaged across all classes.

    Args:
        model: A fitted XGBoost model (XGBClassifier or XGBRegressor).
        data: Feature DataFrame (no target column) used for explanation.
        feature_names: Ordered list of feature column names.

    Returns:
        Array of shape (n_features,) with mean absolute SHAP value per feature.
    """
    logger.info("Computing SHAP importance")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data[feature_names])

    if isinstance(shap_values, list):
        # Older shap versions: multiclass returns a list of arrays, one per
        # class, each shaped (n_samples, n_features).
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        mean_abs = stacked.mean(axis=0).mean(axis=0)
    elif shap_values.ndim == 3:
        # Newer shap versions: multiclass returns a single ndarray shaped
        # (n_samples, n_features, n_classes). Average over samples and
        # classes to collapse to one value per feature.
        mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=-1)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    return mean_abs


def compute_permutation_importance(
    model,
    data: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    metric_fn,
) -> np.ndarray:
    """
    Compute permutation importance for each feature.

    For each feature, shuffles its values and measures the drop in model
    performance. A larger drop means the feature is more important.

    Args:
        model: A fitted XGBoost model (XGBClassifier or XGBRegressor).
        data: DataFrame including both features and the target column.
        target_column: Name of the target column in data.
        feature_names: Ordered list of feature column names.
        metric_fn: Callable with signature (y_true, y_pred) -> float,
            where higher values indicate better performance.

    Returns:
        Array of shape (n_features,) with permutation importance per feature.
        Positive values mean removing the feature hurts performance.
    """
    logger.info("Computing permutation importance")
    X = data[feature_names].copy()
    y = data[target_column].values

    baseline_preds = model.predict(X)
    baseline_score = metric_fn(y, baseline_preds)

    importances = np.zeros(len(feature_names))
    for i, feature in enumerate(feature_names):
        X_shuffled = X.copy()
        rng = np.random.default_rng(seed=42 + i)
        X_shuffled[feature] = rng.permutation(np.asarray(X_shuffled[feature].values))
        shuffled_preds = model.predict(X_shuffled)
        shuffled_score = metric_fn(y, shuffled_preds)
        importances[i] = baseline_score - shuffled_score

    return importances
