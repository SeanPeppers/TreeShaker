"""SHAP and permutation importance computation."""

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_shap_importance(
    model: Any,
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


def compute_shap_interactions(
    model: Any,
    data: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compute pairwise SHAP interaction strengths between features.

    Uses TreeExplainer.shap_interaction_values, which is significantly more
    expensive than plain SHAP values (roughly O(features^2) per sample), so
    this is intended to be called only when interaction analysis is
    explicitly requested.

    The raw interaction tensor has shape (n_samples, n_features, n_features).
    It is reduced by taking the mean over samples and the absolute value,
    producing an (n_features, n_features) matrix. The diagonal represents
    each feature's main effect; off-diagonal entries represent pairwise
    interaction strength. Since the matrix is symmetric, each unordered
    pair is reported once.

    Args:
        model: A fitted XGBoost model (XGBClassifier or XGBRegressor).
        data: Feature DataFrame (no target column) used for explanation.
        feature_names: Ordered list of feature column names.
        top_n: Number of strongest interacting pairs to return.

    Returns:
        DataFrame with columns feature_a, feature_b, interaction_strength,
        sorted by interaction_strength descending, limited to top_n rows.
    """
    logger.info("Computing SHAP interaction values (this can be slow)")
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(data[feature_names])

    if isinstance(interaction_values, list):
        # Older shap versions: multiclass returns a list of arrays, one per
        # class, each shaped (n_samples, n_features, n_features).
        stacked = np.stack([np.abs(iv) for iv in interaction_values], axis=0)
        mean_abs = stacked.mean(axis=0).mean(axis=0)
    elif interaction_values.ndim == 4:
        # Newer shap versions: multiclass returns a single ndarray shaped
        # (n_samples, n_features, n_features, n_classes).
        mean_abs = np.abs(interaction_values).mean(axis=0).mean(axis=-1)
    else:
        mean_abs = np.abs(interaction_values).mean(axis=0)

    n_features = len(feature_names)
    rows = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            rows.append((feature_names[i], feature_names[j], mean_abs[i, j]))

    result = pd.DataFrame(
        rows, columns=["feature_a", "feature_b", "interaction_strength"]
    )
    result = result.sort_values("interaction_strength", ascending=False).reset_index(
        drop=True
    )
    return result.head(top_n)


def _get_predictions(
    model: Any, X: pd.DataFrame, use_predict_proba: bool
) -> np.ndarray:
    """Generate predictions using either predict_proba or predict.

    Args:
        model: A fitted XGBoost model (XGBClassifier or XGBRegressor).
        X: Feature DataFrame to predict on.
        use_predict_proba: If True, calls model.predict_proba(X) and returns
            the positive-class probability (column 1). Otherwise calls
            model.predict(X) for hard labels (classification) or raw
            predictions (regression).

    Returns:
        Array of predictions/probabilities of shape (n_samples,).
    """
    if use_predict_proba:
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X))


def compute_permutation_importance(
    model: Any,
    data: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    use_predict_proba: bool = False,
    n_repeats: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute permutation importance for each feature.

    For each feature, shuffles its values `n_repeats` times and measures
    the drop in model performance each time. A larger mean drop means the
    feature is more important. Averaging over multiple shuffles (rather
    than a single shuffle) reduces noise from any one unlucky/lucky
    permutation -- this matches the standard convention used by, e.g.,
    sklearn's own `permutation_importance`, which also defaults to 5
    repeats.

    Args:
        model: A fitted XGBoost model (XGBClassifier or XGBRegressor).
        data: DataFrame including both features and the target column.
        target_column: Name of the target column in data.
        feature_names: Ordered list of feature column names.
        metric_fn: Callable with signature (y_true, y_pred) -> float,
            where higher values indicate better performance.
        use_predict_proba: If True, predictions are generated via
            model.predict_proba(X)[:, 1] (positive-class probability)
            instead of model.predict(X). This is required for metrics
            such as roc_auc_score that expect scores/probabilities
            rather than hard class labels. Defaults to False.
        n_repeats: Number of independent shuffles per feature. The score
            drop is averaged across repeats. Each repeat uses its own
            deterministic seed (derived from the per-feature seed and the
            repeat index), so results remain fully reproducible for a
            given n_repeats. Defaults to 5.

    Returns:
        A tuple (importances, importances_std), each of shape
        (n_features,). `importances` is the mean permutation importance
        per feature across the n_repeats shuffles; positive values mean
        removing the feature hurts performance. `importances_std` is the
        standard deviation of the per-repeat score drops, a stability
        signal (higher means the importance estimate is noisier).
    """
    logger.info(f"Computing permutation importance (n_repeats={n_repeats})")
    X = data[feature_names].copy()
    y = np.asarray(data[target_column].values)

    baseline_preds = _get_predictions(model, X, use_predict_proba)
    baseline_score = metric_fn(y, baseline_preds)

    importances = np.zeros(len(feature_names))
    importances_std = np.zeros(len(feature_names))
    for i, feature in enumerate(feature_names):
        repeat_drops = np.zeros(n_repeats)
        for r in range(n_repeats):
            X_shuffled = X.copy()
            # A distinct but deterministic seed per feature and repeat
            # keeps results reproducible while still shuffling each
            # column differently across both features and repeats.
            rng = np.random.default_rng(seed=42 + i * n_repeats + r)
            X_shuffled[feature] = rng.permutation(
                np.asarray(X_shuffled[feature].values)
            )
            shuffled_preds = _get_predictions(model, X_shuffled, use_predict_proba)
            shuffled_score = metric_fn(y, shuffled_preds)
            repeat_drops[r] = baseline_score - shuffled_score
        importances[i] = repeat_drops.mean()
        importances_std[i] = repeat_drops.std()

    return importances, importances_std


def compute_mutual_information(
    data: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    task: str,
    random_seed: int,
) -> np.ndarray:
    """
    Compute mutual information between each feature and the target.

    Mutual information is a model-agnostic, information-theoretic measure
    of dependency between a feature and the target. Unlike SHAP and
    permutation importance, it does not depend on the trained model, so it
    can serve as an independent cross-check on those rankings and can catch
    non-linear or categorical dependencies a particular model might
    under-weight.

    Dispatches to sklearn's `mutual_info_classif` for binary/multiclass
    tasks or `mutual_info_regression` for regression tasks.

    Args:
        data: DataFrame including both features and the target column.
        target_column: Name of the target column in data.
        feature_names: Ordered list of feature column names.
        task: One of "binary", "multiclass", or "regression".
        random_seed: Seed for reproducible results (sklearn's MI estimators
            use a random noise term internally).

    Returns:
        Array of shape (n_features,) with mutual information score per
        feature. Values are non-negative; 0 means no detected dependency.
    """
    logger.info("Computing mutual information")
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    X = data[feature_names]
    y = data[target_column]

    if task in ("binary", "multiclass"):
        scores = mutual_info_classif(X, y, random_state=random_seed)
    else:
        scores = mutual_info_regression(X, y, random_state=random_seed)

    return np.asarray(scores)
