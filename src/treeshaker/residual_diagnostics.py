"""Residual-correlation diagnostics: a cross-check against SHAP/permutation/MI.

A feature can look unimportant to SHAP, permutation importance, and mutual
information yet still correlate with where the model's predictions are
systematically wrong (its residuals) -- e.g. a non-linear relationship the
tree model under-uses, or an interaction effect that none of the
single-feature importance signals capture. This module computes that
residual signal and flags features whose correlation with residuals exceeds
a configurable threshold, independent of the existing harm/leakage checks.

This is opt-in (off by default) and disconnected from the other detectors:
it does not change any other importance computation, it only adds an extra
signal that pipeline.py can use to annotate or downgrade a "Drop"
recommendation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import ignore_constant_corr_warnings


@dataclass
class ResidualConfig:
    """Configuration for residual-correlation diagnostics.

    Args:
        residual_corr_threshold: Flag a feature whose absolute Pearson
            correlation with the model's residuals exceeds this value.
            Default 0.3 -- high enough to avoid flagging routine noise,
            low enough to catch a real systematic relationship.
    """

    residual_corr_threshold: float = 0.3


def compute_residuals(
    model: Any,
    data: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    task: str,
) -> np.ndarray:
    """
    Compute per-sample prediction error ("residual") on held-out data.

    The definition of "residual" depends on the task, since classification
    has no single natural analogue of a signed regression residual:

    - regression: signed residual, `y_true - y_pred`. Preserves direction,
      so a feature correlated with residuals reveals which way the model
      is biased for that feature's range (e.g. under-predicting at high
      values), in addition to revealing heteroscedasticity.
    - binary: absolute calibration error in probability space,
      `|y_true - predict_proba(X)[:, 1]|`. There is no meaningful "sign" to
      a classification error, so we use magnitude: how far the predicted
      probability of the positive class was from the actual outcome.
    - multiclass: `1 - predict_proba(X)[true_class]`, i.e. one minus the
      probability mass the model assigned to the actually-observed class.
      This is the natural multiclass generalization of the binary
      calibration error above: 0 means the model was fully confident and
      correct, values near 1 mean the model assigned almost no probability
      to the true outcome.

    Args:
        model: A fitted XGBoost model (XGBClassifier or XGBRegressor).
        data: DataFrame including both features and the target column
            (typically the held-out test set).
        target_column: Name of the target column in data.
        feature_names: Ordered list of feature column names.
        task: One of "binary", "multiclass", or "regression".

    Returns:
        Array of shape (n_samples,) with one residual value per row of
        data, in the same row order as data.
    """
    X = data[feature_names]
    y_true = np.asarray(data[target_column].values)

    if task == "regression":
        y_pred = np.asarray(model.predict(X))
        return y_true - y_pred

    proba = np.asarray(model.predict_proba(X))
    if task == "binary":
        return np.abs(y_true - proba[:, 1])

    # multiclass: probability mass assigned to the true class, per row.
    y_true_int = y_true.astype(int)
    true_class_proba = proba[np.arange(len(y_true_int)), y_true_int]
    return 1.0 - true_class_proba


def compute_residual_correlation(
    data: pd.DataFrame,
    feature_names: list[str],
    residuals: np.ndarray,
) -> np.ndarray:
    """
    Correlate each feature's raw values against the residual array.

    Non-numeric features are skipped (matching the pattern in
    leakage_detection.py's `_check_target_correlation`, which also only
    correlates numeric columns) and reported as NaN.

    Args:
        data: DataFrame containing the feature columns (typically the
            held-out test set), in the same row order used to compute
            residuals.
        feature_names: Ordered list of feature column names.
        residuals: Array of shape (n_samples,) as returned by
            compute_residuals(), in the same row order as data.

    Returns:
        Array of shape (n_features,) with the Pearson correlation between
        each feature's values and residuals. NaN for non-numeric features
        or when the correlation is undefined (e.g. a constant column).
    """
    residual_series = pd.Series(residuals, index=data.index)
    correlations = np.full(len(feature_names), np.nan)

    for i, name in enumerate(feature_names):
        col = data[name]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        with ignore_constant_corr_warnings():
            corr = col.corr(residual_series)
        if pd.isna(corr):
            continue
        correlations[i] = corr

    return correlations


def flag_residual_correlated_features(
    feature_names: list[str],
    residual_correlation: np.ndarray,
    config: ResidualConfig | None = None,
) -> dict[str, str]:
    """
    Build a plain-English note for each feature whose |residual corr| is high.

    Args:
        feature_names: Ordered list of feature column names.
        residual_correlation: Array of shape (n_features,), as returned by
            compute_residual_correlation().
        config: A ResidualConfig instance with the flagging threshold.

    Returns:
        A dict mapping feature name to a plain-English note. Features below
        the threshold (or with NaN correlation) are omitted from the dict.
    """
    config = config or ResidualConfig()
    notes: dict[str, str] = {}

    for name, corr in zip(feature_names, residual_correlation, strict=True):
        if pd.isna(corr):
            continue
        if abs(corr) >= config.residual_corr_threshold:
            notes[name] = (
                "However, this feature correlates with model residuals"
                f" (r={corr:.2f}), suggesting the model may not be using it"
                " effectively — consider testing with a different model or"
                " feature transformation before dropping."
            )

    return notes
