"""Calibrated confidence scores calculation for features."""

import numpy as np


def compute_confidence_scores(
    shap_importance: np.ndarray,
    perm_importance: np.ndarray,
    recommendations: list[str],
    shap_threshold: float,
    perm_threshold: float,
) -> np.ndarray:
    """Compute calibrated confidence scores (0.5 to 1.0) for each feature.

    Score = distance from nearest decision threshold, normalized via sigmoid
    so features clearly above/below threshold score near 1.0 and borderline
    cases score near 0.5.

    Args:
        shap_importance: Array of mean absolute SHAP values, one per feature.
        perm_importance: Array of permutation importance values, one per feature.
        recommendations: List of feature recommendations ('Keep', 'Caution', 'Drop').
        shap_threshold: Decision threshold for SHAP importance.
        perm_threshold: Decision threshold for permutation importance.

    Returns:
        Array of calibrated confidence scores in range [0.5, 1.0].
    """
    n_features = len(shap_importance)
    scores = np.zeros(n_features, dtype=float)

    for i in range(n_features):
        if recommendations[i] == "Caution":
            # Caution represents borderline or ambiguous cases, confidence is 0.5.
            scores[i] = 0.5
        else:
            # Normalized distance from threshold (relative to threshold;
            # guard against zero division).
            d_shap = (
                abs(shap_importance[i] - shap_threshold) / shap_threshold
                if shap_threshold > 0
                else abs(shap_importance[i])
            )
            d_perm = (
                abs(perm_importance[i] - perm_threshold) / perm_threshold
                if perm_threshold > 0
                else abs(perm_importance[i])
            )

            # Distance from the nearest decision threshold
            d = min(d_shap, d_perm)

            # Sigmoid normalization: 1 / (1 + exp(-d))
            # Since d >= 0, sigmoid(d) ranges from 0.5 (d=0) to 1.0 (d -> inf)
            scores[i] = 1.0 / (1.0 + np.exp(-d))

    return scores
