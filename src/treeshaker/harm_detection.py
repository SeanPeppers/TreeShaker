"""Feature harm detection based on importance signals."""

import numpy as np


class HarmDetector:
    """
    Detect features that are likely to harm model performance.

    A feature is considered harmful if:
    - Its mean absolute SHAP value is below the SHAP threshold, OR
    - Its permutation importance is below the permutation threshold.

    Both signals are checked independently; a feature failing either check
    is flagged as harmful.

    Args:
        shap_importance_threshold: Features with mean |SHAP| below this
            value are flagged. Defaults to 0.01.
        perm_importance_threshold: Features with permutation importance
            below this value are flagged. Defaults to 0.05.
    """

    def __init__(
        self,
        shap_importance_threshold: float = 0.01,
        perm_importance_threshold: float = 0.05,
    ) -> None:
        self.shap_threshold = shap_importance_threshold
        self.perm_threshold = perm_importance_threshold

    def flag_harmful_features(
        self,
        shap_importance: np.ndarray,
        perm_importance: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Determine which features are harmful based on importance scores.

        Args:
            shap_importance: Array of mean absolute SHAP values, one per
                feature.
            perm_importance: Array of permutation importance values, one per
                feature.

        Returns:
            A tuple of (is_harmful, harm_reasons) where:
            - is_harmful is a boolean array, True for harmful features.
            - harm_reasons is a list of strings describing why each feature
              was flagged (empty string if the feature is not harmful).
        """
        n_features = len(shap_importance)
        is_harmful = np.zeros(n_features, dtype=bool)
        harm_reasons: list[str] = []

        for i in range(n_features):
            low_shap = shap_importance[i] < self.shap_threshold
            low_perm = perm_importance[i] < self.perm_threshold

            if low_shap and low_perm:
                reason = (
                    "The model ignores this feature entirely — its SHAP "
                    f"importance is near zero ({shap_importance[i]:.3f}, "
                    f"below the {self.shap_threshold:.3f} threshold) and its "
                    f"permutation importance ({perm_importance[i]:.3f}) is "
                    f"below the {self.perm_threshold:.3f} threshold, meaning "
                    "shuffling it has no effect on predictions."
                )
                is_harmful[i] = True
                harm_reasons.append(reason)
            elif low_shap:
                reason = (
                    "The model assigns almost no weight to this feature "
                    f"(SHAP importance: {shap_importance[i]:.3f}, below the "
                    f"{self.shap_threshold:.3f} threshold). It contributes "
                    "negligible signal to predictions."
                )
                is_harmful[i] = True
                harm_reasons.append(reason)
            elif low_perm:
                reason = (
                    "Shuffling this feature barely changes model performance "
                    f"(permutation importance: {perm_importance[i]:.3f}, "
                    f"below the {self.perm_threshold:.3f} threshold), "
                    "meaning the model is not really using it."
                )
                is_harmful[i] = True
                harm_reasons.append(reason)
            else:
                harm_reasons.append("")

        return is_harmful, harm_reasons
