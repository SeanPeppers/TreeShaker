"""Data leakage and logical error detection for feature columns."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class LeakageConfig:
    """Configuration for leakage and logical error detection.

    Args:
        target_corr_threshold: Flag features whose absolute correlation
            with the target exceeds this value. High correlation may
            indicate the feature was derived from the target. Default 0.95.
        dominant_shap_fraction: Flag a feature if it accounts for more
            than this fraction of total SHAP importance. A single feature
            dominating this strongly is unusual and worth investigating.
            Default 0.7.
        near_constant_threshold: Flag features where the most common value
            appears in more than this fraction of rows. Near-constant
            features carry almost no information. Default 0.99.
        near_duplicate_corr_threshold: Flag a feature if its absolute
            correlation with any other feature exceeds this value.
            Near-duplicate features add redundancy without extra signal.
            Default 0.98.
        vif_threshold: Flag a feature if its Variance Inflation Factor
            (VIF) exceeds this value. A high VIF means the feature is
            well-predicted by a linear combination of the other features,
            indicating multicollinearity that pairwise correlation alone
            can miss. Default 10.0.
    """

    target_corr_threshold: float = 0.95
    dominant_shap_fraction: float = 0.7
    near_constant_threshold: float = 0.99
    near_duplicate_corr_threshold: float = 0.98
    vif_threshold: float = 10.0


# Column name patterns that suggest identifiers or future-leaking columns.
_SUSPICIOUS_NAME_PATTERNS = (
    "_id",
    "id_",
    "_key",
    "key_",
    "future_",
    "_future",
    "target",
    "label",
    "y_",
    "_y",
)

_EXACT_ID_NAMES = {"id", "index", "row_id", "record_id", "pk", "uuid"}


class LeakageDetector:
    """
    Detect features that may be causing data leakage or are logically suspect.

    Runs six independent checks against each feature:

    - **Target correlation**: A feature that is almost perfectly correlated
      with the target may have been derived from it, which would be leakage.
    - **SHAP dominance**: If one feature accounts for the vast majority of
      the model's attention, it may contain information that shouldn't be
      available at prediction time.
    - **Near-constant values**: A feature that is almost always the same
      value carries no signal and wastes model capacity.
    - **Near-duplicate features**: Features that are nearly identical to
      another feature add redundancy without extra predictive value.
    - **Multicollinearity (VIF)**: Features that can be predicted almost
      perfectly by a linear combination of the *other* features are
      redundant even when no single pairwise correlation looks extreme.
    - **Suspicious names**: Column names matching common patterns for
      identifiers or future-leaking columns are flagged as a soft warning.

    Each check is independent. A feature can trigger multiple warnings.

    Args:
        config: A LeakageConfig instance with detection thresholds.
    """

    def __init__(self, config: LeakageConfig | None = None) -> None:
        self.config = config or LeakageConfig()

    def detect(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_names: list[str],
        shap_importance: np.ndarray,
    ) -> dict[str, list[str]]:
        """
        Run all leakage checks and return plain-English warnings per feature.

        Args:
            df: The full dataset including the target column.
            target_column: Name of the target column.
            feature_names: Ordered list of feature column names.
            shap_importance: Array of mean absolute SHAP values, one per
                feature in the same order as feature_names.

        Returns:
            A dict mapping each feature name to a list of warning strings.
            Features with no issues map to an empty list.
        """
        warnings: dict[str, list[str]] = {name: [] for name in feature_names}

        self._check_target_correlation(df, target_column, feature_names, warnings)
        self._check_dominant_shap(feature_names, shap_importance, warnings)
        self._check_near_constant(df, feature_names, warnings)
        self._check_near_duplicates(df, feature_names, warnings)
        self._check_multicollinearity(df, feature_names, warnings)
        self._check_suspicious_names(feature_names, target_column, warnings)

        return warnings

    def _check_target_correlation(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_names: list[str],
        warnings: dict[str, list[str]],
    ) -> None:
        """Flag features with suspiciously high correlation to the target."""
        target = df[target_column]
        if not pd.api.types.is_numeric_dtype(target):
            return

        for name in feature_names:
            col = df[name]
            if not pd.api.types.is_numeric_dtype(col):
                continue
            corr = col.corr(target)
            if pd.isna(corr):
                continue
            if abs(corr) >= self.config.target_corr_threshold:
                warnings[name].append(
                    f"Suspiciously high correlation with the target"
                    f" (r={corr:.2f}, at or above the"
                    f" {self.config.target_corr_threshold:.2f} threshold)."
                    f" This feature may have been derived from or calculated"
                    f" using the target variable, which would be data"
                    f" leakage."
                )

    def _check_dominant_shap(
        self,
        feature_names: list[str],
        shap_importance: np.ndarray,
        warnings: dict[str, list[str]],
    ) -> None:
        """Flag a feature that accounts for most of the total SHAP importance."""
        total_shap = shap_importance.sum()
        if total_shap == 0:
            return

        for i, name in enumerate(feature_names):
            fraction = shap_importance[i] / total_shap
            if fraction >= self.config.dominant_shap_fraction:
                pct = int(fraction * 100)
                threshold_pct = int(self.config.dominant_shap_fraction * 100)
                warnings[name].append(
                    f"This feature accounts for {pct}% of total SHAP"
                    f" importance, at or above the {threshold_pct}%"
                    f" threshold. A single feature dominating this strongly"
                    f" is unusual — it may contain information that should"
                    f" not be available at prediction time."
                )

    def _check_near_constant(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        warnings: dict[str, list[str]],
    ) -> None:
        """Flag features where one value dominates almost all rows."""
        for name in feature_names:
            top_freq = df[name].value_counts(normalize=True).iloc[0]
            if top_freq >= self.config.near_constant_threshold:
                pct = int(top_freq * 100)
                threshold_pct = int(self.config.near_constant_threshold * 100)
                warnings[name].append(
                    f"Nearly constant: {pct}% of rows share the same value,"
                    f" at or above the {threshold_pct}% threshold. A feature"
                    f" with almost no variation cannot help the model"
                    f" distinguish between examples."
                )

    def _check_near_duplicates(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        warnings: dict[str, list[str]],
    ) -> None:
        """Flag features that are nearly identical to another feature."""
        numeric_features = [
            name for name in feature_names if pd.api.types.is_numeric_dtype(df[name])
        ]
        if len(numeric_features) < 2:
            return

        corr_matrix = df[numeric_features].corr().abs()

        for i, name in enumerate(numeric_features):
            for j, other in enumerate(numeric_features):
                if j <= i:
                    continue
                corr_val = corr_matrix.loc[name, other]
                if pd.isna(corr_val):
                    continue
                corr_val = float(corr_val)  # type: ignore[arg-type]
                if corr_val >= self.config.near_duplicate_corr_threshold:
                    warnings[name].append(
                        f"Nearly identical to '{other}' (correlation:"
                        f" {corr_val:.2f}, at or above the"
                        f" {self.config.near_duplicate_corr_threshold:.2f}"
                        f" threshold). Keeping both adds redundancy without"
                        f" extra signal — consider dropping one."
                    )

    def _check_multicollinearity(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        warnings: dict[str, list[str]],
    ) -> None:
        """Flag features with high Variance Inflation Factor (VIF).

        VIF for feature i is 1 / (1 - R^2), where R^2 comes from
        regressing feature i on all other numeric features. A high VIF
        means a feature is well-predicted by a linear combination of the
        others — multicollinearity that pairwise correlation alone can
        miss (e.g. three features that are each only moderately
        correlated pairwise but are jointly a near-linear combination of
        each other).
        """
        numeric_features = [
            name for name in feature_names if pd.api.types.is_numeric_dtype(df[name])
        ]
        if len(numeric_features) < 2:
            return

        numeric_df = df[numeric_features].dropna()
        if len(numeric_df) < 2:
            return

        for name in numeric_features:
            others = [other for other in numeric_features if other != name]
            y = numeric_df[name].to_numpy()
            x = numeric_df[others].to_numpy()

            if np.allclose(y, y[0]):
                # Constant column: R^2 is undefined; near-constant check
                # already covers this case, so skip here.
                continue

            model = LinearRegression()
            model.fit(x, y)
            r_squared = model.score(x, y)

            # Clamp to avoid floating point noise pushing slightly above 1
            # or below 0, and guard against division by zero on perfect
            # collinearity.
            r_squared = min(max(r_squared, 0.0), 1.0)
            if r_squared >= 1.0 - 1e-10:
                vif = float("inf")
            else:
                vif = 1.0 / (1.0 - r_squared)

            if vif >= self.config.vif_threshold:
                vif_str = "infinite" if np.isinf(vif) else f"{vif:.1f}"
                warnings[name].append(
                    f"High multicollinearity (VIF={vif_str}, at or above the"
                    f" {self.config.vif_threshold:.1f} threshold). This"
                    f" feature can be closely reconstructed from a"
                    f" combination of the other features, which is"
                    f" redundancy that pairwise correlation alone can miss —"
                    f" consider dropping it or one of the features it"
                    f" overlaps with."
                )

    def _check_suspicious_names(
        self,
        feature_names: list[str],
        target_column: str,
        warnings: dict[str, list[str]],
    ) -> None:
        """Flag column names that suggest identifiers or future-leaking data."""
        for name in feature_names:
            lower = name.lower().replace(" ", "_")

            if lower in _EXACT_ID_NAMES:
                warnings[name].append(
                    f"Column name '{name}' looks like a row identifier."
                    f" Identifiers have no predictive value and can cause"
                    f" the model to memorise training rows."
                )
                continue

            for pattern in _SUSPICIOUS_NAME_PATTERNS:
                if pattern in lower:
                    if pattern in ("target", "label", "y_", "_y"):
                        warnings[name].append(
                            f"Column name '{name}' contains '{pattern}',"
                            f" which often indicates it is derived from"
                            f" or related to the target variable."
                        )
                    elif pattern in ("future_", "_future"):
                        warnings[name].append(
                            f"Column name '{name}' contains '{pattern}',"
                            f" suggesting it may contain information from"
                            f" the future that would not be available at"
                            f" prediction time."
                        )
                    else:
                        warnings[name].append(
                            f"Column name '{name}' contains '{pattern}',"
                            f" which often indicates an identifier or key."
                            f" Identifiers can cause the model to memorise"
                            f" training rows rather than learn patterns."
                        )
                    break
