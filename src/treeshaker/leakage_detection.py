"""Data leakage and logical error detection for feature columns."""

import logging
import re
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats.contingency import association
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# Attempt to import the optional Rust LSH extension for approximate screening.
try:
    import treeshaker_lsh

    _LSH_AVAILABLE = True
except ImportError:
    _LSH_AVAILABLE = False


@dataclass
class LeakageConfig:
    """Configuration for leakage and logical error detection.

    Args:
        target_corr_threshold: Flag features whose absolute correlation
            with the target exceeds this value. High correlation may
            indicate the feature was derived from the target. Default 0.95.
        categorical_assoc_threshold: Flag categorical (non-numeric) features
            whose bias-corrected Cramer's V association with the target
            exceeds this value, for binary/multiclass classification tasks.
            Cramer's V ranges from 0 (no association) to 1 (one variable
            fully determines the other), so a very high value can mean the
            feature is a near-perfect proxy for the target -- e.g. a
            post-outcome categorical field that was recorded after the
            target was known. This check only applies to classification
            tasks; for regression it is skipped (see `_check_target_correlation`
            for the equivalent numeric-feature check, which covers
            regression). Default 0.8.
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
        temporal_feature_patterns: Substrings (case-insensitive) that
            identify time-series engineered features such as lags,
            rolling/expanding windows, exponentially-weighted moving
            averages, shifts, and diffs. These features are *expected*
            to correlate highly with the target, so when a feature name
            matches one of these patterns it is exempted from the target
            correlation check (see `exempt_temporal_features`) and is
            instead checked for structural consistency (leading-NaN count
            matching the parsed lag/window offset). Default covers
            `lag_`, `rolling_`, `ewm_`, `shift_`, `diff_`, `expanding_`.
        exempt_temporal_features: When True (default), features whose name
            matches `temporal_feature_patterns` are exempted from the
            high-target-correlation leakage check. Set to False to
            disable the exemption and treat temporal features like any
            other column.
    """

    target_corr_threshold: float = 0.95
    categorical_assoc_threshold: float = 0.8
    dominant_shap_fraction: float = 0.7
    near_constant_threshold: float = 0.99
    near_duplicate_corr_threshold: float = 0.98
    vif_threshold: float = 10.0
    temporal_feature_patterns: tuple[str, ...] = field(
        default_factory=lambda: (
            "lag_",
            "rolling_",
            "ewm_",
            "shift_",
            "diff_",
            "expanding_",
        )
    )
    exempt_temporal_features: bool = True


# Column name patterns that suggest identifiers or future-leaking columns.
#
# Each entry encodes a whole *token* to match against the column name split
# on common separators (underscore, camelCase boundaries), not a raw
# substring -- see `_check_suspicious_names`. A trailing underscore means
# the token must be the *first* token (suffix-style patterns like "id_",
# "key_", "future_"); a leading underscore means it must be the *last*
# token (prefix-style patterns like "_id", "_key", "_future", "_y"); no
# underscore means it must match any whole token (e.g. "target", "label").
# This prevents false positives such as "timestamp_year" matching "_y"
# (the "y" inside "year" is not a standalone token), while still catching
# genuine cases like "target_y", "label_y", and "id_column".
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

# Matches camelCase boundaries (e.g. "userId" -> "user", "Id") so that
# camelCase column names are tokenized the same way as snake_case ones.
_CAMEL_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokenize_column_name(name: str) -> list[str]:
    """Split a column name into lowercase tokens on `_`, spaces, and
    camelCase boundaries, dropping empty tokens."""
    normalized = _CAMEL_BOUNDARY_PATTERN.sub("_", name).lower().replace(" ", "_")
    return [token for token in normalized.split("_") if token]


def _pattern_matches_tokens(pattern: str, tokens: list[str]) -> bool:
    """Check whether a suspicious-name pattern matches whole tokens.

    `pattern` is one of the entries in `_SUSPICIOUS_NAME_PATTERNS`. Its
    leading/trailing underscore encodes where the matching token must sit:

    - "foo_" (trailing underscore only): the *first* token must equal "foo".
    - "_foo" (leading underscore only): the *last* token must equal "foo".
    - "foo" (no underscore): *any* token must equal "foo".
    """
    if not tokens:
        return False
    bare = pattern.strip("_")
    if pattern.startswith("_") and not pattern.endswith("_"):
        return tokens[-1] == bare
    if pattern.endswith("_") and not pattern.startswith("_"):
        return tokens[0] == bare
    return bare in tokens


# Matches a trailing integer offset after a temporal pattern, e.g.
# "lag_7", "rolling_mean_30" (offset is the *first* trailing integer
# group, e.g. "30"), "shift_1". Used to estimate the expected number of
# leading NaNs for structural validation.
_TRAILING_INT_PATTERN = re.compile(r"(\d+)")


class LeakageDetector:
    """
    Detect features that may be causing data leakage or are logically suspect.

    Runs seven independent checks against each feature:

    - **Target correlation**: A feature that is almost perfectly correlated
      with the target may have been derived from it, which would be leakage.
    - **Categorical target association**: For classification tasks, a
      categorical feature whose bias-corrected Cramer's V association with
      the target is very high may be a near-perfect proxy for the target
      (e.g. a post-outcome field), which would be leakage. This is the
      categorical analogue of the target correlation check above, since
      Pearson correlation only applies to numeric columns.
    - **SHAP dominance**: If one feature accounts for the vast majority of
      the model's attention, it may contain information that shouldn't be
      available at prediction time.
    - **Near-constant values**: A feature that is almost always the same
      value carries no signal and wastes model capacity.
    - **Near-duplicate features**: Features that are nearly identical to
      another *numeric* feature add redundancy without extra predictive
      value. This check (and the `screening_mode`/`leakage_screening`
      exact-vs-approximate choice below) only ever looks at numeric
      columns -- it has no bearing on categorical proxy-leakage, which is
      caught by the categorical target association check above instead.
    - **Multicollinearity (VIF)**: Features that can be predicted almost
      perfectly by a linear combination of the *other* features are
      redundant even when no single pairwise correlation looks extreme.
    - **Suspicious names**: Column names matching common patterns for
      identifiers or future-leaking columns are flagged as a soft warning.

    Each check is independent. A feature can trigger multiple warnings.

    Features whose name matches a temporal pattern (`lag_`, `rolling_`,
    `ewm_`, `shift_`, `diff_`, `expanding_` by default — see
    `LeakageConfig.temporal_feature_patterns`) are exempted from the
    target correlation check, since lag/rolling/etc. features are
    *expected* to correlate highly with the target. Use
    `detect_temporal_structure` to instead check those features for
    structural consistency (leading-NaN count matching the parsed
    offset, no unexpected mid-series NaN gaps). This is a separate
    warning category from leakage, surfaced independently by callers.

    Args:
        config: A LeakageConfig instance with detection thresholds.
        screening_mode: "exact" or "approximate" -- selects the algorithm
            used for the near-duplicate *numeric* feature check only (see
            `_check_near_duplicates`). It has no effect on any other check,
            including the always-on categorical target association
            (Cramer's V) check that is responsible for catching categorical
            proxy-leakage columns.
    """

    def __init__(
        self, config: LeakageConfig | None = None, screening_mode: str = "exact"
    ) -> None:
        self.config = config or LeakageConfig()
        self.screening_mode = screening_mode

    def detect(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_names: list[str],
        shap_importance: np.ndarray,
        task: str = "binary",
    ) -> dict[str, list[str]]:
        """
        Run all leakage checks and return plain-English warnings per feature.

        Args:
            df: The full dataset including the target column.
            target_column: Name of the target column.
            feature_names: Ordered list of feature column names.
            shap_importance: Array of mean absolute SHAP values, one per
                feature in the same order as feature_names.
            task: One of "binary", "multiclass", or "regression". Used by
                the categorical-target-association check, which only
                applies to classification tasks. Default "binary".

        Returns:
            A dict mapping each feature name to a list of warning strings.
            Features with no issues map to an empty list.

        Note:
            Features whose name matches a temporal pattern (see
            `LeakageConfig.temporal_feature_patterns`) are exempt from the
            target-correlation and categorical-target-association checks
            by default, since lags/rolling windows/etc. are *expected* to
            correlate highly with the target. Use `detect_temporal_structure`
            separately to check those features for structural consistency
            instead.
        """
        warnings: dict[str, list[str]] = {name: [] for name in feature_names}

        self._check_target_correlation(df, target_column, feature_names, warnings)
        self._check_categorical_target_association(
            df, target_column, feature_names, task, warnings
        )
        self._check_dominant_shap(feature_names, shap_importance, warnings)
        self._check_near_constant(df, feature_names, warnings)
        self._check_near_duplicates(df, feature_names, warnings)
        self._check_multicollinearity(df, feature_names, warnings)
        self._check_suspicious_names(feature_names, target_column, warnings)

        return warnings

    def detect_temporal_structure(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
    ) -> dict[str, list[str]]:
        """
        Check time-series engineered features for structural consistency.

        Features whose name matches a temporal pattern (lag/rolling/ewm/
        shift/diff/expanding) are exempt from the high-target-correlation
        leakage check, since that correlation is expected. Instead, this
        checks that the leading-NaN count matches the lag/window offset
        parsed from the feature name, and that there are no unexpected
        NaN gaps mid-series — both of which can indicate an off-by-one or
        windowing bug that leaks future information.

        These are returned as a separate category from `detect`'s leakage
        warnings: a temporal structure issue is a correctness bug in how
        the feature was engineered, not evidence the feature was derived
        from the target.

        Args:
            df: The full dataset (does not need the target column).
            feature_names: Ordered list of feature column names.

        Returns:
            A dict mapping each feature name to a list of structural
            warning strings. Features with no issues, or that are not
            temporal features, map to an empty list. Empty if
            `LeakageConfig.exempt_temporal_features` is False.
        """
        warnings: dict[str, list[str]] = {name: [] for name in feature_names}
        if not self.config.exempt_temporal_features:
            return warnings

        for name in feature_names:
            if not self._is_temporal_feature(name):
                continue
            col = df[name]
            offset = self._parse_temporal_offset(name)

            is_na = col.isna().to_numpy()
            n = len(is_na)
            leading_nan_count = int(np.argmax(~is_na)) if not is_na.all() else n

            if offset is not None and leading_nan_count != offset:
                warnings[name].append(
                    f"Temporal structure warning: '{name}' has"
                    f" {leading_nan_count} leading NaN(s), but its name"
                    f" implies an offset/window of {offset}. A mismatch"
                    f" can indicate an off-by-one or windowing bug that"
                    f" leaks future information into earlier rows."
                )

            if leading_nan_count < n:
                trailing = is_na[leading_nan_count:]
                if trailing.any():
                    gap_count = int(trailing.sum())
                    warnings[name].append(
                        f"Temporal structure warning: '{name}' has"
                        f" {gap_count} unexpected NaN(s) after the initial"
                        f" leading-NaN run, which is unusual for a"
                        f" lag/rolling/window feature and worth"
                        f" investigating."
                    )

        return warnings

    def _is_temporal_feature(self, name: str) -> bool:
        """Check whether a feature name matches a configured temporal pattern."""
        lower = name.lower()
        return any(
            pattern.lower() in lower
            for pattern in self.config.temporal_feature_patterns
        )

    @staticmethod
    def _parse_temporal_offset(name: str) -> int | None:
        """Parse the lag/window offset from a temporal feature name.

        Looks for the last integer group in the name (e.g. "rolling_mean_30"
        -> 30, "lag_7" -> 7). For decimal parameters like "ewm_alpha_0.3"
        the fractional part is not an offset, so no integer count can be
        meaningfully derived and None is returned for that case via the
        decimal point check.
        """
        if "." in name:
            # e.g. ewm_alpha_0.3 -- not a leading-NaN-count style parameter.
            return None
        matches = _TRAILING_INT_PATTERN.findall(name)
        if not matches:
            return None
        return int(matches[-1])

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
            if self.config.exempt_temporal_features and self._is_temporal_feature(name):
                continue
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

    def _check_categorical_target_association(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_names: list[str],
        task: str,
        warnings: dict[str, list[str]],
    ) -> None:
        """Flag categorical features with suspiciously high association
        with a classification target.

        `_check_target_correlation` only looks at numeric features (Pearson
        correlation is undefined for categoricals), so a categorical
        feature that is effectively a proxy for the target -- e.g. a
        post-outcome field recorded after the target was already known --
        could otherwise sail through with zero leakage warnings even when
        it dominates SHAP importance. This check fills that gap using
        bias-corrected Cramer's V (Bergsma 2013), a standard measure of
        association between two categorical variables that ranges from 0
        (no association) to 1 (one variable fully determines the other).

        Only applies to classification tasks ("binary"/"multiclass"). For
        regression, the target is continuous and a contingency-table-based
        association measure does not apply cleanly, so this check is
        skipped entirely -- the numeric `_check_target_correlation` check
        already covers numeric features for regression, and a categorical
        feature in a regression problem would need an ANOVA/eta-squared
        style check, which is out of scope here.

        Missing values are treated as their own category rather than
        dropped. This matters in practice: for a column like the Titanic
        dataset's `boat` (lifeboat assignment, recorded only for
        survivors), *whether* the value is missing is itself the leakage
        signal -- dropping NaN rows would throw that signal away and
        could mask a real proxy-for-target column.
        """
        if task not in ("binary", "multiclass"):
            return

        target = df[target_column]

        for name in feature_names:
            col = df[name]
            if pd.api.types.is_numeric_dtype(col):
                continue
            if self.config.exempt_temporal_features and self._is_temporal_feature(name):
                continue

            target_valid = target.notna()
            if target_valid.sum() < 2:
                continue
            col_filled = (
                col[target_valid]
                .astype(object)
                .where(col[target_valid].notna(), "__MISSING__")
            )

            contingency = pd.crosstab(col_filled, target[target_valid])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            try:
                cramers_v = association(
                    contingency.to_numpy(), method="cramer", correction=True
                )
            except (ValueError, ZeroDivisionError):
                continue

            if pd.isna(cramers_v):
                continue

            if cramers_v >= self.config.categorical_assoc_threshold:
                warnings[name].append(
                    f"Suspiciously high association with the target"
                    f" (Cramer's V={cramers_v:.2f}, at or above the"
                    f" {self.config.categorical_assoc_threshold:.2f} threshold)."
                    f" This categorical feature may have been recorded"
                    f" after the outcome was known or otherwise derived"
                    f" from the target, which would be data leakage."
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
        """Flag numeric features that are nearly identical to another numeric
        feature.

        Uses either exact pairwise correlation (O(p^2)) or approximate
        LSH-based screening (near-linear) depending on screening_mode. Only
        ever considers numeric columns (see the `numeric_features` filter
        below) -- categorical proxy-leakage is caught separately by
        `_check_categorical_target_association` (Cramer's V), which always
        runs regardless of screening_mode.
        """
        numeric_features = [
            name for name in feature_names if pd.api.types.is_numeric_dtype(df[name])
        ]
        if len(numeric_features) < 2:
            return

        if self.screening_mode == "approximate":
            self._check_near_duplicates_approximate(df, numeric_features, warnings)
        else:
            self._check_near_duplicates_exact(df, numeric_features, warnings)

    def _check_near_duplicates_exact(
        self,
        df: pd.DataFrame,
        numeric_features: list[str],
        warnings: dict[str, list[str]],
    ) -> None:
        """Exact pairwise correlation screening (O(p^2))."""
        corr_matrix = df[numeric_features].corr().abs()

        for i, name in enumerate(numeric_features):
            for j, other in enumerate(numeric_features):
                if j <= i:
                    continue
                corr_val = corr_matrix.loc[name, other]
                if pd.isna(corr_val):
                    continue
                corr_val = float(cast(float, corr_val))
                if corr_val >= self.config.near_duplicate_corr_threshold:
                    warnings[name].append(
                        f"Nearly identical to '{other}' (correlation:"
                        f" {corr_val:.2f}, at or above the"
                        f" {self.config.near_duplicate_corr_threshold:.2f}"
                        f" threshold). Keeping both adds redundancy without"
                        f" extra signal — consider dropping one."
                    )

    def _check_near_duplicates_approximate(
        self,
        df: pd.DataFrame,
        numeric_features: list[str],
        warnings: dict[str, list[str]],
    ) -> None:
        """Approximate LSH-based near-duplicate screening with fallback."""
        if not _LSH_AVAILABLE:
            logger.warning(
                "leakage_screening='approximate' requested but Rust LSH"
                " extension unavailable. Install with"
                " 'pip install treeshaker[fast]' or falling back to exact"
                " screening. This may be slow on wide datasets."
            )
            self._check_near_duplicates_exact(df, numeric_features, warnings)
            return

        try:
            # Prepare feature matrix as (n_features, n_rows) for LSH
            feature_data = df[numeric_features].to_numpy().T  # Transpose to (p, n)
            feature_data_f64 = feature_data.astype(np.float64)

            # Call Rust LSH function
            candidates = treeshaker_lsh.simhash_screen(
                feature_data_f64, num_hashes=24, seed=42
            )

            # Convert candidate indices to feature names and check thresholds
            for i, j, corr_val in candidates:
                if corr_val >= self.config.near_duplicate_corr_threshold:
                    name_i = numeric_features[i]
                    name_j = numeric_features[j]
                    warnings[name_i].append(
                        f"Nearly identical to '{name_j}' (correlation:"
                        f" {corr_val:.2f}, at or above the"
                        f" {self.config.near_duplicate_corr_threshold:.2f}"
                        f" threshold). Keeping both adds redundancy without"
                        f" extra signal — consider dropping one."
                    )
        except Exception as e:
            logger.warning(
                f"Error during approximate LSH screening, falling back to exact: {e}"
            )
            self._check_near_duplicates_exact(df, numeric_features, warnings)

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
            tokens = _tokenize_column_name(name)

            if lower in _EXACT_ID_NAMES:
                warnings[name].append(
                    f"Column name '{name}' looks like a row identifier."
                    f" Identifiers have no predictive value and can cause"
                    f" the model to memorise training rows."
                )
                continue

            for pattern in _SUSPICIOUS_NAME_PATTERNS:
                if _pattern_matches_tokens(pattern, tokens):
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

    def detect_lookahead_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        time_column: str,
        feature_names: list[str],
    ) -> dict[str, list[str]]:
        """Check for look-ahead leakage in chronological splits.

        When data is split chronologically by time_column, look-ahead leakage
        occurs when a feature in train or val contains information from the
        future (from test rows). This method runs two complementary checks:

        1. Flags features whose distribution shifts sharply and
           systematically across the train/val/test split (measured via
           eta-squared, the fraction of the feature's variance explained
           by split membership). This catches features derived directly
           from the time column used to create the split -- e.g. a "year"
           or "month" column extracted from a timestamp -- since such
           features take on systematically different values in each split
           by construction. A plain Pearson correlation with the time
           column would miss cyclical fields like "month" or "day"; using
           split-membership variance instead detects any feature (linear
           or cyclical) whose value distribution differs meaningfully
           between train, val, and test.
        2. Flags features whose mean shows a large jump from train to val
           but then unexpectedly stabilizes from val to test, which would
           suggest val was already "aware" of test (future information
           leaked into val).

        Args:
            train_df: Training set DataFrame.
            val_df: Validation set DataFrame.
            test_df: Test set DataFrame.
            time_column: Name of the column used for chronological sorting.
            feature_names: List of feature column names to check.

        Returns:
            A dict mapping each feature name to a list of warning strings.
            Features with no look-ahead leakage signals map to an empty list.
        """
        warnings: dict[str, list[str]] = {name: [] for name in feature_names}

        # Concatenate splits with explicit markers to track boundaries
        train_df_marked = train_df.copy()
        train_df_marked["_split"] = "train"
        val_df_marked = val_df.copy()
        val_df_marked["_split"] = "val"
        test_df_marked = test_df.copy()
        test_df_marked["_split"] = "test"

        combined = pd.concat(
            [train_df_marked, val_df_marked, test_df_marked],
            ignore_index=True,
        )

        # Get temporal boundaries
        if time_column not in combined.columns:
            return warnings  # Should not happen if data was split chronologically

        try:
            val_time_range = (val_df[time_column].min(), val_df[time_column].max())
            test_time_range = (
                test_df[time_column].min(),
                test_df[time_column].max(),
            )
        except TypeError:
            # time_column is not numeric (datetime or other non-comparable type)
            return warnings

        # Check for temporal overlap that would indicate look-ahead leakage
        # (train or val should not have values later than test start).
        # In a proper chronological split, this shouldn't happen, but we
        # still need to confirm it before trusting the time column as a
        # genuine split-boundary signal below.
        val_start, val_end = val_time_range
        test_start, test_end = test_time_range
        is_chronological_split = (
            pd.notna(val_start)
            and pd.notna(val_end)
            and pd.notna(test_start)
            and pd.notna(test_end)
            and val_end <= test_start
        )

        # For each numeric feature, check if there's a sharp change at split
        # boundaries that might indicate look-ahead
        for name in feature_names:
            if name not in combined.columns:
                continue
            col = combined[name]
            if not pd.api.types.is_numeric_dtype(col):
                continue

            # A feature whose value distribution differs systematically
            # between train, val, and test -- measured by eta-squared, the
            # fraction of total variance explained by split membership --
            # is, by construction, encoding which side of the split
            # boundary a row falls on. This is the textbook look-ahead
            # leakage case for features derived directly from the time
            # column (e.g. "year"/"month" extracted from a timestamp):
            # that boundary information is not legitimately available at
            # prediction time. Using eta-squared (rather than a plain
            # correlation with the time column) also catches cyclical
            # time-derived fields whose raw values are not monotonic in
            # time but still differ in distribution across splits.
            if name != time_column and is_chronological_split:
                try:
                    train_vals = train_df[name].dropna()
                    val_vals = val_df[name].dropna()
                    test_vals = test_df[name].dropna()
                    groups = [train_vals, val_vals, test_vals]
                    if all(len(g) > 0 for g in groups) and any(
                        g.nunique() > 1 for g in groups
                    ):
                        all_vals = pd.concat(groups)
                        grand_mean = all_vals.mean()
                        ss_total = float(((all_vals - grand_mean) ** 2).sum())
                        if ss_total > 0:
                            ss_between = float(
                                sum(
                                    len(g) * (g.mean() - grand_mean) ** 2
                                    for g in groups
                                )
                            )
                            eta_squared = ss_between / ss_total
                            if eta_squared >= 0.35:
                                warnings[name].append(
                                    f"Look-ahead leakage warning: '{name}'"
                                    f" distribution differs sharply across"
                                    f" the train/val/test split (eta-squared"
                                    f" {eta_squared:.2f} of its variance is"
                                    f" explained by split membership)."
                                    f" Features derived directly from the"
                                    f" time column used to create the"
                                    f" chronological split (e.g. year or"
                                    f" month extracted from a timestamp)"
                                    f" encode the split boundary itself,"
                                    f" which is not legitimately available"
                                    f" information at prediction time."
                                )
                except (TypeError, ValueError):
                    pass

            # Check if feature values in later splits are unusually predictable
            # from earlier split trends (suggesting future information leakage)
            try:
                train_mean = train_df[name].mean()
                val_mean = val_df[name].mean()
                test_mean = test_df[name].mean()

                # If test mean is closer to a specific value and val mean is
                # very different, it might indicate look-ahead leakage
                # (the model saw test data when building val features)
                if pd.notna(train_mean) and pd.notna(val_mean) and pd.notna(test_mean):
                    # Check for anomalous jumps
                    train_to_val_change = abs(val_mean - train_mean)
                    val_to_test_change = abs(test_mean - val_mean)

                    # If val->test change is much smaller than train->val,
                    # it might suggest val was already "aware" of test
                    if train_mean != 0:
                        train_to_val_pct = abs((val_mean - train_mean) / train_mean)
                    else:
                        train_to_val_pct = train_to_val_change

                    if val_mean != 0:
                        val_to_test_pct = abs((test_mean - val_mean) / val_mean)
                    else:
                        val_to_test_pct = val_to_test_change

                    # Flag if the transition smooths out unexpectedly at the
                    # split boundary (suggests information leakage).
                    # High threshold to avoid false positives on clean random data.
                    if (
                        train_to_val_pct > 1.0
                        and val_to_test_pct < train_to_val_pct * 0.05
                    ):
                        warnings[name].append(
                            f"Look-ahead leakage warning: '{name}' shows"
                            f" {train_to_val_pct * 100:.1f}% change from train"
                            f" to validation but only {val_to_test_pct * 100:.1f}%"
                            f" change from validation to test. Features that"
                            f" 'stabilize' at split boundaries may contain"
                            f" information from the future (test set)."
                        )
            except (TypeError, ValueError):
                # Feature is not numeric or has comparison issues
                continue

        return warnings
