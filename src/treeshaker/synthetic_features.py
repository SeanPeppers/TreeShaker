"""Synthetic feature suggestion and injection from SHAP interactions.

A pair of features with a strong SHAP interaction sometimes hints at a useful
derived feature (e.g. a product or ratio term) that the model would benefit
from having explicitly. This module takes the interaction matrix already
computed by `compute_shap_interactions()` (importance.py) and, for the top-N
strongest pairs, builds candidate derived features and scores them by mutual
information against the target.

Operators supported:
  - product  (A * B)        -- captures multiplicative interactions
  - ratio    (A / B)        -- captures rate/proportion relationships; skipped
                               when B contains zeros to avoid inf/nan

Categorical interactions are also supported: when one feature in a pair is
categorical (pandas ``category`` dtype or ``object``), a target-mean encoding
is computed per category level and the resulting numeric column is used in
place of the raw category codes for the interaction.

Loop feedback (``dfe_loop_feedback=True`` in SyntheticConfig): after scoring,
any candidate whose MI score exceeds ``mi_threshold`` is injected back into
the pipeline's feature set so it competes against the raw features in the
full selection run. Raw features that are dominated by their synthetic
replacement get ``recommendation='ReplacedBySynthetic'``.

Advisory mode (default, ``dfe_loop_feedback=False``): candidates are reported
for a human to evaluate, never auto-injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .importance import compute_mutual_information


@dataclass
class SyntheticConfig:
    """Configuration for SHAP-interaction-driven feature suggestion.

    Args:
        synthesize_from_interactions: If True, after SHAP interactions are
            computed, build candidate derived features from the strongest
            interacting pairs and score them by mutual information. Off by
            default. Requires config.compute_shap_interactions to also be True.
        interaction_top_n: Number of strongest interacting pairs to consider.
        mi_threshold: Minimum mutual information a candidate must exceed to be
            included in suggestions (or injected when loop feedback is on).
        operators: List of operators to try per pair. Supported: "product",
            "ratio". Default: ["product", "ratio"].
        dfe_loop_feedback: If True, inject candidates that clear mi_threshold
            back into the pipeline feature set so they compete against raw
            features. Winning synthetic features may cause their raw
            constituents to be marked 'ReplacedBySynthetic'. Off by default.
        dfe_max_inject: Maximum number of synthetic features to inject when
            dfe_loop_feedback is True. The top-scoring candidates by MI are
            selected. Default 5.
    """

    synthesize_from_interactions: bool = False
    interaction_top_n: int = 5
    mi_threshold: float = 0.01
    operators: list[str] = field(default_factory=lambda: ["product", "ratio"])
    dfe_loop_feedback: bool = False
    dfe_max_inject: int = 5


@dataclass
class SyntheticFeatureSuggestion:
    """A single candidate synthetic feature derived from a SHAP interaction.

    Attributes:
        feature_a: First raw feature name in the pair.
        feature_b: Second raw feature name in the pair.
        operator: Operator used ("product" or "ratio").
        suggested_name: Proposed column name for the candidate feature.
        mi_score: Mutual information between the candidate and the target.
    """

    feature_a: str
    feature_b: str
    operator: str
    suggested_name: str
    mi_score: float


def _encode_for_interaction(series: pd.Series, target: pd.Series) -> pd.Series | None:
    """Return a numeric series suitable for interaction feature construction.

    For numeric dtypes: returns the series as-is.
    For categorical/object dtypes: returns target-mean encoding per category
    (the mean of ``target`` within each category level). Returns None if
    encoding produces all-NaN (e.g. unseen categories with no target signal).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
        means = target.groupby(series, observed=True).mean()
        encoded = series.map(means)
        if encoded.isna().all():
            return None
        return encoded.astype(float)
    return None


def _build_candidate(
    col_a: pd.Series,
    col_b: pd.Series,
    operator: str,
) -> pd.Series | None:
    """Build a candidate derived feature from two numeric series.

    Returns None if the operation would produce all-inf/nan (e.g. ratio with
    zero denominator throughout).
    """
    if operator == "product":
        return col_a * col_b
    if operator == "ratio":
        # Skip if denominator contains any zeros — inf values corrupt MI calc.
        if (col_b == 0).any():
            return None
        result = col_a / col_b
        if not np.isfinite(result.to_numpy()).all():
            return None
        return result
    return None


def suggest_synthetic_features(
    data: pd.DataFrame,
    target_column: str,
    interactions: pd.DataFrame,
    task: str,
    random_seed: int,
    config: SyntheticConfig | None = None,
) -> list[SyntheticFeatureSuggestion]:
    """Suggest candidate derived features from strong SHAP interactions.

    Takes the top ``config.interaction_top_n`` rows of ``interactions``,
    constructs candidate features for each pair × operator combination,
    computes mutual information against the target, and keeps candidates
    whose MI score exceeds ``config.mi_threshold``.

    Args:
        data: DataFrame including both features and the target column.
        target_column: Name of the target column in data.
        interactions: DataFrame with columns feature_a, feature_b,
            interaction_strength, sorted descending by interaction_strength.
        task: One of "binary", "multiclass", or "regression".
        random_seed: Seed for reproducible mutual information results.
        config: A SyntheticConfig instance.

    Returns:
        List of SyntheticFeatureSuggestion sorted by mi_score descending.
    """
    config = config or SyntheticConfig()
    top_pairs = interactions.head(config.interaction_top_n)
    target = data[target_column]

    suggestions: list[SyntheticFeatureSuggestion] = []
    seen_names: set[str] = set()

    for _, row in top_pairs.iterrows():
        feature_a: str = row["feature_a"]
        feature_b: str = row["feature_b"]

        col_a = _encode_for_interaction(data[feature_a], target)
        col_b = _encode_for_interaction(data[feature_b], target)
        if col_a is None or col_b is None:
            continue

        for operator in config.operators:
            op_suffix = "x" if operator == "product" else "div"
            suggested_name = f"{feature_a}__{op_suffix}__{feature_b}"
            if suggested_name in seen_names:
                continue

            candidate_col = _build_candidate(col_a, col_b, operator)
            if candidate_col is None:
                continue

            candidate = data[[target_column]].copy()
            candidate[suggested_name] = candidate_col

            mi_score = compute_mutual_information(
                data=candidate,
                target_column=target_column,
                feature_names=[suggested_name],
                task=task,
                random_seed=random_seed,
            )[0]

            if mi_score > config.mi_threshold:
                seen_names.add(suggested_name)
                suggestions.append(
                    SyntheticFeatureSuggestion(
                        feature_a=feature_a,
                        feature_b=feature_b,
                        operator=operator,
                        suggested_name=suggested_name,
                        mi_score=float(mi_score),
                    )
                )

    suggestions.sort(key=lambda s: s.mi_score, reverse=True)
    return suggestions


def build_injection_frame(
    data: pd.DataFrame,
    target_column: str,
    suggestions: list[SyntheticFeatureSuggestion],
    max_inject: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the augmented DataFrame for DFE loop feedback.

    Takes the top ``max_inject`` suggestions by MI score, adds their
    candidate columns to ``data``, and returns the augmented frame plus the
    list of injected column names.

    Args:
        data: Full dataset (features + target).
        target_column: Name of the target column.
        suggestions: Sorted list of SyntheticFeatureSuggestion (best first).
        max_inject: Maximum columns to inject.

    Returns:
        Tuple of (augmented_df, injected_names).
    """
    target = data[target_column]
    augmented = data.copy()
    injected: list[str] = []

    for suggestion in suggestions[:max_inject]:
        col_a = _encode_for_interaction(data[suggestion.feature_a], target)
        col_b = _encode_for_interaction(data[suggestion.feature_b], target)
        if col_a is None or col_b is None:
            continue
        candidate = _build_candidate(col_a, col_b, suggestion.operator)
        if candidate is None:
            continue
        augmented[suggestion.suggested_name] = candidate
        injected.append(suggestion.suggested_name)

    return augmented, injected
