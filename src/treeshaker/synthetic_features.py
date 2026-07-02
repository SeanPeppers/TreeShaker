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


def _encode_for_interaction(
    series: pd.Series,
    target: pd.Series,
    means: pd.Series | None = None,
) -> pd.Series | None:
    """Return a numeric series suitable for interaction feature construction.

    For numeric dtypes: returns the series as-is.
    For categorical/object dtypes: returns target-mean encoding per category
    (the mean of ``target`` within each category level). When ``means`` is
    provided it is used as the fixed encoding map (fit elsewhere, e.g. on the
    training split) instead of recomputing from ``target`` — this lets callers
    fit the encoding on one split and apply it to another without target
    leakage. Returns None if encoding produces all-NaN (e.g. unseen categories
    with no target signal).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
        externally_fit = means is not None
        if means is None:
            means = target.groupby(series, observed=True).mean()
        encoded = series.map(means)
        if encoded.isna().all():
            return None
        if externally_fit and encoded.isna().any():
            # Categories present here but unseen in the fit split (e.g. a test
            # level absent from train) map to NaN. Fall back to the mean of the
            # fit-split category means rather than leaking this split's target.
            encoded = encoded.fillna(means.mean())
        return encoded.astype(float)
    return None


def fit_interaction_encoders(
    data: pd.DataFrame,
    target_column: str,
    suggestions: list[SyntheticFeatureSuggestion],
) -> dict[str, pd.Series]:
    """Fit target-mean encodings for the categorical features used by suggestions.

    Computes, for every categorical/object constituent feature referenced by
    ``suggestions``, the per-category mean of the target on ``data``. Intended
    to be fit on the training split and passed to :func:`build_injection_frame`
    so the same encoding is applied to a held-out split — otherwise a
    synthetic built from a categorical feature would encode that split's own
    target and win the DFE competition through leakage rather than merit.
    """
    referenced = {
        feature
        for suggestion in suggestions
        for feature in (suggestion.feature_a, suggestion.feature_b)
    }
    all_means = _fit_feature_means(data, target_column)
    return {name: means for name, means in all_means.items() if name in referenced}


def _fit_feature_means(
    source: pd.DataFrame | None, target_column: str
) -> dict[str, pd.Series]:
    """Fit per-category target means for every categorical/object column.

    Returns an empty dict when ``source`` is None. Used both by
    :func:`fit_interaction_encoders` (filtered to the features a set of
    suggestions references) and by :func:`suggest_synthetic_features` (to
    encode categoricals from a separate fit split without target leakage).
    """
    if source is None:
        return {}
    target = source[target_column]
    means: dict[str, pd.Series] = {}
    for feature in source.columns:
        if feature == target_column:
            continue
        series = source[feature]
        if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
            means[feature] = target.groupby(series, observed=True).mean()
    return means


#: A ratio denominator is treated as near-singular when its smallest magnitude
#: is below this fraction of the denominator's robust scale (median absolute
#: value). Such a value doesn't produce inf, but does blow the ratio up to an
#: extreme finite magnitude that would spuriously dominate MI/SHAP.
_RATIO_NEAR_SINGULAR_FRACTION = 1e-6


def _build_candidate(
    col_a: pd.Series,
    col_b: pd.Series,
    operator: str,
) -> pd.Series | None:
    """Build a candidate derived feature from two numeric series.

    Returns None if the operation would produce a numerically unstable feature:
    a ratio whose denominator contains an exact zero (inf), a near-singular
    denominator (a value negligibly small relative to the denominator's own
    scale, which yields an extreme finite blow-up), or any non-finite result.
    """
    if operator == "product":
        return col_a * col_b
    if operator == "ratio":
        denom_abs = col_b.abs()
        # Skip if denominator contains any zeros — inf values corrupt MI calc.
        if (denom_abs == 0).any():
            return None
        # Skip near-singular denominators: a value that is tiny relative to the
        # denominator's robust scale (median |B|) produces an extreme, unstable
        # ratio that would spuriously dominate the mutual-information score.
        denom_scale = float(denom_abs.median())
        if denom_scale > 0 and bool(
            (denom_abs < _RATIO_NEAR_SINGULAR_FRACTION * denom_scale).any()
        ):
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
    encoder_source: pd.DataFrame | None = None,
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
        encoder_source: Optional separate DataFrame (e.g. the training split)
            used to fit the target-mean encoding for categorical constituents.
            When provided, categoricals are encoded with means computed on
            ``encoder_source`` and merely applied to ``data``, so the MI score
            of a categorical synthetic isn't inflated by encoding ``data``'s
            own (held-out) target. When None, encoding is fit in-sample on
            ``data`` (backward-compatible default).

    Returns:
        List of SyntheticFeatureSuggestion sorted by mi_score descending.
    """
    config = config or SyntheticConfig()
    top_pairs = interactions.head(config.interaction_top_n)
    target = data[target_column]

    encoder_means = _fit_feature_means(encoder_source, target_column)

    suggestions: list[SyntheticFeatureSuggestion] = []
    seen_names: set[str] = set()

    for _, row in top_pairs.iterrows():
        feature_a: str = row["feature_a"]
        feature_b: str = row["feature_b"]

        col_a = _encode_for_interaction(
            data[feature_a], target, encoder_means.get(feature_a)
        )
        col_b = _encode_for_interaction(
            data[feature_b], target, encoder_means.get(feature_b)
        )
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
    encoders: dict[str, pd.Series] | None = None,
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
        encoders: Optional pre-fit target-mean encodings (see
            :func:`fit_interaction_encoders`). When provided, categorical
            constituents are encoded with this fixed map rather than ``data``'s
            own target, so the same encoding fit on the training split can be
            applied to a held-out split without leakage.

    Returns:
        Tuple of (augmented_df, injected_names).
    """
    target = data[target_column]
    encoders = encoders or {}
    augmented = data.copy()
    injected: list[str] = []

    for suggestion in suggestions[:max_inject]:
        col_a = _encode_for_interaction(
            data[suggestion.feature_a], target, encoders.get(suggestion.feature_a)
        )
        col_b = _encode_for_interaction(
            data[suggestion.feature_b], target, encoders.get(suggestion.feature_b)
        )
        if col_a is None or col_b is None:
            continue
        candidate = _build_candidate(col_a, col_b, suggestion.operator)
        if candidate is None:
            continue
        augmented[suggestion.suggested_name] = candidate
        injected.append(suggestion.suggested_name)

    return augmented, injected
