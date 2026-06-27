"""Synthetic feature suggestion from SHAP interactions.

A pair of features with a strong SHAP interaction sometimes hints at a useful
derived feature (e.g. a product term) that the model would benefit from
having explicitly, rather than having to reconstruct the interaction from
splits on the two raw columns. This module takes the interaction matrix
already computed by `compute_shap_interactions()` (importance.py) and, for
the top-N strongest pairs, builds a candidate pairwise-product feature and
scores it by mutual information against the target.

This is advisory only: candidates are reported for a human to evaluate, never
auto-injected into the model or training data. It is opt-in (off by default)
and disconnected from the rest of the pipeline -- it only reads the
interaction DataFrame and the raw data, it does not change any other
importance computation.
"""

from dataclasses import dataclass

import pandas as pd

from .importance import compute_mutual_information


@dataclass
class SyntheticConfig:
    """Configuration for SHAP-interaction-driven feature suggestion.

    Args:
        synthesize_from_interactions: If True, after SHAP interactions are
            computed, build candidate pairwise-product features from the
            strongest interacting pairs and score them by mutual
            information. Off by default. Requires
            config.compute_shap_interactions to also be True.
        interaction_top_n: Number of strongest interacting pairs (from the
            interaction DataFrame, already sorted descending) to consider
            as candidates.
        mi_threshold: Minimum mutual information score a candidate must
            exceed (strictly greater than) to be included in the returned
            suggestions. Default 0.01 -- low enough to surface plausible
            candidates, high enough to filter out noise-level scores.
    """

    synthesize_from_interactions: bool = False
    interaction_top_n: int = 5
    mi_threshold: float = 0.01


@dataclass
class SyntheticFeatureSuggestion:
    """A single candidate synthetic feature derived from a SHAP interaction.

    Attributes:
        feature_a: First raw feature name in the pair.
        feature_b: Second raw feature name in the pair.
        suggested_name: Proposed column name for the candidate feature,
            formatted as "{feature_a}__x__{feature_b}".
        mi_score: Mutual information between the candidate (elementwise
            product of feature_a and feature_b) and the target.
    """

    feature_a: str
    feature_b: str
    suggested_name: str
    mi_score: float


def suggest_synthetic_features(
    data: pd.DataFrame,
    target_column: str,
    interactions: pd.DataFrame,
    task: str,
    random_seed: int,
    config: SyntheticConfig | None = None,
) -> list[SyntheticFeatureSuggestion]:
    """
    Suggest candidate pairwise-product features from strong SHAP interactions.

    Takes the top `config.interaction_top_n` rows of `interactions` (already
    sorted descending by interaction_strength -- see
    `compute_shap_interactions()`), constructs an elementwise-product
    candidate feature for each pair, computes its mutual information against
    the target, and keeps only candidates whose MI score exceeds
    `config.mi_threshold`.

    Args:
        data: DataFrame including both features and the target column
            (typically the held-out test set), used to build candidate
            columns and compute mutual information.
        target_column: Name of the target column in data.
        interactions: DataFrame with columns feature_a, feature_b,
            interaction_strength, as returned by compute_shap_interactions().
            Must be sorted descending by interaction_strength (the standard
            output order); only the first interaction_top_n rows are used.
        task: One of "binary", "multiclass", or "regression".
        random_seed: Seed for reproducible mutual information results.
        config: A SyntheticConfig instance. interaction_top_n and
            mi_threshold are read from it; synthesize_from_interactions is
            not checked here (callers are expected to gate on it before
            calling this function).

    Returns:
        List of SyntheticFeatureSuggestion, sorted by mi_score descending,
        limited to candidates with mi_score > config.mi_threshold. Empty
        list if no candidate clears the threshold or interactions is empty.
    """
    config = config or SyntheticConfig()
    top_pairs = interactions.head(config.interaction_top_n)

    suggestions: list[SyntheticFeatureSuggestion] = []
    for _, row in top_pairs.iterrows():
        feature_a = row["feature_a"]
        feature_b = row["feature_b"]
        suggested_name = f"{feature_a}__x__{feature_b}"

        candidate = data[[target_column]].copy()
        candidate[suggested_name] = data[feature_a] * data[feature_b]

        mi_score = compute_mutual_information(
            data=candidate,
            target_column=target_column,
            feature_names=[suggested_name],
            task=task,
            random_seed=random_seed,
        )[0]

        if mi_score > config.mi_threshold:
            suggestions.append(
                SyntheticFeatureSuggestion(
                    feature_a=feature_a,
                    feature_b=feature_b,
                    suggested_name=suggested_name,
                    mi_score=float(mi_score),
                )
            )

    suggestions.sort(key=lambda s: s.mi_score, reverse=True)
    return suggestions
