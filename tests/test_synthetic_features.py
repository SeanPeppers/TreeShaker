"""Tests for synthetic_features.py: SHAP-interaction-driven feature suggestion."""

import numpy as np
import pandas as pd
import pytest

from treeshaker.synthetic_features import (
    SyntheticConfig,
    SyntheticFeatureSuggestion,
    suggest_synthetic_features,
)


@pytest.fixture
def crossing_data():
    """Regression data where the target is driven by a product of two features.

    feature_a and feature_b are independent noise; the target is (mostly)
    feature_a * feature_b, so neither raw feature alone carries much signal
    but their product does. This gives the synthetic-feature candidate a
    real MI signal to detect, while an unrelated feature_c stays pure noise.
    """
    rng = np.random.default_rng(42)
    n = 400
    feature_a = rng.normal(size=n)
    feature_b = rng.normal(size=n)
    feature_c = rng.normal(size=n)
    target = feature_a * feature_b + rng.normal(scale=0.05, size=n)
    return pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            "target": target,
        }
    )


@pytest.fixture
def interactions_df():
    """A minimal interaction DataFrame matching compute_shap_interactions() output."""
    return pd.DataFrame(
        {
            "feature_a": ["feature_a", "feature_a", "feature_b"],
            "feature_b": ["feature_b", "feature_c", "feature_c"],
            "interaction_strength": [0.9, 0.2, 0.1],
        }
    )


def test_synthetic_config_defaults():
    cfg = SyntheticConfig()
    assert cfg.synthesize_from_interactions is False
    assert cfg.interaction_top_n == 5
    assert cfg.mi_threshold == 0.01


def test_suggest_synthetic_features_finds_real_crossing(crossing_data, interactions_df):
    """A strong real interaction should surface as a high-MI suggestion."""
    suggestions = suggest_synthetic_features(
        data=crossing_data,
        target_column="target",
        interactions=interactions_df,
        task="regression",
        random_seed=42,
        config=SyntheticConfig(mi_threshold=0.01),
    )

    assert len(suggestions) >= 1
    top = suggestions[0]
    assert isinstance(top, SyntheticFeatureSuggestion)
    assert {top.feature_a, top.feature_b} == {"feature_a", "feature_b"}
    assert top.suggested_name == "feature_a__x__feature_b"
    assert top.mi_score > 0.01


def test_suggest_synthetic_features_sorted_descending_by_mi(
    crossing_data, interactions_df
):
    suggestions = suggest_synthetic_features(
        data=crossing_data,
        target_column="target",
        interactions=interactions_df,
        task="regression",
        random_seed=42,
        config=SyntheticConfig(mi_threshold=0.0),
    )
    scores = [s.mi_score for s in suggestions]
    assert scores == sorted(scores, reverse=True)


def test_mi_threshold_filters_weak_candidates(crossing_data, interactions_df):
    """A very high threshold should filter out every candidate."""
    suggestions = suggest_synthetic_features(
        data=crossing_data,
        target_column="target",
        interactions=interactions_df,
        task="regression",
        random_seed=42,
        config=SyntheticConfig(mi_threshold=10.0),
    )
    assert suggestions == []


def test_interaction_top_n_limits_candidates_considered(crossing_data, interactions_df):
    """With top_n=1, only the strongest pair (feature_a/feature_b) is considered."""
    suggestions = suggest_synthetic_features(
        data=crossing_data,
        target_column="target",
        interactions=interactions_df,
        task="regression",
        random_seed=42,
        config=SyntheticConfig(interaction_top_n=1, mi_threshold=0.0),
    )
    pairs = {(s.feature_a, s.feature_b) for s in suggestions}
    assert pairs <= {("feature_a", "feature_b")}


def test_suggest_synthetic_features_empty_interactions_returns_empty(crossing_data):
    empty = pd.DataFrame(columns=["feature_a", "feature_b", "interaction_strength"])
    suggestions = suggest_synthetic_features(
        data=crossing_data,
        target_column="target",
        interactions=empty,
        task="regression",
        random_seed=42,
    )
    assert suggestions == []
