"""Tests for DFE extension: ratio operator, categorical interactions, loop feedback."""

import numpy as np
import pandas as pd
import pytest

from treeshaker.synthetic_features import (
    SyntheticConfig,
    SyntheticFeatureSuggestion,
    _build_candidate,
    _encode_for_interaction,
    build_injection_frame,
    suggest_synthetic_features,
)


def _make_interactions(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature_a": a, "feature_b": b, "interaction_strength": 1.0 - i * 0.1}
            for i, (a, b) in enumerate(pairs)
        ]
    )


# --- _encode_for_interaction ---


def test_encode_numeric_passthrough():
    s = pd.Series([1.0, 2.0, 3.0])
    result = _encode_for_interaction(s, pd.Series([0, 1, 0]))
    pd.testing.assert_series_equal(result, s)


def test_encode_categorical_target_mean():
    s = pd.Categorical(["a", "a", "b", "b"])
    target = pd.Series([0.0, 1.0, 1.0, 1.0])
    result = _encode_for_interaction(pd.Series(s), target)
    assert result is not None
    assert result.iloc[0] == pytest.approx(0.5)  # mean of [0,1] for "a"
    assert result.iloc[2] == pytest.approx(1.0)  # mean of [1,1] for "b"


def test_encode_object_dtype():
    s = pd.Series(["x", "x", "y"])
    target = pd.Series([1.0, 1.0, 0.0])
    result = _encode_for_interaction(s, target)
    assert result is not None
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[2] == pytest.approx(0.0)


# --- _build_candidate ---


def test_build_product():
    a = pd.Series([2.0, 3.0])
    b = pd.Series([4.0, 5.0])
    result = _build_candidate(a, b, "product")
    assert result is not None
    np.testing.assert_array_almost_equal(result, [8.0, 15.0])


def test_build_ratio():
    a = pd.Series([6.0, 9.0])
    b = pd.Series([2.0, 3.0])
    result = _build_candidate(a, b, "ratio")
    assert result is not None
    np.testing.assert_array_almost_equal(result, [3.0, 3.0])


def test_build_ratio_skipped_when_zero_denominator():
    a = pd.Series([1.0, 2.0])
    b = pd.Series([0.0, 1.0])
    result = _build_candidate(a, b, "ratio")
    assert result is None


# --- suggest_synthetic_features ---


def test_product_and_ratio_both_suggested():
    rng = np.random.default_rng(42)
    n = 200
    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n) + 1.0
    target = X1 * X2 + rng.standard_normal(n) * 0.1
    data = pd.DataFrame({"X1": X1, "X2": X2, "y": target})
    interactions = _make_interactions([("X1", "X2")])
    cfg = SyntheticConfig(
        synthesize_from_interactions=True,
        interaction_top_n=1,
        mi_threshold=0.0,
        operators=["product", "ratio"],
    )
    suggestions = suggest_synthetic_features(
        data, "y", interactions, "regression", 42, cfg
    )
    operators = {s.operator for s in suggestions}
    assert "product" in operators


def test_ratio_skipped_for_zero_denominator_in_suggest():
    data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [0.0, 0.0, 0.0], "y": [1, 0, 1]})
    interactions = _make_interactions([("A", "B")])
    cfg = SyntheticConfig(operators=["ratio"], mi_threshold=0.0)
    suggestions = suggest_synthetic_features(data, "y", interactions, "binary", 42, cfg)
    assert all(s.operator != "ratio" for s in suggestions)


def test_categorical_interaction_suggested():
    rng = np.random.default_rng(0)
    n = 300
    cat = pd.Categorical(rng.choice(["low", "high"], n))
    num = rng.standard_normal(n)
    target = (cat == "high").astype(float) * num + rng.standard_normal(n) * 0.2
    data = pd.DataFrame({"cat": cat, "num": num, "y": target})
    interactions = _make_interactions([("cat", "num")])
    cfg = SyntheticConfig(
        synthesize_from_interactions=True,
        interaction_top_n=1,
        mi_threshold=0.0,
        operators=["product"],
    )
    suggestions = suggest_synthetic_features(
        data, "y", interactions, "regression", 42, cfg
    )
    assert len(suggestions) >= 1
    assert suggestions[0].feature_a == "cat" or suggestions[0].feature_b == "cat"


def test_default_operators_include_ratio():
    cfg = SyntheticConfig()
    assert "ratio" in cfg.operators
    assert "product" in cfg.operators


# --- build_injection_frame ---


def test_build_injection_frame_adds_columns():
    data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0], "y": [0, 1, 0]})
    suggestions = [
        SyntheticFeatureSuggestion("A", "B", "product", "A__x__B", 0.5),
    ]
    augmented, injected = build_injection_frame(data, "y", suggestions, max_inject=5)
    assert "A__x__B" in augmented.columns
    assert injected == ["A__x__B"]
    pd.testing.assert_series_equal(
        augmented["A__x__B"], data["A"] * data["B"], check_names=False
    )


def test_build_injection_frame_respects_max_inject():
    data = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [2.0, 3.0], "C": [3.0, 4.0], "y": [0, 1]}
    )
    suggestions = [
        SyntheticFeatureSuggestion("A", "B", "product", "A__x__B", 0.9),
        SyntheticFeatureSuggestion("A", "C", "product", "A__x__C", 0.8),
    ]
    _, injected = build_injection_frame(data, "y", suggestions, max_inject=1)
    assert len(injected) == 1
    assert injected[0] == "A__x__B"


def test_dfe_loop_feedback_config_defaults():
    cfg = SyntheticConfig()
    assert cfg.dfe_loop_feedback is False
    assert cfg.dfe_max_inject == 5
