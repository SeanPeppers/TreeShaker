"""Tests for DFE extension: ratio operator, categorical interactions, loop feedback."""

import numpy as np
import pandas as pd
import pytest

from treeshaker.config import Config
from treeshaker.pipeline import FeatureSelectionPipeline
from treeshaker.synthetic_features import (
    SyntheticConfig,
    SyntheticFeatureSuggestion,
    _build_candidate,
    _encode_for_interaction,
    _fit_feature_means,
    build_injection_frame,
    fit_interaction_encoders,
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


def test_build_ratio_skipped_for_near_singular_denominator():
    # 1e-12 is ~zero relative to the median |B| of 1.5 -> extreme blow-up,
    # rejected even though it is nonzero and the raw division stays finite.
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([1e-12, 1.0, 2.0])
    assert _build_candidate(a, b, "ratio") is None


def test_build_ratio_kept_when_denominator_well_scaled():
    # Small but well-scaled denominators (all same order of magnitude) are fine.
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([0.5, 1.0, 1.5])
    result = _build_candidate(a, b, "ratio")
    assert result is not None
    np.testing.assert_array_almost_equal(result, [2.0, 2.0, 2.0])


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


# --- fit_interaction_encoders + leakage-free application ---


def test_fit_interaction_encoders_only_categorical():
    data = pd.DataFrame(
        {"num": [1.0, 2.0, 3.0], "cat": ["a", "a", "b"], "y": [1.0, 0.0, 1.0]}
    )
    suggestions = [
        SyntheticFeatureSuggestion("num", "cat", "product", "num__x__cat", 0.5)
    ]
    encoders = fit_interaction_encoders(data, "y", suggestions)
    assert set(encoders) == {"cat"}  # numeric constituent not encoded
    assert encoders["cat"].loc["a"] == pytest.approx(0.5)


def test_encoder_applied_consistently_across_splits():
    train = pd.DataFrame(
        {
            "cat": ["a", "a", "b", "b"],
            "num": [1.0, 1.0, 1.0, 1.0],
            "y": [0.0, 1.0, 1.0, 1.0],
        }
    )
    # Test split has a totally different target; without a fixed encoder the
    # categorical encoding would leak the test target.
    test = pd.DataFrame({"cat": ["a", "b"], "num": [1.0, 1.0], "y": [9.0, 9.0]})
    suggestions = [
        SyntheticFeatureSuggestion("cat", "num", "product", "cat__x__num", 0.5)
    ]
    encoders = fit_interaction_encoders(train, "y", suggestions)
    _, injected = build_injection_frame(
        test, "y", suggestions, max_inject=5, encoders=encoders
    )
    aug, _ = build_injection_frame(
        test, "y", suggestions, max_inject=5, encoders=encoders
    )
    # "a" -> train mean 0.5, "b" -> train mean 1.0, times num (1.0)
    assert aug["cat__x__num"].tolist() == pytest.approx([0.5, 1.0])


def test_encode_for_interaction_uses_provided_means():
    s = pd.Series(["a", "b"])
    fixed = pd.Series({"a": 10.0, "b": 20.0})
    result = _encode_for_interaction(s, pd.Series([0.0, 0.0]), means=fixed)
    assert result is not None
    assert result.tolist() == pytest.approx([10.0, 20.0])


def test_encode_unseen_category_filled_with_fit_mean():
    # "c" is unseen in the externally-fit map -> filled with mean(10, 20)=15,
    # not with this series' own target (which would be leakage).
    s = pd.Series(["a", "c"])
    fixed = pd.Series({"a": 10.0, "b": 20.0})
    result = _encode_for_interaction(s, pd.Series([99.0, 99.0]), means=fixed)
    assert result is not None
    assert result.tolist() == pytest.approx([10.0, 15.0])


def test_fit_feature_means_none_returns_empty():
    assert _fit_feature_means(None, "y") == {}


def test_fit_feature_means_computes_categorical_only():
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"], "y": [0.0, 1.0]})
    means = _fit_feature_means(df, "y")
    assert set(means) == {"cat"}
    assert means["cat"].loc["a"] == pytest.approx(0.0)
    assert means["cat"].loc["b"] == pytest.approx(1.0)


def test_suggest_encodes_categoricals_from_encoder_source():
    # Train and test give the categorical opposite target-means. With
    # encoder_source=train, the encoding must reflect TRAIN means, so a
    # candidate built on test uses the train-fit mapping (no test leakage).
    rng = np.random.default_rng(0)
    n = 200
    cats = rng.choice(["a", "b"], n)
    train = pd.DataFrame(
        {"cat": cats, "num": rng.standard_normal(n), "y": (cats == "b").astype(float)}
    )
    # Test uses the inverted target relationship.
    test_cats = rng.choice(["a", "b"], n)
    test = pd.DataFrame(
        {
            "cat": test_cats,
            "num": rng.standard_normal(n),
            "y": (test_cats == "a").astype(float),
        }
    )
    interactions = _make_interactions([("cat", "num")])
    cfg = SyntheticConfig(interaction_top_n=1, mi_threshold=-1.0, operators=["product"])
    # Should run without leaking and produce a suggestion referencing cat.
    suggestions = suggest_synthetic_features(
        test, "y", interactions, "regression", 42, cfg, encoder_source=train
    )
    assert len(suggestions) >= 1
    assert suggestions[0].feature_a == "cat" or suggestions[0].feature_b == "cat"


# --- real competition: _evaluate_synthetic_replacements ---


def _prep_pipeline_for_competition(target_fn) -> FeatureSelectionPipeline:
    rng = np.random.default_rng(0)
    n = 400
    A = rng.uniform(1.0, 5.0, n)
    B = rng.uniform(1.0, 5.0, n)
    y = target_fn(A, B, rng)
    df = pd.DataFrame({"A": A, "B": B, "y": y})
    train, val, test = df.iloc[:240], df.iloc[240:320], df.iloc[320:]

    cfg = Config(
        input_file="unused.csv",
        target_column="y",
        task="regression",
        synthetic_config=SyntheticConfig(dfe_loop_feedback=True, dfe_max_inject=3),
    )
    pipe = FeatureSelectionPipeline(cfg)
    pipe.train_data, pipe.val_data, pipe.test_data = (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
    pipe.feature_names = ["A", "B"]
    pipe.results = pd.DataFrame(
        {"feature_name": ["A", "B"], "recommendation": ["Keep", "Keep"]}
    )
    pipe.suggested_features = [
        SyntheticFeatureSuggestion("A", "B", "product", "A__x__B", 1.0)
    ]
    return pipe


def test_evaluate_replacement_marks_when_synthetic_dominates():
    # target is exactly the product -> synthetic A*B should outrank A and B.
    pipe = _prep_pipeline_for_competition(lambda A, B, rng: A * B)
    replaced, winners = pipe._evaluate_synthetic_replacements()
    assert replaced == {"A", "B"}
    assert "A__x__B" in winners
    assert winners["A__x__B"] > 0.0


def test_evaluate_replacement_empty_when_nothing_injectable():
    # Ratio with an all-zero denominator can't be built -> no injection, so no
    # raw feature is demoted (a synthetic that never entered can't win).
    pipe = _prep_pipeline_for_competition(lambda A, B, rng: A + B)
    for split in (pipe.train_data, pipe.test_data):
        split["B"] = 0.0
    pipe.suggested_features = [
        SyntheticFeatureSuggestion("A", "B", "ratio", "A__div__B", 1.0)
    ]
    assert pipe._evaluate_synthetic_replacements() == (set(), {})


def test_append_synthetic_winner_rows_adds_keep_rows():
    pipe = _prep_pipeline_for_competition(lambda A, B, rng: A * B)
    results = pd.DataFrame(
        {
            "feature_name": ["A", "B"],
            "shap_importance": [0.1, 0.2],
            "recommendation": ["ReplacedBySynthetic", "ReplacedBySynthetic"],
            "reason": ["", ""],
        }
    )
    out = pipe._append_synthetic_winner_rows(results, {"A__x__B": 0.9})
    row = out[out["feature_name"] == "A__x__B"]
    assert len(row) == 1
    assert row.iloc[0]["recommendation"] == "Keep"
    assert row.iloc[0]["shap_importance"] == 0.9
    # existing rows untouched
    assert set(out["feature_name"]) == {"A", "B", "A__x__B"}


def test_append_synthetic_winner_rows_noop_when_empty():
    pipe = _prep_pipeline_for_competition(lambda A, B, rng: A * B)
    results = pd.DataFrame({"feature_name": ["A"], "recommendation": ["Keep"]})
    out = pipe._append_synthetic_winner_rows(results, {})
    pd.testing.assert_frame_equal(out, results)
