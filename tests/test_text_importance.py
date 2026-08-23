"""Tests for token-level SHAP attribution (text_importance.py)."""

import numpy as np
import pandas as pd
import pytest

from treeshaker.text_importance import (
    TextConfig,
    TextImportancePipeline,
    aggregate_token_importance,
    compute_token_shap_importance,
    summarize_attention,
)


def _marker_predict_fn(texts: list[str]) -> np.ndarray:
    """Deterministic: 1.0 if 'good' appears in the text, else 0.0."""
    return np.array([1.0 if "good" in t else 0.0 for t in texts])


def test_compute_token_shap_importance_marker_token_gets_high_attribution():
    texts = ["this movie is good", "this movie is bad", "a good story overall"]
    token_shap = compute_token_shap_importance(_marker_predict_fn, texts)

    assert set(token_shap.columns) == {"example_index", "token", "shap_value"}
    good_rows = token_shap[token_shap["token"] == "good"]
    other_rows = token_shap[token_shap["token"] != "good"]
    assert (good_rows["shap_value"].abs() > 0.9).all()
    assert (other_rows["shap_value"].abs() < 1e-6).all()


def test_compute_token_shap_importance_one_row_per_token_per_example():
    texts = ["a b c", "d e"]
    token_shap = compute_token_shap_importance(_marker_predict_fn, texts)
    assert len(token_shap[token_shap["example_index"] == 0]) == 3
    assert len(token_shap[token_shap["example_index"] == 1]) == 2


def test_aggregate_token_importance_groups_and_sorts():
    token_shap = pd.DataFrame(
        {
            "example_index": [0, 0, 1, 1],
            "token": ["good", "movie", "good", "bad"],
            "shap_value": [0.8, 0.0, 1.0, -0.5],
        }
    )
    summary = aggregate_token_importance(token_shap)

    assert list(summary.columns) == ["token", "mean_abs_shap", "n_occurrences"]
    good_row = summary[summary["token"] == "good"].iloc[0]
    assert good_row["mean_abs_shap"] == pytest.approx(0.9)  # mean(0.8, 1.0)
    assert good_row["n_occurrences"] == 2
    # Sorted descending by mean_abs_shap -- "good" (0.9) must rank above
    # "bad" (0.5) and "movie" (0.0).
    assert summary.iloc[0]["token"] == "good"


def test_aggregate_token_importance_empty_input():
    empty = pd.DataFrame(columns=["example_index", "token", "shap_value"])
    summary = aggregate_token_importance(empty)
    assert list(summary.columns) == ["token", "mean_abs_shap", "n_occurrences"]
    assert len(summary) == 0


def test_text_importance_pipeline_end_to_end(tmp_path):
    texts = [
        "this movie is good",
        "this movie is bad",
        "a good story overall",
        "terrible and bad",
    ]
    output_file = tmp_path / "text_results.csv"
    pipeline = TextImportancePipeline(
        predict_fn=_marker_predict_fn,
        texts=texts,
        config=TextConfig(top_n=3, output_file=str(output_file)),
    )
    results = pipeline.run()

    assert isinstance(results, pd.DataFrame)
    assert len(results) <= 3
    assert results.iloc[0]["token"] == "good"
    assert output_file.exists()
    saved = pd.read_csv(output_file)
    assert list(saved["token"]) == list(results["token"])
    assert pipeline.token_shap is not None


def test_summarize_attention_shape_and_columns():
    tokens = ["a", "b", "c", "d", "e"]
    attention = np.random.default_rng(0).random((2, 4, 5, 5))
    summary = summarize_attention(attention, tokens)
    assert list(summary.columns) == ["token", "mean_attention_received"]
    assert len(summary) == 5
    assert set(summary["token"]) == set(tokens)


def test_summarize_attention_accepts_single_layer_tensor():
    """(n_heads, seq_len, seq_len) without a leading layer axis must work
    the same as a (1, n_heads, seq_len, seq_len) tensor."""
    tokens = ["a", "b", "c"]
    rng = np.random.default_rng(1)
    single_layer = rng.random((4, 3, 3))
    multi_layer = single_layer[np.newaxis, ...]

    result_single = summarize_attention(single_layer, tokens)
    result_multi = summarize_attention(multi_layer, tokens)

    pd.testing.assert_frame_equal(result_single, result_multi)


def test_summarize_attention_identifies_most_attended_token():
    """A token that receives uniformly more attention than the others
    across every layer/head/query position must rank first."""
    tokens = ["low", "low", "HIGH"]
    n_layers, n_heads, seq_len = 2, 3, 3
    attention = np.full((n_layers, n_heads, seq_len, seq_len), 0.1)
    attention[..., 2] = 0.9  # every query position attends heavily to "HIGH"

    summary = summarize_attention(attention, tokens)
    assert summary.iloc[0]["token"] == "HIGH"


def test_summarize_attention_rejects_mismatched_token_count():
    attention = np.random.default_rng(0).random((2, 4, 5, 5))
    with pytest.raises(ValueError, match="must match len\\(tokens\\)"):
        summarize_attention(attention, ["only", "three", "tokens"])
