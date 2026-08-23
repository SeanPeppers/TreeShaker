"""Token-level SHAP attribution for text/transformer models.

Parallel to importance.py, but deliberately not routed through
FeatureSelectionPipeline: variable-length text doesn't fit the tabular
train/val/test-split, collinearity, or leakage-detection machinery built
around named scalar columns. This module and TextImportancePipeline are a
separate, lightweight path for the "which tokens drove this prediction"
question, reusing SHAP's own text explainer (shap.maskers.Text) rather than
building token attribution from scratch. No NN framework dependency: a
tokenizer is any object shap.maskers.Text accepts (a HF tokenizer, a plain
callable, or None for its own whitespace-based default -- confirmed working
via a mocked predict_fn), and predict_fn is any callable
list[str] -> np.ndarray, mirroring config.custom_model's "bring your own
model" contract.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_token_shap_importance(
    predict_fn: Callable[[list[str]], np.ndarray],
    texts: list[str],
    tokenizer: Any | None = None,
) -> pd.DataFrame:
    """
    Compute per-token SHAP attribution for each input text.

    Args:
        predict_fn: Callable taking a list of strings and returning an
            array of shape (n_texts,) (binary/regression score) or
            (n_texts, n_classes) (multiclass/multi-output). Any model can
            be wrapped into this shape -- an HF pipeline, a scikit-learn
            text classifier, a custom scoring function -- TreeShaker itself
            has no model dependency here.
        texts: The input texts to explain.
        tokenizer: Passed to shap.maskers.Text. None (default) uses SHAP's
            own simple whitespace/punctuation tokenizer; pass a HF
            tokenizer (or any tokenizer shap.maskers.Text accepts) for
            subword-aware masking that matches how the model actually sees
            the text.

    Returns:
        Long-format DataFrame with columns: example_index (position in
        texts), token, shap_value (signed, per-class-averaged for
        multiclass). One row per token per example -- pass to
        aggregate_token_importance for a corpus-level summary.
    """
    logger.info(f"Computing token-level SHAP importance for {len(texts)} texts")
    masker = shap.maskers.Text(tokenizer=tokenizer)
    explainer = shap.Explainer(predict_fn, masker)
    explanation = explainer(list(texts))

    rows: list[dict[str, Any]] = []
    for i, (tokens, values) in enumerate(
        zip(explanation.data, explanation.values, strict=True)
    ):
        values = np.asarray(values)
        # Multiclass/multi-output predict_fn: values is (n_tokens, n_classes)
        # for this example -- average over classes/outputs, matching
        # compute_shap_importance's multiclass handling in importance.py.
        per_token = values.mean(axis=-1) if values.ndim == 2 else values
        for token, value in zip(tokens, per_token, strict=True):
            rows.append(
                {
                    "example_index": i,
                    "token": str(token).strip(),
                    "shap_value": float(value),
                }
            )
    return pd.DataFrame(rows, columns=["example_index", "token", "shap_value"])


def aggregate_token_importance(token_shap: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a long-format per-example token SHAP table (see
    compute_token_shap_importance) into one row per distinct token across
    the whole corpus.

    Args:
        token_shap: Output of compute_token_shap_importance.

    Returns:
        DataFrame with columns: token, mean_abs_shap, n_occurrences, sorted
        by mean_abs_shap descending. mean_abs_shap is the mean of |SHAP|
        across every occurrence of that token -- "how much this token
        consistently drives predictions when present," analogous to the
        tabular pipeline's per-feature SHAP importance.
    """
    if token_shap.empty:
        return pd.DataFrame(columns=["token", "mean_abs_shap", "n_occurrences"])
    grouped = token_shap.assign(abs_shap=token_shap["shap_value"].abs()).groupby(
        "token", as_index=False
    )
    summary = grouped.agg(
        mean_abs_shap=("abs_shap", "mean"), n_occurrences=("abs_shap", "size")
    )
    return summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def summarize_attention(
    attention_weights: np.ndarray,
    tokens: list[str],
) -> pd.DataFrame:
    """
    Aggregate raw attention weights onto tokens: for each token, how much
    total attention it receives from the rest of the sequence, averaged
    over heads and layers.

    NOT a validated measure of causal feature importance -- attention
    weights show where the model *looked*, not what causally drove its
    output (the "attention is not explanation" finding is well-established
    in the interpretability literature). Treat this as an exploratory
    diagnostic alongside compute_token_shap_importance's real attribution,
    never as a standalone Keep/Caution/Drop-style verdict.

    Takes the raw attention tensor directly rather than a model+tokenizer,
    so it has no dependency on any specific model-calling convention (e.g.
    Hugging Face's output_attentions=True) -- the caller runs their model
    however they normally would and hands the resulting weights here.

    Args:
        attention_weights: Array of shape (n_layers, n_heads, seq_len,
            seq_len) or (n_heads, seq_len, seq_len). attention_weights[...,
            i, j] is how much position i attends to position j.
        tokens: Token strings for the seq_len positions, same order as the
            attention tensor's last two axes.

    Returns:
        DataFrame with columns: token, mean_attention_received, sorted
        descending. mean_attention_received is the mean, over all
        layers/heads and all attending (query) positions, of the attention
        weight this token received.
    """
    if attention_weights.ndim == 3:
        attention_weights = attention_weights[np.newaxis, ...]
    if attention_weights.shape[-1] != len(tokens):
        raise ValueError(
            f"attention_weights' last axis ({attention_weights.shape[-1]}) must"
            f" match len(tokens) ({len(tokens)})."
        )
    # Mean over layers, heads, and query (attending) positions -- axes
    # (0, 1, 2) of (n_layers, n_heads, seq_len_query, seq_len_key) -- leaves
    # one value per key position (the token being attended *to*).
    mean_attention_received = attention_weights.mean(axis=(0, 1, 2))
    return (
        pd.DataFrame(
            {
                "token": tokens,
                "mean_attention_received": mean_attention_received,
            }
        )
        .sort_values("mean_attention_received", ascending=False)
        .reset_index(drop=True)
    )


@dataclass
class TextConfig:
    """Configuration for TextImportancePipeline."""

    tokenizer: Any | None = None
    """Passed to shap.maskers.Text -- see compute_token_shap_importance."""
    top_n: int = 20
    """Number of top tokens (by mean_abs_shap) to keep in the results table."""
    output_file: str = "text_results.csv"
    """Path the aggregate results table is written to."""


@dataclass
class TextImportancePipeline:
    """
    Lightweight, standalone pipeline for token-level SHAP attribution on
    text/transformer models -- the text counterpart to
    FeatureSelectionPipeline, but not a mode of it (see module docstring
    for why: variable-length text doesn't fit the tabular pipeline's
    column-shaped machinery).

    Example:
        >>> from treeshaker.text_importance import (
        ...     TextImportancePipeline, TextConfig,
        ... )
        >>> pipeline = TextImportancePipeline(
        ...     predict_fn=my_model_predict,
        ...     texts=["this movie is good", "this movie is bad"],
        ...     config=TextConfig(top_n=10),
        ... )
        >>> results = pipeline.run()
    """

    predict_fn: Callable[[list[str]], np.ndarray]
    texts: list[str]
    config: TextConfig = field(default_factory=TextConfig)

    def __post_init__(self) -> None:
        self.token_shap: pd.DataFrame | None = None
        self.results: pd.DataFrame | None = None

    def run(self) -> pd.DataFrame:
        """
        Compute per-token SHAP attribution across self.texts, aggregate to
        one row per distinct token, keep the top config.top_n by
        mean_abs_shap, and save to config.output_file.

        Returns:
            The aggregate results DataFrame (see aggregate_token_importance).
        """
        self.token_shap = compute_token_shap_importance(
            self.predict_fn, self.texts, tokenizer=self.config.tokenizer
        )
        aggregated = aggregate_token_importance(self.token_shap)
        self.results = aggregated.head(self.config.top_n).reset_index(drop=True)
        self.results.to_csv(self.config.output_file, index=False)
        logger.info(
            f"Token importance computed for {len(self.texts)} texts;"
            f" top {len(self.results)} tokens saved to {self.config.output_file}"
        )
        return self.results
