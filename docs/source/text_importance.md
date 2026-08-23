# Text/transformer token importance

A separate, lightweight companion to the main tabular pipeline: **which
tokens drove a text/transformer model's prediction**, using SHAP's own text
explainer. Not a mode of `FeatureSelectionPipeline` — variable-length text
has no CSV/feature-name shape to route through the tabular
train/val/test-split, collinearity, or leakage-detection machinery built
around named scalar columns.

No NN framework dependency: TreeShaker never imports `transformers`,
`torch`, or similar. You bring a `predict_fn` (any callable
`list[str] -> np.ndarray`) — the same "bring your own model" contract as
`Config.custom_model` — and optionally a tokenizer object.

## Python API

```python
from treeshaker.text_importance import TextImportancePipeline, TextConfig

def predict_fn(texts: list[str]):
    return my_model.predict_proba(texts)[:, 1]

pipeline = TextImportancePipeline(
    predict_fn=predict_fn,
    texts=["this movie is good", "this movie is bad"],
    config=TextConfig(top_n=20, output_file="text_results.csv"),
)
results = pipeline.run()
```

`results` has columns `token`, `mean_abs_shap`, `n_occurrences`, sorted by
`mean_abs_shap` descending — the token-level analog of the tabular
pipeline's per-feature SHAP importance ("how much this token consistently
drives predictions when present").

Lower-level functions, if you want per-example detail instead of the
corpus-level aggregate:
- `compute_token_shap_importance(predict_fn, texts, tokenizer=None)` — long-format
  DataFrame, one row per token per example (`example_index`, `token`, `shap_value`).
- `aggregate_token_importance(token_shap_df)` — collapses that into the
  corpus-level table `TextImportancePipeline.run()` returns.

### Tokenizer

`tokenizer` is passed straight to `shap.maskers.Text`. Omit it (the
default) to use SHAP's own whitespace/punctuation tokenizer; pass a Hugging
Face tokenizer (or anything else `shap.maskers.Text` accepts) for
subword-aware masking that matches how your model actually sees the text.

### Attention inspection (exploratory, not a verdict)

```python
from treeshaker.text_importance import summarize_attention

summary = summarize_attention(attention_weights, tokens)
```

`attention_weights` is a raw `(n_layers, n_heads, seq_len, seq_len)` (or
`(n_heads, seq_len, seq_len)`) array — however you got it from your model
(e.g. Hugging Face's `output_attentions=True`). This function only does the
aggregation math; it has no model-calling convention baked in.

**This is not a validated measure of causal feature importance.** Attention
weights show where the model *looked*, not what causally drove its output
("attention is not explanation" is a well-established interpretability
finding). Use it as an exploratory diagnostic alongside
`compute_token_shap_importance`'s real attribution, never as a standalone
Keep/Caution/Drop-style verdict.

## CLI: `treeshaker-text`

A CLI can't carry a live Python callable, so `--predict-fn` (and
`--tokenizer`) are resolved from a `module:attribute` dotted path — the
same convention as `gunicorn`'s `app:app` or setuptools console-script
entry points.

```bash
treeshaker-text \
  --predict-fn my_model:predict \
  --texts-file reviews.csv --text-column review \
  --output text_results.csv --top-n 20
```

| Argument | Default | Description |
|---|---|---|
| `--predict-fn` | *(required)* | `module:function` — `list[str] -> np.ndarray` |
| `--texts-file` | *(required)* | `.txt` (one text per line) or `.csv` |
| `--text-column` | `None` | Required when `--texts-file` is a CSV |
| `--tokenizer` | `None` | Optional `module:tokenizer` dotted path |
| `--output` | `text_results.csv` | Output CSV path |
| `--top-n` | `20` | Number of top tokens to keep |
| `--verbose` | off | Debug-level logging |

## MCP: `analyze_text_importance`

Same dotted-path convention as the CLI (an MCP tool call can't carry a live
callable either):

```python
analyze_text_importance(
    predict_fn="my_model:predict",
    texts_file="reviews.csv",
    text_column="review",
    output_path="text_results.csv",
    top_n=20,
)
```

Returns `{"top_tokens": [{"token": ..., "mean_abs_shap": ...,
"n_occurrences": ...}, ...], "output_file": ...}` on success, or
`{"error": "<message>"}` on failure — same convention as `run_pipeline` and
the other MCP tools.
