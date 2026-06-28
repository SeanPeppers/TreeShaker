# Harm detection

A feature is flagged as harmful (recommendation: `Drop`) if **either** of these conditions is true:

- **Low SHAP** — mean absolute SHAP value < `--shap-threshold` (default `0.01`). The feature contributes near-zero signal to the model's predictions.
- **Low permutation importance** — permutation importance < `--perm-threshold` (default `0.05`). Shuffling the feature doesn't hurt performance, meaning the model isn't using it.

Both thresholds are configurable. Raise them to be more aggressive about flagging features; lower them to only flag the most clearly useless ones.

## Configuring thresholds

```bash
# More aggressive (flag more features as harmful)
treeshaker --input data.csv --target price --task regression \
  --shap-threshold 0.02 --perm-threshold 0.1

# More conservative (only flag the most clearly useless)
treeshaker --input data.csv --target price --task regression \
  --shap-threshold 0.005 --perm-threshold 0.02
```

From the Python API:

```python
config = Config(
    input_file="data.csv",
    target_column="price",
    task="regression",
    shap_importance_threshold=0.02,
    permutation_importance_threshold=0.1,
)
```

## Class imbalance special case

Under severe class imbalance (minority class < `--imbalance-threshold`, default 10%), permutation importance collapses toward zero for *every* feature — shuffling barely moves an already-insensitive metric like accuracy, regardless of how much the model actually relies on the feature.

If low permutation importance alone were still enough to flag harm, **every feature in a severely imbalanced dataset would get marked `Drop`**, including the model's most important drivers.

**So when severe imbalance is detected, low permutation importance alone no longer flags a feature as harmful — SHAP importance must *also* be low** (i.e. both signals must agree). SHAP remains reliable under imbalance, so it's the fallback signal.

TreeShaker logs two plain-English warnings when this happens:

1. Initial warning when class imbalance is detected
2. A second warning: *"Severe class imbalance detected — permutation importance threshold is being skipped for harm detection; relying on SHAP importance alone for this run."*

This only changes behavior when severe imbalance is actually detected; balanced datasets see no change.

### Understanding the imbalance threshold

```bash
# Raise the threshold to be stricter about what counts as "severe"
treeshaker --input data.csv --target churn --task binary \
  --imbalance-threshold 0.15

# Lower it to be more conservative
treeshaker --input data.csv --target rare_event --task binary \
  --imbalance-threshold 0.05
```

From the Python API:

```python
config = Config(
    input_file="data.csv",
    target_column="churn",
    task="binary",
    imbalance_warning_threshold=0.15,
)
```

## Validating the reduction

TreeShaker tells you which features to drop, but the proof that dropping them didn't hurt is something you confirm yourself by comparing two model fits. This is a **recommended manual workflow** — TreeShaker does not currently run this comparison automatically or generate this table for you.

### Step-by-step

1. Run TreeShaker once on the full feature set and note the `Drop` recommendations.
2. Train (or re-run TreeShaker with `--skip-tuning` and the same `--xgboost-params`) one model on **all** features, and a second model on only the features TreeShaker marked `Keep` (optionally `Keep` + `Caution`, depending on how conservative you want to be).
3. Evaluate both models on the same held-out test set, using the metric that matches your task:
   - ROC-AUC for binary classification
   - Accuracy for multiclass classification
   - R² or WAPE for regression (TreeShaker's `--permutation-metric wape` uses the same WAPE definition)
4. Compare metrics:

   | Model | Features used | Metric | Training/inference cost |
   |---|---|---|---|
   | Full | All N features | — | Higher |
   | Reduced | Only `Keep` features | — | Lower |

   If the reduced model's metric is at or near parity with the full model (e.g. WAPE within a small tolerance, or R² not meaningfully lower), the reduction is validated: you get a simpler, cheaper, more explainable model with no real loss of predictive performance. If the reduced model is meaningfully worse, treat that as a signal to revisit the thresholds or move some `Caution` features back to `Keep` rather than trusting the smaller feature set blindly.

### Using bootstrap importance for extra confidence

Before committing to a reduced feature set, run TreeShaker again with bootstrap importance to check ranking stability:

```bash
treeshaker --input data.csv --target price --task regression \
  --n-bootstrap 30 --shap-threshold 0.01 --perm-threshold 0.05
```

A `Drop` recommendation backed by a stable bootstrap ranking (low `shap_importance_std`) is much safer to act on than one based on a single fit. If a feature's ranking jumps around across bootstrap refits, that's a signal to be more conservative about dropping it.

## Error handling

TreeShaker validates common input mistakes up front and reports them in plain English instead of a raw traceback — for example:

- Missing input file
- Empty or malformed CSV
- Target column with no non-null values
- `--test-size`/`--val-size` too small to stratify the test or validation split

All of these exit with code `1` and a one-line, actionable error message.
