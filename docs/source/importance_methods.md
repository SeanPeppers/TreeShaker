# Feature importance methods

TreeShaker uses **three independent importance signals** — SHAP, permutation importance, and mutual information — computed on the held-out test set. Cross-checking them reduces the chance of a single noisy metric driving a harmful recommendation.

## SHAP importance

SHAP (SHapley Additive exPlanations) values decompose each prediction into the contribution of each feature. TreeShaker computes **mean absolute SHAP** for each feature across all test samples — how much weight the model assigns to that feature on average.

SHAP is **stable under class imbalance** — a feature's SHAP value doesn't collapse just because the minority class is rare.

```python
from treeshaker import FeatureSelectionPipeline, Config

config = Config(
    input_file="data.csv",
    target_column="churn",
    task="binary",
)
pipeline = FeatureSelectionPipeline(config)
results = pipeline.run()

# SHAP values are the primary ranking signal
print(results[["feature_name", "shap_importance"]].head())
```

## Permutation importance

Shuffling one feature's values at a time and measuring how much the model's score drops. A single shuffle is noisy, so TreeShaker shuffles each feature multiple times (default `--n-permutations 5`) and reports the **mean** drop as `permutation_importance`, plus the **std** as `permutation_importance_std`.

Permutation importance is **unreliable under severe class imbalance** — shuffling barely moves an already-insensitive metric like accuracy, regardless of how much the model actually relies on the feature. When severe imbalance is detected, low permutation importance alone no longer flags a feature as harmful; SHAP importance must also be low.

```bash
# Increase the number of shuffles for more stable estimates
treeshaker --input data.csv --target churn --task binary --n-permutations 10
```

### Custom metric for permutation importance

By default, the metric is chosen automatically per task (ROC-AUC for binary, accuracy for multiclass, negative MSE for regression). Override with `--permutation-metric`:

```bash
treeshaker --input data.csv --target volume --task regression --permutation-metric wape
```

Supported values: `roc_auc`, `accuracy`, `neg_mse`, `wape`. `wape` (Weighted Absolute Percentage Error) is standard for volume/demand forecasting — a relative-error read in percent-of-total-volume terms.

## Mutual information

Model-agnostic mutual information between each feature and the target. This measures dependence independent of whether the model actually uses the feature — a cross-check on SHAP and permutation importance.

For continuous targets (regression), TreeShaker uses `sklearn.feature_selection.mutual_info_regression`. For discrete targets (binary/multiclass), it uses `mutual_info_classif`.

Missing values are median-imputed internally for mutual information computation only — this does not affect the model fit or any other check.

## Bootstrap importance stability

A single model fit's importance ranking can be noisy, especially on smaller datasets — a feature ranked 3rd might rank 6th on a different sample.

Set `--n-bootstrap` to refit the model on bootstrap resamples (sampled with replacement from the combined train+val data) using the **same** tuned hyperparameters Optuna found:

```bash
treeshaker --input data.csv --target churn --task binary --n-bootstrap 30
```

This adds two columns to output: `shap_importance_mean` and `shap_importance_std`, the mean and std of each feature's SHAP importance across the refits. High `shap_importance_std` relative to `shap_importance_mean` signals a feature whose ranking isn't stable.

**Note:** Permutation importance is skipped per iteration since it requires a full extra prediction pass per feature, which would be expensive. SHAP alone is enough to assess ranking stability.

## SHAP interaction values

Plain SHAP importance ranks each feature independently. It can miss cases where two features are individually weak but jointly carry a lot of signal — a non-linear interaction.

Pass `--shap-interactions` to additionally compute pairwise SHAP interaction values:

```bash
treeshaker --input data.csv --target churn --task binary --shap-interactions
```

This writes a second file, `<output_stem>_interactions.csv`, with columns `feature_a`, `feature_b`, `interaction_strength`, sorted descending and limited to the top 10 pairs.

**Note:** This is expensive — roughly O(features²) per sample. Off by default.

From the Python API:

```python
config = Config(
    input_file="data.csv",
    target_column="churn",
    task="binary",
    compute_shap_interactions=True,
)
pipeline = FeatureSelectionPipeline(config)
results = pipeline.run()
print(pipeline.interactions)  # DataFrame of top 10 pairs
```

## Feature synthesis from interactions

[SHAP interaction values](#shap-interaction-values) tell you *which* feature pairs interact strongly. Feature synthesis takes the strongest pairs and builds candidate **product features** for each, scoring them by mutual information against the target.

Pass `--shap-interactions --synthesize-features` (requires both flags):

```bash
treeshaker --input data.csv --target churn --task binary \
  --shap-interactions --synthesize-features \
  --synthesis-top-n 5 --synthesis-mi-threshold 0.01
```

This writes `<output_stem>_suggested_features.csv` with candidates named `"{feature_a}__x__{feature_b}"` that scored high on mutual information.

**Advisory only** — suggested features are never added to the model or training data, and never change any ranking or recommendation. You decide whether to engineer them.

From the Python API:

```python
from treeshaker.synthetic_features import SyntheticConfig

config = Config(
    input_file="data.csv",
    target_column="churn",
    task="binary",
    compute_shap_interactions=True,
    synthetic_config=SyntheticConfig(
        synthesize_from_interactions=True,
        interaction_top_n=5,
        mi_threshold=0.01,
    ),
)
pipeline = FeatureSelectionPipeline(config)
results = pipeline.run()
for s in pipeline.suggested_features:
    print(s.suggested_name, s.mi_score)
```

## Residual correlation diagnostics

SHAP, permutation importance, and mutual information all describe a feature's importance, but none of them look at the model's *prediction errors*. A feature can rank as unimportant on all three signals and still correlate with where the model is systematically wrong.

Pass `--check-residuals` to compute per-feature correlation with model prediction residuals on the held-out test set:

```bash
treeshaker --input data.csv --target price --task regression --check-residuals
```

This adds a `residual_correlation` column to output. When a feature flagged `Drop` correlates with residuals, the recommendation is **downgraded to `Caution`** with a note, since residual correlation contradicts the importance-based case for dropping it.

**Definition of "residual" per task:**

- **Regression** — signed residual, `y_true - y_pred`
- **Binary** — absolute calibration error in probability space, `|y_true - predict_proba(X)[:, 1]|`
- **Multiclass** — one minus the probability assigned to the true class

```python
from treeshaker.residual_diagnostics import ResidualConfig

config = Config(
    input_file="data.csv",
    target_column="price",
    task="regression",
    check_residuals=True,
    residual_config=ResidualConfig(residual_corr_threshold=0.25),
)
pipeline = FeatureSelectionPipeline(config)
results = pipeline.run()
print(results[["feature_name", "recommendation", "residual_correlation"]])
```
