# Quick start

Get your first feature selection result in under 5 minutes.

## CLI

```bash
treeshaker --input data.csv --target price --task regression
```

This writes `treeshaker_results.csv` to the current directory with feature rankings and recommendations.

### More examples

```bash
# Binary classification with an HTML report
treeshaker --input data.csv --target churn --task binary \
  --output report.html --format html

# Multiclass with custom Optuna trials and thresholds
treeshaker --input data.csv --target category --task multiclass \
  --optuna-trials 100 \
  --shap-threshold 0.02 \
  --perm-threshold 0.1

# Pass custom XGBoost parameters as JSON
treeshaker --input data.csv --target y --task regression \
  --xgboost-params '{"max_depth": 10, "learning_rate": 0.05}'
```

## Python API

```python
from treeshaker import FeatureSelectionPipeline, Config

config = Config(
    input_file="data.csv",
    target_column="price",
    task="regression",
)

pipeline = FeatureSelectionPipeline(config)
results = pipeline.run()

print(results)
```

### Full control example

```python
from treeshaker import FeatureSelectionPipeline, Config, OptunaConfig, XGBoostConfig

config = Config(
    input_file="data.csv",
    target_column="churn",
    task="binary",
    output_file="results.csv",
    output_format="csv",
    test_size=0.2,
    random_seed=42,
    shap_importance_threshold=0.02,
    permutation_importance_threshold=0.1,
    optuna_config=OptunaConfig(
        n_trials=100,
        timeout=600,
        seed=42,
    ),
    xgboost_config=XGBoostConfig(
        max_depth=8,
        learning_rate=0.05,
        extra_params={"tree_method": "hist"},
    ),
    verbose=True,
)

pipeline = FeatureSelectionPipeline(config)
results = pipeline.run()

# Inspect results
print(results[results["recommendation"] == "Drop"])
```

## Understanding output

Results are a pandas DataFrame (or CSV if using the CLI) with these columns:

| Column | Type | Description |
|---|---|---|
| `feature_name` | str | Feature column name |
| `shap_importance` | float | Mean absolute SHAP value |
| `permutation_importance` | float | Mean performance drop across shuffles |
| `permutation_importance_std` | float | Std dev of performance drop |
| `mutual_info_score` | float | Mutual information with target |
| `recommendation` | str | `Drop`, `Caution`, or `Keep` |
| `reason` | str | Plain-English explanation (empty if `Keep`) |

Example output:

```
feature_name  shap_importance  permutation_importance  mutual_info_score  recommendation  reason
feature_3     0.4821           0.312                   0.398               Keep
feature_0     0.3104           0.198                   0.271               Keep
feature_7     0.1823           0.091                   0.155               Keep
feature_2     0.0043           0.009                   0.002               Drop            The model assigns almost no weight to this feature (SHAP importance: 0.0043, below the 0.010 threshold)...
feature_9     0.0011           -0.002                  0.000               Drop            The model ignores this feature entirely...
customer_id   0.0512           0.041                   0.061               Caution         Column name 'customer_id' looks like a row identifier...
```

## Next steps

- **Validate the reduction** — train two models (all features vs. recommended features only) and compare their metrics on a held-out test set.
- **Configure thresholds** — see [Configuration reference](config_reference.md) for flags like `--shap-threshold` and `--perm-threshold`.
- **Use advanced options** — enable `--n-bootstrap` for ranking stability, `--shap-interactions` for pairwise importance, `--check-residuals` for error diagnostics.
- **Understand the pipeline** — read [How it works](how_it_works.md) to understand the train/val/test split and why importance is trustworthy.
