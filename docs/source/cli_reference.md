# CLI reference

Complete command-line interface documentation for TreeShaker.

## Basic syntax

```bash
treeshaker --input <file> --target <column> --task <type> [options]
```

## Required arguments

| Argument | Description |
|---|---|
| `--input FILE` | Path to input CSV file |
| `--target COLUMN` | Target column name |
| `--task TASK` | Prediction task: `binary`, `multiclass`, or `regression` |

## Output

| Argument | Default | Description |
|---|---|---|---|
| `--output FILE` | `treeshaker_results.csv` | Output file path |
| `--format FORMAT` | `csv` | Output format: `csv`, or `html` for a report with a color-coded Keep/Caution/Drop table and an embedded SHAP importance chart |

## Data splits

| Argument | Default | Description |
|---|---|---|---|
| `--test-size FRAC` | `0.2` | Fraction of full dataset held out as final test set |
| `--val-size FRAC` | `0.2` | Fraction of remaining train+val pool used for validation |
| `--time-column COL` | `None` | Optional column name for chronological train/val/test splitting |

**Example:** `--test-size 0.2 --val-size 0.2` means 80% train+val, 20% test; then 80%*(1-0.2)=64% train, 16% val.

## Importance thresholds

| Argument | Default | Description |
|---|---|---|---|
| `--shap-threshold FLOAT` | `0.01` | Flag features with mean \|SHAP\| below this |
| `--perm-threshold FLOAT` | `0.05` | Flag features with permutation importance below this |

## Leakage detection

| Argument | Default | Description |
|---|---|---|---|
| `--target-corr-threshold FLOAT` | `0.95` | Flag numeric features with \|correlation to target\| ≥ this |
| `--dominant-shap-fraction FLOAT` | `0.7` | Flag if one feature accounts for ≥ this fraction of total SHAP |
| `--skip-leakage-check` | off | Disable all leakage and logical-error checks |
| `--leakage-screening MODE` | `exact` | Algorithm for near-duplicate numeric feature screening: `exact` (pairwise correlation, always accurate) or `approximate` (SimHash-based LSH via optional Rust extension, faster on wide data) |

## Hyperparameter tuning (Optuna)

| Argument | Default | Description |
|---|---|---|---|
| `--optuna-trials INT` | `50` | Number of hyperparameter tuning trials |
| `--optuna-timeout INT` | `300` | Optuna timeout in seconds |
| `--skip-tuning` | off | Skip Optuna; train directly with configured XGBoost params |

## XGBoost configuration

| Argument | Default | Description |
|---|---|---|---|
| `--xgboost-params JSON` | `{}` | Custom XGBoost parameters as a JSON string |
| `--device DEVICE` | `auto` | Device XGBoost trains on: `auto`, `cuda`, or `cpu` |

**Example:** `--xgboost-params '{"max_depth": 10, "learning_rate": 0.05}'`

## Feature grouping

| Argument | Default | Description |
|---|---|---|---|
| `--feature-groups JSON` | `{}` | Feature groups as JSON mapping group name to feature list |
| `--group-by-prefix` | off | Auto-derive groups from common prefix before last underscore |

**Examples:**
```bash
# Explicit grouping
--feature-groups '{"price": ["price_min", "price_max"], "time": ["timestamp_year", "timestamp_month"]}'

# Auto-grouping (disabled if --feature-groups is set)
--group-by-prefix
```

## Permutation importance

| Argument | Default | Description |
|---|---|---|---|
| `--n-permutations INT` | `5` | Number of shuffles per feature; mean is reported |
| `--permutation-metric METRIC` | auto | Override metric for permutation importance only |

**Supported metrics:** `roc_auc` (binary), `accuracy` (multiclass), `neg_mse` (regression), `wape` (volume forecasting).

## Bootstrap importance stability (opt-in)

| Argument | Default | Description |
|---|---|---|---|
| `--n-bootstrap INT` | `0` | Number of bootstrap refits (0 = disabled) |
| `--bootstrap-sample-frac FLOAT` | `1.0` | Fraction of train+val rows per bootstrap sample |
| `--bootstrap-stability-threshold FLOAT` | `None` | When set and `--n-bootstrap` > 0, gates recommendations on SHAP importance ranking stability (Drop/Keep → Caution if exceeded) |

**Example:** `--n-bootstrap 30` adds `shap_importance_mean` and `shap_importance_std` columns.

## Advanced checks (opt-in)

| Argument | Default | Description |
|---|---|---|---|
| `--shap-interactions` | off | Compute pairwise SHAP interaction values (expensive) |
| `--check-residuals` | off | Compute per-feature correlation with model residuals |
| `--residual-corr-threshold FLOAT` | `0.3` | Flag if \|residual correlation\| ≥ this (only with `--check-residuals`) |
| `--use-boruta` | off | Run Boruta feature selection and add `boruta_decision` column |
| `--use-rfecv` | off | Run sklearn RFECV feature selection and add `rfecv_ranking` column |
| `--use-lightgbm` | off | Compute LightGBM feature importances as `lightgbm_importance` column |
| `--use-catboost` | off | Compute CatBoost feature importances as `catboost_importance` column |
| `--use-rf-baseline` | off | Compute Random Forest baseline importances as `rf_importance` column |
| `--use-sklearn-permutation-importance` | off | Compute sklearn permutation importance as `sklearn_permutation_importance` column |
| `--use-gpu` | off | Enable GPU acceleration for XGBoost (sets device to `cuda`) |
| `--use-optuna-pruning` | off | Enable Optuna trial pruning and early stopping during hyperparameter tuning |
| `--flag-leakage-importance-mismatch` | off | Flag features that are high-importance (Keep) but have active leakage warnings |
| `--compute-train-importance` | off | Compute SHAP importance on train+val in addition to test, to detect train-vs-test gaps |
| `--train-test-delta-threshold FLOAT` | `None` | When `--compute-train-importance` is set, flag features whose train-vs-test SHAP importance delta exceeds this |
| `--compute-importance-divergence` | off | Compute a per-feature divergence score across SHAP/permutation/mutual-info rankings |


## Feature synthesis (opt-in)

| Argument | Default | Description |
|---|---|---|---|
| `--synthesize-features` | off | Suggest product features from SHAP interactions (requires `--shap-interactions`) |
| `--synthesis-top-n INT` | `5` | Number of strongest interaction pairs to consider |
| `--synthesis-mi-threshold FLOAT` | `0.01` | Minimum mutual information score for a candidate |

## Class imbalance

| Argument | Default | Description |
|---|---|---|---|
| `--imbalance-threshold FLOAT` | `0.1` | Warn if minority class fraction < this value |

## Sampling and optimization

| Argument | Default | Description |
|---|---|---|---|
| `--shap-sample-size INT` | unset (all rows) | Cap SHAP computation to this many rows; useful on large datasets |
| `--random-seed INT` | `42` | Random seed for reproducibility |
| `--large-dataset-sample-threshold INT` | `None` | Row count threshold above which auto-stratified sampling is applied before train/val/test splitting |
| `--large-dataset-sample-frac FLOAT` | `0.5` | Fraction of rows to sample when row count exceeds the threshold |

## Parallel feature evaluation

| Argument | Default | Description |
|---|---|---|
| `--parallel-feature-evaluation` | off | Run the per-feature permutation-importance loop across worker processes instead of sequentially |
| `--n-jobs INT` | `-1` | Worker thread count when `--parallel-feature-evaluation` is set (-1 = all available cores); ignored otherwise |
| `--streaming` | off | Opt into the out-of-core / streaming data path: read the CSV in chunks and downcast numeric dtypes per chunk to lower peak memory |
| `--streaming-chunk-size INT` | `100000` | Rows per chunk when `--streaming` is set; ignored otherwise |
| `--n-stability-runs INT` | `0` | Re-run the pipeline N times with different random seeds; appends `stability_score` and `stability_votes` columns. Use with `--skip-tuning` for reasonable runtime |
| `--stability-seeds INT [INT ...]` | auto | Explicit seeds for stability runs (must match `--n-stability-runs`); auto-derived from `--random-seed` when omitted |

## Experiment tracking (opt-in)

| Argument | Default | Description |
|---|---|---|
| `--use-mlflow` | off | Log this run to MLflow for experiment tracking (requires the optional `mlflow` package, `pip install treeshaker[mlflow]`) |
| `--mlflow-experiment-name NAME` | `treeshaker` | MLflow experiment name to log runs under; only used when `--use-mlflow` is set |

## Miscellaneous

| Argument | Description |
|---|---|
| `--verbose` | Enable debug logging |
| `--help` | Show help message and exit |

## Examples

### Binary classification with custom thresholds

```bash
treeshaker --input customer_data.csv --target churn --task binary \
  --shap-threshold 0.02 --perm-threshold 0.1 \
  --output churn_analysis.csv
```

### Regression with HTML report

```bash
treeshaker --input housing.csv --target price --task regression \
  --format html --output housing_report.html
```

### Multiclass with Optuna tuning

```bash
treeshaker --input iris.csv --target species --task multiclass \
  --optuna-trials 100 --optuna-timeout 600
```

### Custom XGBoost hyperparameters, skip tuning

```bash
treeshaker --input data.csv --target y --task regression \
  --skip-tuning \
  --xgboost-params '{"max_depth": 8, "learning_rate": 0.05, "tree_method": "hist"}'
```

### Bootstrap importance for ranking stability

```bash
treeshaker --input data.csv --target churn --task binary \
  --n-bootstrap 30 \
  --output churn_results.csv
```

### SHAP interactions and feature synthesis

```bash
treeshaker --input data.csv --target churn --task binary \
  --shap-interactions \
  --synthesize-features --synthesis-top-n 10 --synthesis-mi-threshold 0.01
```

### Feature groups with automatic prefix grouping

```bash
treeshaker --input data.csv --target target --task regression \
  --group-by-prefix
```

### Residual diagnostics for regression

```bash
treeshaker --input data.csv --target price --task regression \
  --check-residuals --residual-corr-threshold 0.25
```

### GPU acceleration (CUDA)

```bash
treeshaker --input large_data.csv --target y --task regression \
  --device cuda
```

### Large dataset with SHAP sampling

```bash
treeshaker --input huge_dataset.csv --target target --task binary \
  --shap-sample-size 5000 \
  --n-permutations 10
```

### Disable leakage detection

```bash
treeshaker --input data.csv --target y --task regression \
  --skip-leakage-check
```

### Full-featured example

```bash
treeshaker \
  --input data.csv \
  --target churn \
  --task binary \
  --output churn_analysis.csv \
  --format html \
  --test-size 0.2 \
  --val-size 0.2 \
  --shap-threshold 0.015 \
  --perm-threshold 0.08 \
  --optuna-trials 100 \
  --optuna-timeout 600 \
  --xgboost-params '{"max_depth": 8, "learning_rate": 0.05}' \
  --n-permutations 10 \
  --n-bootstrap 30 \
  --shap-interactions \
  --check-residuals \
  --group-by-prefix \
  --random-seed 42 \
  --verbose
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Error (configuration, validation, or runtime) |

Errors are logged to stderr with a clear, actionable message before exit.
