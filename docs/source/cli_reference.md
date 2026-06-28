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
| `--format FORMAT` | `csv` | Output format: `csv` or `html` |

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
