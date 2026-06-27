# TreeShaker

Automated feature selection using XGBoost, SHAP, and Optuna. Give it a CSV and a target column — it tunes a model, ranks every feature by importance, and flags the ones that are hurting you.

## How it works

1. **Tune** — Optuna searches for the best XGBoost hyperparameters on a train/validation split
2. **Train** — a final XGBoost model is fit on the tuned parameters
3. **Rank** — SHAP values and permutation importance are computed for every feature
4. **Flag** — features are marked harmful if their SHAP importance is near zero *or* if shuffling them doesn't degrade the model (i.e., they carry no signal)
5. **Output** — results saved as a ranked CSV or HTML report

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.13+.

---

## Quick start

### CLI

```bash
treeshaker --input data.csv --target price --task regression
```

This writes `treeshaker_results.csv` to the current directory.

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

### Python API

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

Full control with custom settings:

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

# Inspect after the run
print(results[results["recommendation"] == "Drop"])
```

---

## Output

Results are sorted by SHAP importance (descending).

| Column | Type | Description |
|---|---|---|
| `feature_name` | str | Feature column name |
| `shap_importance` | float | Mean absolute SHAP value across validation samples |
| `permutation_importance` | float | Performance drop when the feature is shuffled |
| `recommendation` | str | `Drop`, `Caution`, or `Keep` |
| `reason` | str | Plain-English explanation. Empty if `Keep` |

Example output:

```
feature_name  shap_importance  permutation_importance  recommendation  reason
feature_3     0.4821           0.312                   Keep
feature_0     0.3104           0.198                   Keep
feature_7     0.1823           0.091                   Keep
feature_2     0.0043           0.009                   Drop            The model assigns almost no weight to this feature (SHAP: 0.0043)...
feature_9     0.0011           -0.002                  Drop            The model ignores this feature entirely...
customer_id   0.0512           0.041                   Caution         Column name 'customer_id' looks like a row identifier...
```

`recommendation` logic:

- **Drop** — harmful (low SHAP and/or low permutation importance)
- **Caution** — leakage/logical warning only, model is still using the feature
- **Keep** — no issues found
- If both harmful *and* a leakage warning fire, the feature is still `Drop`, with both reasons combined.

---

## CLI reference

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | Yes | — | Path to input CSV file |
| `--target` | Yes | — | Target column name |
| `--task` | Yes | — | `binary`, `multiclass`, or `regression` |
| `--output` | No | `treeshaker_results.csv` | Output file path |
| `--format` | No | `csv` | `csv` or `html` |
| `--xgboost-params` | No | `{}` | Custom XGBoost params as a JSON string |
| `--optuna-trials` | No | `50` | Number of Optuna tuning trials |
| `--optuna-timeout` | No | `300` | Optuna timeout in seconds |
| `--shap-threshold` | No | `0.01` | Flag features with mean \|SHAP\| below this |
| `--perm-threshold` | No | `0.05` | Flag features with permutation importance below this |
| `--target-corr-threshold` | No | `0.95` | Flag features whose correlation with the target exceeds this |
| `--dominant-shap-fraction` | No | `0.7` | Flag a feature if it accounts for more than this fraction of total SHAP importance |
| `--skip-leakage-check` | No | `False` | Disable all leakage/logical-error checks |
| `--test-size` | No | `0.2` | Fraction of data used for validation |
| `--random-seed` | No | `42` | Random seed for reproducibility |
| `--verbose` | No | `False` | Enable debug logging |

---

## Supported task types

| Task | Target column | Evaluation metric |
|---|---|---|
| `binary` | 2 unique values | ROC-AUC |
| `multiclass` | 3+ unique values | Accuracy |
| `regression` | Numeric | Negative MSE |

---

## Harm detection

A feature is flagged as harmful if **either** condition is true:

- **Low SHAP** — mean absolute SHAP value < `--shap-threshold` (default `0.01`). The feature contributes near-zero signal to the model's predictions.
- **Low permutation importance** — permutation importance < `--perm-threshold` (default `0.05`). Shuffling the feature doesn't hurt performance, meaning the model isn't using it.

Both thresholds are configurable. Raise them to be more aggressive about flagging features; lower them to only flag the most clearly useless ones.

---

## Leakage detection

Five independent checks flag features that may be causing data leakage or are otherwise logically suspect. A feature can trigger more than one.

| Check | Trigger | Why it matters |
|---|---|---|
| Target correlation | `\|r\|` with target ≥ `--target-corr-threshold` (default `0.95`) | Feature may have been derived from the target |
| SHAP dominance | One feature accounts for ≥ `--dominant-shap-fraction` (default `0.7`) of total SHAP | A single feature this dominant may leak future/target info |
| Near-constant | Same value in > 99% of rows | No signal, wastes model capacity |
| Near-duplicate | Correlation ≥ 0.98 with another feature | Redundant — adds no extra signal |
| Suspicious names | Matches patterns like `_id`, `future_`, `label`, `target` | Likely an identifier or future-leaking column |

Leakage warnings produce a `Caution` recommendation on their own; combined with harm detection they escalate to `Drop`. Disable all five checks with `--skip-leakage-check`.

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Run tests
pytest

# Run tests with coverage
pytest --cov=treeshaker --cov-report=html
```

### CI

Every push and pull request to `main` runs ruff lint and format checks via GitHub Actions. Merges are blocked if checks fail.

To enable branch protection, go to **GitHub → Settings → Branches → main** and require the `ruff` status check to pass before merging.
