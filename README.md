# TreeShaker

[![Lint](https://github.com/SeanPeppers/TreeShaker/actions/workflows/lint.yml/badge.svg)](https://github.com/SeanPeppers/TreeShaker/actions/workflows/lint.yml)
![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)

Automated feature selection using XGBoost, SHAP, and Optuna. Give it a CSV and a target column — it tunes a model, ranks every feature by importance, and flags the ones that are hurting you.

## Why TreeShaker?

- **Fast feature ranking** — SHAP + permutation importance to identify harmful features
- **Leakage detection** — catches suspicious patterns: high target correlation, SHAP dominance, near-duplicates, multicollinearity (VIF), suspicious column names
- **Three output formats** — CSV results, HTML reports, or Python API
- **Automated thresholds** — sensible defaults, fully customizable
- **Plain-English explanations** — each feature gets a Drop/Caution/Keep recommendation with reasoning

## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
  - [CLI](#cli)
  - [Python API](#python-api)
- [Output](#output)
- [CLI reference](#cli-reference)
- [Supported task types](#supported-task-types)
- [Harm detection](#harm-detection)
- [Leakage detection](#leakage-detection)
- [SHAP interaction values](#shap-interaction-values)
- [Class imbalance warning](#class-imbalance-warning)
- [Bootstrap importance stability (opt-in)](#bootstrap-importance-stability-opt-in)
- [Feature-group importance](#feature-group-importance)
- [Development](#development)

## How it works

TreeShaker splits your data three ways: **train**, **validation**, and **test**. `--test-size` (default `0.2`) is held out from the *full* dataset as the final test set; `--val-size` (default `0.2`) is then taken from the *remaining* train+val pool for validation. The test set is never used to fit any model — not during tuning, not in the final fit — so importance numbers reflect genuine out-of-sample behavior rather than performance on data the model has already seen.

1. **Tune** — Optuna searches for the best XGBoost hyperparameters, training on `train` and evaluating each trial on `val` (model selection / early-stopping signal). This step is optional: pass `--skip-tuning` to skip it entirely and train directly with the default or manually-specified (`--xgboost-params`) XGBoost parameters, which is useful for quick runs
2. **Train** — a final XGBoost model is fit on `train` + `val` combined (more data for the production-quality model), using the tuned (or directly configured) parameters. `test` is never included in this fit
3. **Rank** — SHAP values, permutation importance, and mutual information are computed for every feature, evaluated against the held-out `test` set
4. **Flag** — features are marked harmful if their SHAP importance is near zero *or* if shuffling them doesn't degrade the model (i.e., they carry no signal)
5. **Output** — results saved as a ranked CSV or HTML report

> **Behavior change for existing users:** previously TreeShaker used only a train/validation split, and the final model's importance/SHAP/permutation/mutual-information scores were computed on `val_data` — the *same* validation data already included in the final model's training fit. That meant importance was measured on data the model had partially memorized, not genuinely held out, which could inflate importance scores (especially permutation importance and mutual information). The 3-way split fixes this: importance is now computed on a `test` set that is never fit on by any model. If you were relying on `--test-size` to mean "validation fraction," note its meaning has changed to "held-out final test fraction" — use the new `--val-size` flag to control the validation fraction taken from the remaining train+val pool. Expect your numeric importance values to change after upgrading, even with the same `--test-size`.

## Installation

Requires Python 3.13+.

```bash
pip install treeshaker
```

For development:

```bash
git clone https://github.com/SeanPeppers/TreeShaker.git
cd TreeShaker
pip install -e ".[dev]"
```

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

## Output

Results are sorted by SHAP importance (descending).

| Column | Type | Description |
|---|---|---|
| `feature_name` | str | Feature column name |
| `shap_importance` | float | Mean absolute SHAP value across held-out test samples |
| `permutation_importance` | float | Performance drop when the feature is shuffled |
| `mutual_info_score` | float | Model-agnostic mutual information between the feature and the target (cross-check on SHAP/permutation) |
| `recommendation` | str | `Drop`, `Caution`, or `Keep` |
| `reason` | str | Plain-English explanation. Empty if `Keep` |
| `shap_importance_mean` *(optional)* | float | Mean SHAP importance across bootstrap refits. Only present when `--n-bootstrap` > 0 |
| `shap_importance_std` *(optional)* | float | Std dev of SHAP importance across bootstrap refits — higher means a less stable ranking. Only present when `--n-bootstrap` > 0 |

Example output:

```
feature_name  shap_importance  permutation_importance  mutual_info_score  recommendation  reason
feature_3     0.4821           0.312                   0.398               Keep
feature_0     0.3104           0.198                   0.271               Keep
feature_7     0.1823           0.091                   0.155               Keep
feature_2     0.0043           0.009                   0.002               Drop            The model assigns almost no weight to this feature (SHAP: 0.0043)...
feature_9     0.0011           -0.002                  0.000               Drop            The model ignores this feature entirely...
customer_id   0.0512           0.041                   0.061               Caution         Column name 'customer_id' looks like a row identifier...
```

`recommendation` logic:

- **Drop** — harmful (low SHAP and/or low permutation importance)
- **Caution** — leakage/logical warning only, model is still using the feature
- **Keep** — no issues found
- If both harmful *and* a leakage warning fire, the feature is still `Drop`, with both reasons combined.

## CLI reference

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | Yes | — | Path to input CSV file |
| `--target` | Yes | — | Target column name |
| `--task` | Yes | — | `binary`, `multiclass`, or `regression` |
| `--output` | No | `treeshaker_results.csv` | Output file path |
| `--format` | No | `csv` | `csv` or `html` |
| `--xgboost-params` | No | `{}` | Custom XGBoost params as a JSON string |
| `--feature-groups` | No | `{}` | Feature groups as a JSON string mapping group name to a list of feature names, e.g. `'{"price": ["price_min", "price_max"]}'` |
| `--group-by-prefix` | No | `False` | Auto-derive feature groups from the common prefix before the last underscore (e.g. `price_min`/`price_max` -> group `price`). Ignored if `--feature-groups` is set |
| `--optuna-trials` | No | `50` | Number of Optuna tuning trials |
| `--optuna-timeout` | No | `300` | Optuna timeout in seconds |
| `--shap-threshold` | No | `0.01` | Flag features with mean \|SHAP\| below this |
| `--perm-threshold` | No | `0.05` | Flag features with permutation importance below this |
| `--target-corr-threshold` | No | `0.95` | Flag features whose correlation with the target exceeds this |
| `--dominant-shap-fraction` | No | `0.7` | Flag a feature if it accounts for more than this fraction of total SHAP importance |
| `--skip-leakage-check` | No | `False` | Disable all leakage/logical-error checks |
| `--skip-tuning` | No | `False` | Skip Optuna hyperparameter tuning and train the final model directly using the configured (default or `--xgboost-params`) XGBoost parameters |
| `--shap-interactions` | No | `False` | Compute pairwise SHAP interaction values (opt-in, expensive) and save them to `<output_stem>_interactions.csv` |
| `--imbalance-threshold` | No | `0.1` | Warn if the minority class fraction (binary/multiclass) falls below this |
| `--n-bootstrap` | No | `0` | Number of bootstrap resamples to refit and assess SHAP importance ranking stability. `0` disables this feature |
| `--bootstrap-sample-frac` | No | `1.0` | Fraction of combined train+val rows sampled (with replacement) per bootstrap iteration |
| `--test-size` | No | `0.2` | Fraction of the *full* dataset held out as the final test set. Never used to fit any model; used for importance evaluation and leakage statistics |
| `--val-size` | No | `0.2` | Fraction of the *remaining* train+val pool (after `--test-size` is removed) used for validation during Optuna tuning |
| `--random-seed` | No | `42` | Random seed for reproducibility |
| `--verbose` | No | `False` | Enable debug logging |

---

## Error handling

TreeShaker validates common input mistakes up front and reports them in
plain English instead of a raw traceback — for example, a missing input
file, an empty or malformed CSV, a target column with no non-null values,
or a `--test-size`/`--val-size` too small to stratify the test or
validation split. All of these exit with code `1` and a one-line,
actionable error message.

## Supported task types

| Task | Target column | Evaluation metric |
|---|---|---|
| `binary` | 2 unique values | ROC-AUC |
| `multiclass` | 3+ unique values | Accuracy |
| `regression` | Numeric | Negative MSE |

## Harm detection

A feature is flagged as harmful if **either** condition is true:

- **Low SHAP** — mean absolute SHAP value < `--shap-threshold` (default `0.01`). The feature contributes near-zero signal to the model's predictions.
- **Low permutation importance** — permutation importance < `--perm-threshold` (default `0.05`). Shuffling the feature doesn't hurt performance, meaning the model isn't using it.

Both thresholds are configurable. Raise them to be more aggressive about flagging features; lower them to only flag the most clearly useless ones.

## Leakage detection

Six independent checks flag features that may be causing data leakage or are otherwise logically suspect. A feature can trigger more than one.

| Check | Trigger | Why it matters |
|---|---|---|
| Target correlation | `\|r\|` with target ≥ `--target-corr-threshold` (default `0.95`) | Feature may have been derived from the target |
| SHAP dominance | One feature accounts for ≥ `--dominant-shap-fraction` (default `0.7`) of total SHAP | A single feature this dominant may leak future/target info |
| Near-constant | Same value in > 99% of rows | No signal, wastes model capacity |
| Near-duplicate | Correlation ≥ 0.98 with another feature | Redundant — adds no extra signal |
| Multicollinearity (VIF) | Variance Inflation Factor ≥ `vif_threshold` (default `10.0`) | Feature is a near-linear combination of *other* features — redundancy pairwise correlation alone can miss |
| Suspicious names | Matches patterns like `_id`, `future_`, `label`, `target` | Likely an identifier or future-leaking column |

Leakage warnings produce a `Caution` recommendation on their own; combined with harm detection they escalate to `Drop`. Disable all six checks with `--skip-leakage-check`.

Unlike importance evaluation (SHAP, permutation importance, mutual information — all computed on the held-out `test` set, see [How it works](#how-it-works)), leakage detection is a statistical property check across the *entire* dataset, not a held-out-generalization concern. So these checks intentionally run against `train` + `val` + `test` combined, to see the full dataset's correlation/VIF/duplicate statistics.

## Class imbalance warning

Under extreme class imbalance (e.g. a 99%/1% binary target), accuracy and even ROC-AUC can barely move regardless of which feature is shuffled — making permutation importance unreliable. Features can look unimportant simply because the metric is insensitive to what's being shuffled.

TreeShaker checks the target distribution for binary and multiclass tasks (not applicable to regression) after loading data. If the minority class makes up less than `--imbalance-threshold` (default `0.1`, i.e. 10%) of the target, a plain-English warning is logged. This is visibility only — it does not block the pipeline or change the output schema. If you see this warning, consider interpreting permutation importance cautiously, or evaluating with a metric more sensitive to the minority class (e.g. PR-AUC or F1).

## SHAP interaction values

Plain SHAP importance ranks each feature independently. It can miss cases where two features are individually weak but jointly carry a lot of signal — a non-linear interaction.

Pass `--shap-interactions` (or set `compute_shap_interactions=True` in `Config`) to additionally compute pairwise SHAP interaction values via `shap.TreeExplainer.shap_interaction_values`. This is **opt-in and off by default** because it's substantially more expensive than ordinary SHAP — roughly O(features²) per sample.

When enabled:

- The main `results` DataFrame returned by `pipeline.run()` is **unchanged** — same columns, same shape.
- A second file, `<output_stem>_interactions.csv`, is written alongside the main output. For `treeshaker_results.csv` this is `treeshaker_results_interactions.csv`.
- The interactions file has columns `feature_a`, `feature_b`, `interaction_strength`, sorted by `interaction_strength` descending, limited to the top 10 strongest pairs.
- The same data is also available on `pipeline.interactions` after calling `run()` from the Python API.

```bash
treeshaker --input data.csv --target churn --task binary --shap-interactions
```

```python
from treeshaker import FeatureSelectionPipeline, Config

config = Config(
    input_file="data.csv",
    target_column="churn",
    task="binary",
    compute_shap_interactions=True,
)
pipeline = FeatureSelectionPipeline(config)
results = pipeline.run()
print(pipeline.interactions)
```

## Bootstrap importance stability (opt-in)

TreeShaker normally fits a single final XGBoost model and reports its SHAP and permutation importance. A single fit's importance ranking can be noisy, especially on smaller datasets — a feature ranked 3rd might just as easily rank 6th on a slightly different sample.

Set `--n-bootstrap` to a positive integer to additionally refit the model `n` times on bootstrap resamples (sampled with replacement from the combined train+validation data) using the **same** tuned hyperparameters Optuna already found — Optuna is not re-run per iteration, since that would be far too expensive for what this feature is checking. Each bootstrap refit recomputes SHAP importance only, evaluated on the held-out `test` set; permutation importance is skipped per iteration since it requires a full extra prediction pass per feature, which would make the cost scale with both `n_bootstrap` and feature count. SHAP alone is enough to assess ranking stability.

```bash
treeshaker --input data.csv --target churn --task binary --n-bootstrap 30
```

This adds two columns to the output: `shap_importance_mean` and `shap_importance_std`, the mean and standard deviation of each feature's SHAP importance across the `n_bootstrap` refits. A high `shap_importance_std` relative to `shap_importance_mean` signals a feature whose importance ranking isn't stable and shouldn't be trusted from the single main fit alone.

By default `n_bootstrap` is `0`, so this is fully opt-in and the output schema is unchanged unless you ask for it.

## Feature-group importance

Real datasets often have related features that share a prefix (e.g. `price_min`, `price_max`, `price_avg` all belong to a "price" group). Beyond per-feature recommendations, TreeShaker can optionally aggregate SHAP importance per group, so you can decide whether to drop an entire group at once.

This is off by default and has two ways to turn it on:

1. **Explicit groups** — pass a mapping of group name to feature names, via `feature_groups` on `Config` or `--feature-groups` on the CLI:

   ```python
   from treeshaker import FeatureSelectionPipeline, Config

   config = Config(
       input_file="data.csv",
       target_column="price",
       task="regression",
       feature_groups={"price": ["price_min", "price_max", "price_avg"]},
   )
   pipeline = FeatureSelectionPipeline(config)
   results = pipeline.run()
   print(pipeline.group_summary)
   ```

   ```bash
   treeshaker --input data.csv --target target --task regression \
     --feature-groups '{"price": ["price_min", "price_max", "price_avg"]}'
   ```

2. **Automatic prefix grouping** — set `group_by_prefix=True` on `Config` (or pass `--group-by-prefix` on the CLI) to auto-derive groups from the common prefix before the last underscore in each feature name (e.g. `price_min`/`price_max` -> group `"price"`). Features with no underscore, or whose prefix only matches one feature, aren't grouped. Ignored if `feature_groups` is set explicitly.

When grouping is active (either way), the per-feature `results` DataFrame returned by `run()` is unchanged. A separate group summary is computed and:

- exposed as `pipeline.group_summary` (a `pandas.DataFrame`)
- saved alongside the main output as `<output_stem>_groups.csv` (e.g. `treeshaker_results_groups.csv`)

`group_summary` columns:

| Column | Type | Description |
|---|---|---|
| `group_name` | str | Group name |
| `n_features` | int | Number of features in the group found in the dataset |
| `total_shap_importance` | float | **Sum** of `shap_importance` across the group's features |
| `features` | str | Comma-separated list of the group's feature names |

Importance is **summed**, not averaged, per group — the sum reads as "total model attention this group accounts for," which is the more interpretable number when deciding whether to drop an entire group.

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

### Dependency pinning

This project uses [`uv`](https://docs.astral.sh/uv/) to pin exact dependency versions in `uv.lock`, so local installs and CI use the same resolved versions instead of re-resolving "latest" on every run. `uv.lock` is committed and should be regenerated whenever `pyproject.toml` dependencies change:

```bash
uv lock
```

Commit the updated `uv.lock` alongside the `pyproject.toml` change.

### CI

Every push and pull request to `main` runs ruff lint and format checks via GitHub Actions. Merges are blocked if checks fail.

To enable branch protection, go to **GitHub → Settings → Branches → main** and require the `ruff` status check to pass before merging.
