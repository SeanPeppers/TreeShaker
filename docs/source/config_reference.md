# Configuration reference

## Top-level Config

All parameters are optional unless noted. These are settable from the Python API (`Config` dataclass) or via CLI flags (`--flag-name`).

### Data and I/O

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `input_file` | `--input` | str | *required* | Path to input CSV file |
| `target_column` | `--target` | str | *required* | Target column name |
| `task` | `--task` | str | *required* | One of `binary`, `multiclass`, `regression` |
| `output_file` | `--output` | str | `treeshaker_results.csv` | Output file path |
| `output_format` | `--format` | str | `csv` | `csv` or `html` |

### Split configuration

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `test_size` | `--test-size` | float | `0.2` | Fraction of full dataset held out as final test set (0, 1) |
| `val_size` | `--val-size` | float | `0.2` | Fraction of remaining train+val pool used for validation (0, 1) |
| `time_column` | `--time-column` | str or None | `None` | Optional column name for chronological train/val/test splitting |

**Important:** `test_size + val_size` must be < 1 so training data has positive size.

### Feature importance thresholds

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `shap_importance_threshold` | `--shap-threshold` | float | `0.01` | Flag features with mean \|SHAP\| below this |
| `permutation_importance_threshold` | `--perm-threshold` | float | `0.05` | Flag features with permutation importance below this |
| `compute_confidence_scores` | `--compute-confidence-scores` | bool | `False` | Compute a confidence heuristic score (0.5–1.0, not a calibrated probability) for each feature's recommendation |

### Leakage detection

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `leakage_config` | — | `LeakageConfig` | defaults | Nested leakage detection configuration |
| `skip_leakage_check` | `--skip-leakage-check` | bool | `False` | Disable all leakage/logical-error checks |
| `leakage_screening` | `--leakage-screening` | str | `exact` | Algorithm for near-duplicate *numeric* feature screening: `exact` (pairwise correlation, always accurate) or `approximate` (SimHash-based LSH via optional Rust extension, faster on wide data). Does not affect categorical proxy-leakage detection. |

See [LeakageConfig](#leakageconfig) below.

### Feature groups

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `feature_groups` | `--feature-groups` | dict or None | `None` | JSON mapping group name to list of feature names |
| `group_by_prefix` | `--group-by-prefix` | bool | `False` | Auto-derive groups from prefix before last underscore |

When set, each group's member columns are collapsed in the main results
table into one aggregate group row — useful for treating a block of
related columns (e.g. a 768-dim embedding flattened into
`embedding_0`..`embedding_767`) as one unit instead of hundreds of
individually near-meaningless verdicts:
- Group SHAP importance is the sum of member SHAP importances.
- Group permutation importance shuffles every member column together (not
  independently — that's what "remove this group" means for correlated
  columns).
- The group's Keep/Caution/Drop verdict is computed from those aggregate
  values via the same thresholds as any other feature.
- Leakage/temporal warnings that only reference another member of the
  *same* group are dropped as expected/uninteresting; warnings referencing
  anything else still surface on the group row.

Ungrouped columns are unaffected. Every member column must exist among the
dataset's feature columns (raises a config error at run time otherwise). A
separate `<output>_groups.csv` SHAP-sum summary is still written, as
before. Only applies to permutation importance and SHAP-derived columns —
other optional per-backend columns (e.g. `boruta_decision`,
`lightgbm_importance`) are not aggregated to the group level and are
`NaN` on group rows.

Example: `--feature-groups '{"price": ["price_min", "price_max"]}'`

### Permutation importance

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `n_permutations` | `--n-permutations` | int | `5` | Number of shuffles per feature, averaged for stability |
| `permutation_metric` | `--permutation-metric` | str or None | None | Override metric for permutation importance only. One of `roc_auc`, `accuracy`, `neg_mse`, `wape` |

### Hyperparameter tuning (Optuna)

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `skip_tuning` | `--skip-tuning` | bool | `False` | Skip Optuna; train directly with configured XGBoost params |
| `optuna_config` | — | `OptunaConfig` | defaults | Nested Optuna configuration |

See [OptunaConfig](#optunaconfig) below.

### XGBoost configuration

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `xgboost_config` | — | `XGBoostConfig` | defaults | Nested XGBoost configuration |

See [XGBoostConfig](#xgboostconfig) below.

### Bootstrap importance stability (opt-in)

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `bootstrap_config` | — | `BootstrapConfig` | defaults | Nested bootstrap configuration |

See [BootstrapConfig](#bootstrapconfig) below.

### Advanced checks (opt-in)

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `compute_shap_interactions` | `--shap-interactions` | bool | `False` | Compute pairwise SHAP interaction values (expensive) |
| `check_residuals` | `--check-residuals` | bool | `False` | Compute per-feature correlation with model residuals |
| `residual_config` | — | `ResidualConfig` | defaults | Nested residual diagnostics configuration |
| `use_boruta` | `--use-boruta` | bool | `False` | Run Boruta feature selection and output `boruta_decision` |
| `use_rfecv` | `--use-rfecv` | bool | `False` | Run sklearn RFECV feature selection and output `rfecv_ranking` |
| `use_lightgbm` | `--use-lightgbm` | bool | `False` | Compute LightGBM feature importances as `lightgbm_importance` |
| `use_catboost` | `--use-catboost` | bool | `False` | Compute CatBoost feature importances as `catboost_importance` |
| `use_rf_baseline` | `--use-rf-baseline` | bool | `False` | Compute Random Forest baseline importances as `rf_importance` |
| `use_sklearn_permutation_importance` | `--use-sklearn-permutation-importance` | bool | `False` | Compute sklearn permutation importance as `sklearn_permutation_importance` |
| `use_gpu` | `--use-gpu` | bool | `False` | Configure XGBoost to use GPU acceleration (sets `device` to `cuda`) |
| `use_optuna_pruning` | `--use-optuna-pruning` | bool | `False` | Enable Optuna trial pruning and early stopping during tuning |
| `flag_leakage_importance_mismatch` | `--flag-leakage-importance-mismatch` | bool | `False` | Flag features that are high-importance (Keep) but have active leakage warnings |
| `compute_train_importance` | `--compute-train-importance` | bool | `False` | Compute SHAP importance on train+val in addition to test, to detect train-vs-test gaps |
| `train_test_importance_delta_threshold` | `--train-test-delta-threshold` | float or None | None | Flag features whose train-vs-test SHAP importance delta exceeds this (requires `compute_train_importance`) |
| `compute_importance_divergence` | `--compute-importance-divergence` | bool | `False` | Compute a per-feature divergence score across SHAP/permutation/mutual-info rankings |


See [ResidualConfig](#residualconfig) below.

### Feature synthesis (opt-in)

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `synthetic_config` | — | `SyntheticConfig` | defaults | Nested synthetic feature configuration |

See [SyntheticConfig](#syntheticconfig) below.

### Class imbalance

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `imbalance_warning_threshold` | `--imbalance-threshold` | float | `0.1` | Warn if minority class fraction < this value |

### Sampling and SHAP optimization

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `shap_sample_size` | `--shap-sample-size` | int or None | None | Cap SHAP computation to this many rows; useful on large datasets |
| `large_dataset_sample_threshold` | `--large-dataset-sample-threshold` | int or None | None | Row count threshold above which auto-stratified sampling is applied before splitting |
| `large_dataset_sample_frac` | `--large-dataset-sample-frac` | float | `0.5` | Fraction of rows to sample when `large_dataset_sample_threshold` is exceeded, (0, 1] |

### Parallel feature evaluation

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `parallel_feature_evaluation` | `--parallel-feature-evaluation` | bool | `False` | Run the per-feature permutation-importance loop across worker processes via joblib instead of sequentially |
| `n_jobs` | `--n-jobs` | int | `-1` | Worker count when `parallel_feature_evaluation` is True (-1 = use all available cores); ignored otherwise |
| `streaming` | `--streaming` | bool | `False` | Opt into the out-of-core / streaming data path: read the CSV in chunks, downcast numeric dtypes per chunk to reduce peak memory, and train XGBoost via external-memory `QuantileDMatrix` (CPU only). |
| `streaming_chunk_size` | `--streaming-chunk-size` | int | `100000` | Rows per chunk when `streaming` is True; ignored otherwise |
| `n_stability_runs` | `--n-stability-runs` | int | `0` | Re-run the pipeline N times with different random seeds; appends `stability_score` and `stability_votes` columns. 0 = disabled |
| `stability_seeds` | `--stability-seeds` | list[int] | `null` | Explicit seeds for stability runs (length must equal `n_stability_runs`); auto-derived when null |
| `streaming_eval_sample_size` | `--streaming-eval-sample-size` | int | `50000` | Maximum rows used for SHAP, permutation importance, mutual information, residual diagnostics, and leakage detection when `streaming` is True. Rows are drawn as a random sample so post-training analysis stays tractable on large datasets. Order-dependent checks (temporal structure detection) always use the full dataset. Ignored when `streaming` is False. |

### Experiment tracking (opt-in)

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `use_mlflow` | `--use-mlflow` | bool | `False` | Log this run to MLflow for experiment tracking (requires the optional `mlflow` package) |
| `mlflow_experiment_name` | `--mlflow-experiment-name` | str | `treeshaker` | MLflow experiment name to log runs under when `use_mlflow` is True |

### Miscellaneous

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `random_seed` | `--random-seed` | int | `42` | Random seed for reproducibility |
| `verbose` | `--verbose` | bool | `False` | Enable debug logging |
| `custom_model` | — (Python API only) | object or None | None | Pre-instantiated sklearn-API-compatible estimator (must implement `.fit`/`.predict`, and `.predict_proba` for classifiers) to use as the primary model instead of XGBoost. Not exposed as a CLI flag or MCP argument since it is a live Python object, not a serializable value. |

## Nested configurations

### OptunaConfig

Hyperparameter tuning configuration.

```python
from treeshaker import Config, OptunaConfig

config = Config(
    input_file="data.csv",
    target_column="price",
    task="regression",
    optuna_config=OptunaConfig(
        n_trials=100,
        timeout=600,
        seed=42,
        show_progress_bar=False,
    ),
)
```

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `n_trials` | `--optuna-trials` | int | `50` | Number of hyperparameter tuning trials |
| `timeout` | `--optuna-timeout` | int | `300` | Timeout in seconds for the entire tuning run |
| `seed` | — | int | `42` | Random seed for Optuna (usually auto-set from `random_seed`) |
| `show_progress_bar` | — | bool | `False` | Show Optuna progress bar during tuning |

### XGBoostConfig

XGBoost hyperparameter and device configuration.

```python
from treeshaker import Config, XGBoostConfig

config = Config(
    input_file="data.csv",
    target_column="churn",
    task="binary",
    xgboost_config=XGBoostConfig(
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        min_child_weight=1,
        device="auto",
        extra_params={"tree_method": "hist"},
    ),
)
```

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `max_depth` | — | int | `6` | Maximum tree depth |
| `learning_rate` | — | float | `0.1` | Learning rate (eta) |
| `subsample` | — | float | `0.8` | Fraction of rows sampled per tree |
| `colsample_bytree` | — | float | `0.8` | Fraction of features sampled per tree |
| `gamma` | — | float | `0.0` | Minimum loss reduction to split a leaf |
| `min_child_weight` | — | int | `1` | Minimum sum of instance weight in a child |
| `device` | `--device` | str | `auto` | One of `auto`, `cuda`, `cpu` |
| `extra_params` | `--xgboost-params` | dict | `{}` | Additional XGBoost parameters as JSON (CLI) or dict (API) |

### BootstrapConfig

Bootstrap ensemble importance stability configuration.

```python
from treeshaker import Config, BootstrapConfig

config = Config(
    input_file="data.csv",
    target_column="churn",
    task="binary",
    bootstrap_config=BootstrapConfig(
        n_bootstrap=30,
        sample_frac=1.0,
    ),
)
```

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `n_bootstrap` | `--n-bootstrap` | int | `0` | Number of bootstrap refits (0 = disabled) |
| `sample_frac` | `--bootstrap-sample-frac` | float | `1.0` | Fraction of train+val rows per bootstrap sample (0, 1] |
| `shap_importance_stability_threshold` | `--bootstrap-stability-threshold` | float or None | None | When set and bootstrap is enabled, gates recommendations on SHAP importance ranking stability (coefficient of variation); exceeding it demotes Drop/Keep to Caution |

### LeakageConfig

Leakage and logical-error detection configuration.

```python
from treeshaker import Config
from treeshaker.leakage_detection import LeakageConfig

config = Config(
    input_file="data.csv",
    target_column="price",
    task="regression",
    leakage_config=LeakageConfig(
        target_corr_threshold=0.95,
        categorical_assoc_threshold=0.8,
        dominant_shap_fraction=0.7,
        vif_threshold=10.0,
        exempt_temporal_features=True,
        suspicious_name_patterns=["_id", "future_", "label", "target"],
    ),
)
```

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `target_corr_threshold` | `--target-corr-threshold` | float | `0.95` | Flag numeric features with \|correlation to target\| ≥ this |
| `categorical_assoc_threshold` | — | float | `0.8` | Flag non-numeric features with bias-corrected Cramer's V ≥ this |
| `dominant_shap_fraction` | `--dominant-shap-fraction` | float | `0.7` | Flag if one feature accounts for ≥ this fraction of total SHAP |
| `vif_threshold` | — | float | `10.0` | Flag features with Variance Inflation Factor ≥ this |
| `exempt_temporal_features` | — | bool | `True` | Exempt temporal patterns from target-correlation check |
| `temporal_feature_patterns` | — | list[str] | default list | Patterns marking features as temporal (e.g. `lag_`, `rolling_`) |
| `suspicious_name_patterns` | — | list[str] | default list | Patterns marking features as suspicious (e.g. `_id`, `future_`) |

### ResidualConfig

Residual correlation diagnostics configuration.

```python
from treeshaker import Config
from treeshaker.residual_diagnostics import ResidualConfig

config = Config(
    input_file="data.csv",
    target_column="price",
    task="regression",
    check_residuals=True,
    residual_config=ResidualConfig(
        residual_corr_threshold=0.3,
    ),
)
```

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `residual_corr_threshold` | `--residual-corr-threshold` | float | `0.3` | Flag if \|residual correlation\| ≥ this value |

### SyntheticConfig

Synthetic feature suggestion configuration.

```python
from treeshaker import Config
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
```

| Parameter | CLI flag | Type | Default | Description |
|---|---|---|---|---|
| `synthesize_from_interactions` | `--synthesize-features` | bool | `False` | Suggest product features from top SHAP interactions |
| `interaction_top_n` | `--synthesis-top-n` | int | `5` | Number of strongest interaction pairs to consider |
| `mi_threshold` | `--synthesis-mi-threshold` | float | `0.01` | Minimum mutual information for a candidate to be reported |

## Supported data types

TreeShaker validates and normalizes column dtypes automatically before fitting.

| Column type | What happens | Notes |
|---|---|---|
| **Numeric** (int/float) | Works as-is | No transformation |
| **Boolean** | Works as-is | XGBoost and SHAP handle natively |
| **Low-cardinality categorical** | Auto-cast to `category` dtype | XGBoost 2.x fits natively with `enable_categorical=True` |
| **High-cardinality categorical** | Auto-cast to `category` dtype | Cardinality alone doesn't change handling; category sets unified across splits |
| **Datetime** | Auto-derives 4 fields | Extracts year, month, day, dayofweek; removes raw column |
| **Free text** (mostly unique, long strings) | Raises `ValidationError` | Pre-vectorize (TF-IDF, embeddings) before passing to TreeShaker |
| **Missing values** (NaN) | Works as-is | XGBoost learns default split direction; MI uses median imputation |

**How categorical vs. free text is decided:** a string column is treated as free text (and rejected) only if:
- **≥50% unique** values, **AND**
- **≥20 characters** average length

A high-cardinality but short-valued column (5-character ZIP code, UUID) is treated as ordinary categorical.

## Validation

TreeShaker validates configuration values up front and raises `ConfigError` with clear messages for common mistakes:

- Invalid task type (not one of `binary`, `multiclass`, `regression`)
- Invalid split sizes (e.g. `test_size + val_size ≥ 1`)
- Invalid output format (not `csv` or `html`)
- SHAP/permutation thresholds out of valid range
- Invalid permutation metric
- Invalid device (not one of `auto`, `cuda`, `cpu`)
- Forced `device='cuda'` when no usable GPU detected

These exit with exit code `1` and a one-line, actionable error message.

## Feature notes

Behavior details for the opt-in features, with the CLI flag and Python parameter for each.


- **Held-out evaluation:** every importance number is computed on a test split the model never sees, not on data it has partly memorized.
- **Stability:** permutation importance is averaged over multiple shuffles, and an opt-in bootstrap ensemble shows whether a feature's rank holds up.
- **Plain-English explanations:** each feature gets a Drop/Caution/Keep recommendation with its reasoning.
- **Independent cross-checks:** SHAP, permutation importance, mutual information, leakage statistics, and residual correlation are computed separately. An opt-in importance divergence score (`compute_importance_divergence` / `--compute-importance-divergence`) flags features where these signals disagree about rank.
- **Confidence heuristic score:** opt-in with `--compute-confidence-scores` (`compute_confidence_scores=True`). Each recommendation gets a score from 0.5 to 1.0. Features clearly above or below the threshold score near 1.0; borderline cases and "Caution" recommendations score near 0.5. It is a threshold-margin heuristic, not a calibrated probability, so 0.85 does not mean "85% likely correct."
- **Skip leakage checks:** opt out with `--skip-leakage-check` (`skip_leakage_check=True`) to disable all leakage and logical-error detection (the seven checks in `LeakageDetector`, including look-ahead leakage) when it's unnecessary or too slow for a run. SHAP and permutation importance still run.
- **Extended temporal leakage detection:** three checks on top of look-ahead detection. (1) It warns when `time_column` data was not monotonically sorted before pipeline ingestion, because lag and rolling features built on unsorted data are already corrupt. (2) It detects random splits on time-series data by flagging overlapping train/val/test time ranges, and recommends a chronological split via `time_column=`. (3) It identifies future-peeking rolling-window features by the `offset-1` leading NaN signature of an unshifted `shift=0` rolling window, which it tells apart from generic offset mismatches.
- **Shadow probe noise floor:** `--n-shadow-probes` / `n_shadow_probes` injects randomly shuffled columns before training to set a noise-floor baseline, then flags any real feature whose importance falls below the threshold. The percentile is configurable with `--shadow-probe-percentile` / `shadow_probe_percentile`.
- **Boruta feature selection:** opt-in with `--use-boruta` (`use_boruta=True`). Adds a `boruta_decision` column (`Confirmed`, `Tentative`, or `Rejected`).
- **RFECV feature selection:** opt-in with `--use-rfecv` (`use_rfecv=True`). Adds an `rfecv_ranking` column (1 is the selected rank).
- **LightGBM importance cross-validation:** opt-in model swap with `--use-lightgbm` (`use_lightgbm=True`). Adds a `lightgbm_importance` column.
- **CatBoost importance backend:** opt-in model evaluation with native categorical handling, using `--use-catboost` (`use_catboost=True`). Adds a `catboost_importance` column.
- **Random Forest baseline importance:** opt-in cross-check with `--use-rf-baseline` (`use_rf_baseline=True`). Adds an `rf_importance` column.
- **Sklearn permutation importance:** opt-in with `--use-sklearn-permutation-importance` (`use_sklearn_permutation_importance=True`). Adds a `sklearn_permutation_importance` column, computed with scikit-learn's `permutation_importance` on the existing trained model.
- **GPU acceleration:** opt-in with `--use-gpu` (`use_gpu=True`). Configures XGBoost models with CUDA execution (`device="cuda"`).
- **Optuna pruning and early stopping:** opt-in with `--use-optuna-pruning` (`use_optuna_pruning=True`). Uses Optuna's MedianPruner and early stopping during hyperparameter search to skip unpromising trials and shorten tuning.
- **Large-dataset stratified sampling:** opt-in with `--large-dataset-sample-threshold` and `--large-dataset-sample-frac`. When the row count exceeds the threshold, it applies stratified sampling (for classification tasks) before the train/val/test split.
- **MLflow experiment tracking:** opt-in with `--use-mlflow` (`use_mlflow=True`). Logs Config parameters, per-feature importance and score columns, and the results table as a CSV artifact. The experiment name is set with `--mlflow-experiment-name` / `mlflow_experiment_name`. Requires the optional `mlflow` package (`pip install treeshaker[mlflow]`).
- **HTML report:** pass `--format html` (or an `output_path` ending in `.html` via MCP) for a report with a color-coded Keep/Caution/Drop table and an embedded SHAP importance chart, instead of a plain CSV.
- **Custom model passthrough (Python API only):** pass any pre-instantiated, sklearn-API-compatible estimator via `Config(custom_model=...)` to use it instead of XGBoost as the primary model. It needs `.fit` and `.predict`; classifiers also need `.predict_proba`. Examples are a `RandomForestClassifier` or a neural network wrapped with skorch or scikeras. TreeShaker clones it before fitting and skips Optuna tuning, which is XGBoost-specific. SHAP importance falls back to the slower model-agnostic `shap.Explainer` for non-tree models, and is skipped with a logged warning if SHAP can't be computed at all. There is no `--custom-model` CLI flag and no MCP option, because a live Python object can't be passed as a string; set it when constructing `Config` in Python.
- **Parallel feature evaluation:** opt-in with `--parallel-feature-evaluation` (`parallel_feature_evaluation=True`). Runs the per-feature permutation-importance loop across worker processes (`--n-jobs` / `n_jobs`, default -1 for all cores) instead of sequentially. It speeds up datasets with many features, and results are numerically identical to the sequential path.
- **Multi-seed stability report:** opt-in with `--n-stability-runs N` (`n_stability_runs=N`). Re-runs the pipeline N more times with different random seeds and appends two columns to the results: `stability_score` (the fraction of seeds that agree with the final recommendation, 0 to 1) and `stability_votes` (for example `Keep:4,Drop:1`). It measures how stable recommendations are across data splits. Combine it with `--skip-tuning` to keep total runtime reasonable.
- **DFE ratio operator and categorical interactions:** with `synthesize_from_interactions=True`, synthetic candidates include A/B ratio features (safe for zero denominators) alongside products, and categorical features are target-mean-encoded before interactions are built. Set `dfe_loop_feedback=True` to feed the top-scoring synthetics back into the pipeline. Raw constituents dominated by a winning synthetic get `recommendation=ReplacedBySynthetic`, and each winning synthetic is added to the results table as its own `Keep` row with its measured SHAP importance, so the replacement sits next to the features it displaced. Loop feedback is a real competition: each injected synthetic is re-fit alongside the raw features, and a raw constituent is marked `ReplacedBySynthetic` only when its synthetic derivative's SHAP importance strictly exceeds its own. Categorical encoders are fit on train and applied to test, so a synthetic can't win through leakage. Enable it with `--dfe-loop-feedback` (`dfe_loop_feedback=True`) or MCP `dfe_loop_feedback`, cap injections with `--dfe-max-inject` (`dfe_max_inject`, default 5), and choose operators with `--synthesis-operators` (`operators=[...]`). Without loop feedback the path stays advisory: candidates are written to `<output_stem>_suggested_features.csv` and never added automatically.
- **Streaming and out-of-core data path:** opt-in with `--streaming` (`streaming=True`) for datasets too large to hold comfortably in RAM. It reads the input CSV in row chunks of `--streaming-chunk-size` (`streaming_chunk_size`, default 100 000) and downcasts numeric dtypes per chunk (int64 to the smallest int, float64 to float32) before concatenating, which lowers peak memory compared with one full-width read. It is off by default, and the in-memory path is unchanged. With `streaming=True`, XGBoost training itself runs out-of-core through an external-memory `ExtMemQuantileDMatrix` and a custom `DataIter` that streams on-disk pickle chunks, so the fit never holds the full matrix in RAM (peak training memory is O(chunk), not O(dataset)). The fitted booster is wrapped in a thin sklearn-style adapter, so SHAP, permutation importance, and diagnostics use the same API. Post-training analysis (SHAP, permutation importance, mutual information, residual diagnostics, leakage detection) draws a bounded random sample of up to `--streaming-eval-sample-size` (`streaming_eval_sample_size`, default 50 000) rows to stay tractable; order-dependent checks such as temporal structure detection always use the full dataset. Combining `streaming=True` with `custom_model` raises a `ConfigError` at construction time, because the external-memory path is XGBoost-specific. The out-of-core training path is CPU-only: XGBoost GPU external memory needs cupy or RAPIDS-resident chunks, which TreeShaker doesn't depend on, so a requested GPU device falls back to CPU with a logged warning.
