# TreeShaker

[![Lint](https://github.com/SeanPeppers/TreeShaker/actions/workflows/lint.yml/badge.svg)](https://github.com/SeanPeppers/TreeShaker/actions/workflows/lint.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/SeanPeppers/TreeShaker/badge)](https://scorecard.dev/viewer/?uri=github.com/SeanPeppers/TreeShaker)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/15980/badge)](https://www.bestpractices.dev/projects/15980)
![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)

Automated feature selection using XGBoost, SHAP, and Optuna. Give it a CSV and a target column; it tunes a model, ranks every feature by importance, and flags the ones that are hurting you.

## Why TreeShaker?

Most feature-importance tools report one number per feature from one model fit, so you can't tell whether that number is signal or noise. TreeShaker checks each number several ways:

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

## Quick start

**CLI:**
```bash
# Not published to PyPI: install from GitHub. Pin to a release tag
# rather than @main for a stable version.
pip install git+https://github.com/SeanPeppers/TreeShaker.git@v0.1.2
treeshaker --input data.csv --target price --task regression
```


**Python API:**
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

[`examples/california_housing_walkthrough.ipynb`](examples/california_housing_walkthrough.ipynb) is an executed, end-to-end walkthrough on a real dataset, with leakage and multicollinearity injected to show detection in action.

## Documentation

The docs aren't hosted. The source is in `docs/source/`:

- [Installation guide](docs/source/installation.md)
- [How it works](docs/source/how_it_works.md)
- [Feature importance methods](docs/source/importance_methods.md)
- [Leakage detection](docs/source/leakage_detection.md)
- [Configuration reference](docs/source/config_reference.md)
- [CLI reference](docs/source/cli_reference.md)
- [MCP server setup](docs/source/mcp_server.md)

Build a browsable HTML site locally:
```bash
pip install "treeshaker[docs]"
sphinx-build docs/source docs/_build
open docs/_build/index.html
```

## Development

```bash
git clone https://github.com/SeanPeppers/TreeShaker.git
cd TreeShaker
pip install -e ".[dev]"

# Lint, format, type-check, test

ruff check src/ tests/ demo/
ruff format src/ tests/ demo/
mypy
pytest
```

See [Contributing](docs/source/contributing.md) for details.

## License

MIT
