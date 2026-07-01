# TreeShaker

[![Lint](https://github.com/SeanPeppers/TreeShaker/actions/workflows/lint.yml/badge.svg)](https://github.com/SeanPeppers/TreeShaker/actions/workflows/lint.yml)
![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)

Automated feature selection using XGBoost, SHAP, and Optuna. Give it a CSV and a target column — it tunes a model, ranks every feature by importance, and flags the ones that are hurting you.

## Why TreeShaker?

Most feature-importance tools give you a single number per feature from a single model fit, with no way to tell whether that number is signal or noise. TreeShaker's core differentiator is **explainability you can actually trust**, backed by real mechanism rather than a one-shot importance score. Key features include:

- **Genuinely held-out evaluation** — every importance number is computed on a test split the model never sees, not on data it has partially memorized.
- **Stability, not just a point estimate** — permutation importance is averaged over multiple shuffles, and an opt-in bootstrap ensemble shows whether a feature's rank holds up.
- **Plain-English explanations** — each feature gets a Drop/Caution/Keep recommendation with clear reasoning.
- **Cross-checked, not single-signal** — SHAP, permutation importance, mutual information, leakage statistics, and residual correlation are all computed independently. An opt-in importance divergence score (`compute_importance_divergence` / `--compute-importance-divergence`) flags features where these signals disagree about a feature's rank.
- **Calibrated confidence scores** — opt-in with `--compute-confidence-scores` (`compute_confidence_scores=True`) to compute a confidence score (0.5–1.0) for each feature's recommendation. Features clearly above/below threshold score near 1.0, while borderline cases (and "Caution" recommendations) score near 0.5.
- **Skip leakage checks** — opt-out with `--skip-leakage-check` (`skip_leakage_check=True`) to disable all leakage/logical-error detection (the seven checks in `LeakageDetector`, including look-ahead leakage) when it's known to be unnecessary or too slow for a given run. SHAP/permutation importance still run.
- **Extended temporal leakage detection** — three new checks on top of existing look-ahead detection: (1) warns when `time_column` data was not monotonically sorted before pipeline ingestion (lag/rolling features engineered on unsorted data are already corrupt); (2) detects random splits on time-series data by flagging overlapping train/val/test time ranges and recommending a chronological split via `time_column=`; (3) identifies future-peeking rolling window features by recognising the `offset-1` leading NaN signature of an unshifted `shift=0` rolling window and distinguishing it from generic offset mismatches.
- **Shadow probe noise-floor detection** — inject randomly shuffled columns (`--n-shadow-probes` / `n_shadow_probes`) into the dataset before training to establish a noise floor baseline, then flag any real feature whose importance falls below the baseline threshold (percentile configurable via `--shadow-probe-percentile` / `shadow_probe_percentile`).
- **Boruta feature selection** — opt-in feature selection with `--use-boruta` (`use_boruta=True`), adding a `boruta_decision` column (`Confirmed`, `Tentative`, or `Rejected`).
- **RFECV feature selection** — opt-in feature elimination with `--use-rfecv` (`use_rfecv=True`), adding an `rfecv_ranking` column (1 = selected feature rank).
- **LightGBM importance cross-validation** — opt-in model swap with `--use-lightgbm` (`use_lightgbm=True`), adding a `lightgbm_importance` column.
- **CatBoost importance backend** — opt-in model evaluation with native categorical handling using `--use-catboost` (`use_catboost=True`), adding a `catboost_importance` column.
- **Random Forest baseline importance** — opt-in cross-check column with `--use-rf-baseline` (`use_rf_baseline=True`), adding an `rf_importance` column.
- **Sklearn permutation importance** — opt-in evaluation with `--use-sklearn-permutation-importance` (`use_sklearn_permutation_importance=True`), adding an `sklearn_permutation_importance` column computed using scikit-learn's permutation_importance on the existing trained model.
- **GPU acceleration** — opt-in GPU support with `--use-gpu` (`use_gpu=True`), configuring XGBoost models with CUDA execution (`device="cuda"`).
- **Optuna pruning & early stopping** — opt-in trial pruning with `--use-optuna-pruning` (`use_optuna_pruning=True`), using Optuna's MedianPruner and early stopping during hyperparameter search to skip unpromising trials and reduce tuning runtime.
- **Large-dataset auto-stratified sampling** — opt-in sampling with `--large-dataset-sample-threshold` and `--large-dataset-sample-frac`, applying stratified sampling (for classification tasks) prior to train/val/test splitting when row counts exceed the threshold.
- **MLflow experiment tracking** — opt-in run logging with `--use-mlflow` (`use_mlflow=True`), logging Config parameters, per-feature importance/score columns, and the results table as a CSV artifact to MLflow (experiment name configurable via `--mlflow-experiment-name` / `mlflow_experiment_name`). Requires the optional `mlflow` package (`pip install treeshaker[mlflow]`).
- **HTML report** — pass `--format html` (or an `output_path` ending in `.html` via MCP) for a report with a color-coded Keep/Caution/Drop table and an embedded SHAP importance chart, instead of a plain CSV.
- **Custom model passthrough (Python API only)** — pass any pre-instantiated sklearn-API-compatible estimator (`.fit`/`.predict`, classifiers also need `.predict_proba`) via `Config(custom_model=...)` to use it as the primary model instead of XGBoost — e.g. a `RandomForestClassifier`, or an NN wrapped with skorch/scikeras. TreeShaker clones it before fitting and skips Optuna tuning (XGBoost-specific). SHAP importance falls back to the slower model-agnostic `shap.Explainer` for non-tree models, or is skipped with a logged warning if SHAP can't be computed at all. Not available via `--custom-model` CLI flag or MCP (it's a live Python object, not a string-able value) — set it only when constructing `Config` directly in Python.
- **Parallel feature evaluation** — opt-in with `--parallel-feature-evaluation` (`parallel_feature_evaluation=True`), running the per-feature permutation-importance loop across worker processes (`--n-jobs` / `n_jobs`, default -1 = all cores) instead of sequentially. Speeds up evaluation on datasets with many features; results are numerically identical to the sequential path.
- **Multi-seed stability report** — opt-in with `--n-stability-runs N` (`n_stability_runs=N`) to re-run the pipeline N additional times with different random seeds and append `stability_score` (fraction of seeds agreeing with the final recommendation, 0–1) and `stability_votes` (e.g. `Keep:4,Drop:1`) columns to the results. Useful for measuring how stable recommendations are across data splits. Use with `--skip-tuning` to keep total runtime reasonable.
- **DFE ratio operator & categorical interactions** — when `synthesize_from_interactions=True`, synthetic candidates include A/B ratio features (zero-denominator safe) alongside products, and categorical features are target-mean-encoded before interaction construction. Set `dfe_loop_feedback=True` to inject the top-scoring synthetics back into the pipeline; raw constituents dominated by a winning synthetic receive `recommendation=ReplacedBySynthetic`. Cap injections with `dfe_max_inject` (default 5).
- **Streaming / out-of-core data path** — opt-in with `--streaming` (`streaming=True`) for datasets too large to comfortably hold in RAM. Reads the input CSV in `--streaming-chunk-size` (`streaming_chunk_size`, default 100000) row chunks and downcasts numeric dtypes per chunk (int64 → smallest int, float64 → float32) before concatenating, lowering peak memory versus a single full-width read. Off by default; the in-memory path is unchanged. With `streaming=True`, XGBoost training itself runs out-of-core via an external-memory `ExtMemQuantileDMatrix` + a custom `DataIter` that streams on-disk chunks of the train+val data, so the fit never holds the full matrix in RAM (peak training memory is O(chunk), not O(dataset)); the fitted booster is wrapped in a thin sklearn-style adapter so downstream SHAP/permutation/diagnostics are unchanged. The out-of-core training path is CPU-only — XGBoost GPU external memory needs cupy/RAPIDS-resident chunks (not a TreeShaker dependency), so a requested GPU device falls back to CPU for the streaming fit with a warning. (Later stages add sample-based explainability for the streaming case.)
- **Streaming / out-of-core data path** — opt-in with `--streaming` (`streaming=True`) for datasets too large to comfortably hold in RAM. Reads the input CSV in `--streaming-chunk-size` (`streaming_chunk_size`, default 100 000) row chunks and downcasts numeric dtypes per chunk (int64 → smallest int, float64 → float32) before concatenating, lowering peak memory versus a single full-width read. Off by default; the in-memory path is unchanged. With `streaming=True`, XGBoost training itself runs out-of-core via an external-memory `ExtMemQuantileDMatrix` + a custom `DataIter` that streams on-disk pickle chunks, so the fit never holds the full matrix in RAM (peak training memory is O(chunk), not O(dataset)); the fitted booster is wrapped in a thin sklearn-style adapter so downstream SHAP/permutation/diagnostics consume the same API. Post-training analysis (SHAP, permutation importance, mutual information, residual diagnostics, leakage detection) draws a bounded random sample of up to `--streaming-eval-sample-size` (`streaming_eval_sample_size`, default 50 000) rows so those steps stay tractable; order-dependent checks (temporal structure detection) always use the full dataset. Combining `streaming=True` with `custom_model` raises a `ConfigError` at construction time (the XGBoost external-memory path is XGBoost-specific). The out-of-core training path is CPU-only — XGBoost GPU external memory needs cupy/RAPIDS-resident chunks (not a TreeShaker dependency), so a requested GPU device falls back to CPU with a logged warning.

## Quick start

**CLI:**
```bash
pip install treeshaker
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

See [`examples/california_housing_walkthrough.ipynb`](examples/california_housing_walkthrough.ipynb) for an executed, end-to-end walkthrough on a real dataset (with injected leakage/multicollinearity to show detection in action).

## Documentation

This repo is private, so docs aren't hosted (Read the Docs / GitHub Pages free tiers require a public repo). Source docs live in `docs/source/`, including:

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
