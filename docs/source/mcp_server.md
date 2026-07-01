# MCP server

TreeShaker ships a local [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) server that exposes the pipeline as structured tool calls, so an AI assistant such as Claude Code can run feature selection, inspect a dataset, read back results, and discover configuration options directly — without shelling out to the CLI and parsing CSV/stdout.

**Important:** The MCP server runs locally on your machine; nothing is hosted remotely. It's meant to be launched by an MCP client, not run standalone in a terminal.

## Installation

```bash
pip install "treeshaker[mcp]"
```

This installs the `mcp` package alongside TreeShaker.

## Running the server

Start the MCP server:

```bash
python -m treeshaker.mcp_server
# or, via the installed entry point:
treeshaker-mcp
```

This starts a stdio-transport MCP server — it's meant to be launched by an MCP client, not monitored in a terminal you watch.

## Connecting from Claude Code

Add an entry to your `.mcp.json` (project) or `claude_desktop_config.json` (Claude Desktop):

```json
{
  "mcpServers": {
    "treeshaker": {
      "command": "python",
      "args": ["-m", "treeshaker.mcp_server"]
    }
  }
}
```

If `treeshaker` isn't on the default Python path, point `command` at the interpreter inside your virtualenv (e.g., `.venv/bin/python` on Unix or `.venv\Scripts\python.exe` on Windows):

```json
{
  "mcpServers": {
    "treeshaker": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "treeshaker.mcp_server"]
    }
  }
}
```

Once connected, Claude Code will discover and expose four TreeShaker tools.

## Tools

### run_pipeline

Run the full feature selection pipeline end to end.

**Arguments:**

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `input_path` | string | Yes | — | Path to input CSV file |
| `target_column` | string | Yes | — | Name of target column |
| `task` | string | Yes | — | One of `binary`, `multiclass`, `regression` |
| `output_path` | string | No | `results` | Where to write results (`.csv` appended if missing; use a `.html` path for a color-coded report with an embedded SHAP importance chart) |
| `check_residuals` | boolean | No | `false` | Compute residual-correlation diagnostics |
| `compute_shap_interactions` | boolean | No | `false` | Compute pairwise SHAP interaction values |
| `synthesize_features` | boolean | No | `false` | Suggest product features from SHAP interactions |
| `n_bootstrap` | integer | No | `0` | Number of bootstrap refits (0 = disabled) |
| `device` | string | No | `auto` | One of `auto`, `cuda`, `cpu` |
| `skip_tuning` | boolean | No | `false` | Skip Optuna and train directly |
| `skip_leakage_check` | boolean | No | `false` | Skip all leakage/logical-error checks (harm detection still runs) |
| `test_size` | number | No | `0.2` | Fraction of full dataset for test set |
| `val_size` | number | No | `0.2` | Fraction of remaining pool for validation |
| `random_seed` | integer | No | `42` | Random seed for reproducibility |
| `n_permutations` | integer | No | `5` | Number of shuffles per feature |
| `permutation_metric` | string | No | `null` | Override metric (e.g., `roc_auc`, `accuracy`, `neg_mse`, `wape`) |
| `shap_sample_size` | integer | No | `null` | Cap SHAP to N rows (useful for large datasets) |
| `group_by_prefix` | boolean | No | `false` | Auto-derive feature groups from prefix |
| `use_boruta` | boolean | No | `false` | Run Boruta feature selection and add `boruta_decision` column |
| `use_rfecv` | boolean | No | `false` | Run sklearn RFECV feature selection and add `rfecv_ranking` column |
| `use_lightgbm` | boolean | No | `false` | Compute LightGBM feature importances as `lightgbm_importance` column |
| `use_catboost` | boolean | No | `false` | Compute CatBoost feature importances as `catboost_importance` column |
| `use_rf_baseline` | boolean | No | `false` | Compute Random Forest baseline feature importances as `rf_importance` column |
| `use_sklearn_permutation_importance` | boolean | No | `false` | Compute sklearn permutation importance on the existing trained model |
| `use_gpu` | boolean | No | `false` | Enable GPU acceleration for XGBoost (opt-in) |
| `use_optuna_pruning` | boolean | No | `false` | Enable Optuna trial pruning and early stopping during tuning |
| `large_dataset_sample_threshold` | integer | No | `null` | Row count threshold above which auto-stratified sampling is applied |
| `large_dataset_sample_frac` | number | No | `0.5` | Fraction of rows to sample if `large_dataset_sample_threshold` is exceeded |
| `use_mlflow` | boolean | No | `false` | Log this run to MLflow for experiment tracking (requires the optional `mlflow` package) |
| `mlflow_experiment_name` | string | No | `treeshaker` | MLflow experiment name to log under; only used when `use_mlflow` is `true` |
| `parallel_feature_evaluation` | boolean | No | `false` | Run the per-feature permutation-importance loop across worker processes instead of sequentially |
| `n_jobs` | integer | No | `-1` | Worker thread count when `parallel_feature_evaluation` is `true` (-1 = all cores); ignored otherwise |
| `streaming` | boolean | No | `false` | Opt into the out-of-core / streaming data path: read the CSV in chunks, downcast numeric dtypes, and train XGBoost via external-memory `QuantileDMatrix` (CPU only). |
| `streaming_chunk_size` | integer | No | `100000` | Rows per chunk when `streaming` is `true`; ignored otherwise |
| `n_stability_runs` | integer | No | `0` | Re-run the pipeline N times with different random seeds; appends `stability_score` and `stability_votes` columns. Use with `skip_tuning=true` for reasonable runtime |
| `compute_confidence_scores` | boolean | No | `false` | Compute calibrated confidence scores (0.5–1.0) for each feature's recommendation |
| `streaming_eval_sample_size` | integer | No | `50000` | Maximum rows for SHAP/permutation/MI/residual/leakage analysis when `streaming` is `true`; a random sample is drawn to keep post-training steps tractable. Ignored when `streaming` is `false`. |
| `config_overrides` | object | No | `null` | Additional Config fields as key-value pairs |

**Returns:** A structured summary including:
- `status`: `success` or `error`
- `output_file`: Path to results file
- `feature_count`: Total features in the dataset
- `keep_count`: Features recommended to keep
- `drop_count`: Features recommended to drop
- `caution_count`: Features with warnings
- `top_features_by_shap`: List of top 5 features by SHAP importance
- `warnings`: List of leakage/imbalance warnings (if any)

### inspect_dataset

Report per-column dtype, missing-value rate, and preprocessing suggestions for a CSV without running the full pipeline.

**Arguments:**

| Argument | Type | Required | Description |
|---|---|---|---|
| `input_path` | string | Yes | Path to CSV file |

**Returns:** For each column:
- `column_name`: Column name
- `dtype`: Detected pandas dtype
- `missing_rate`: Fraction of null values
- `unique_count`: Number of unique non-null values
- `suggestion`: Preprocessing note (e.g., "will be converted to datetime", "likely free text — pre-vectorize before passing to TreeShaker")

### get_recommendations

Read back a previously-written results file as structured JSON.

**Arguments:**

| Argument | Type | Required | Description |
|---|---|---|---|
| `output_path` | string | Yes | Path to results CSV or HTML file |

**Returns:** A list of features with columns:
- `feature_name`
- `shap_importance`
- `permutation_importance`
- `permutation_importance_std`
- `mutual_info_score`
- `recommendation` (`Drop`, `Caution`, `Keep`)
- `reason` (explanation if not `Keep`)
- Additional columns if bootstrap or residual diagnostics were enabled

### list_config_schema

Introspect the `Config` dataclass (and nested sub-configs) so a client can discover what's configurable.

**Arguments:** None

**Returns:** A JSON schema describing all Config fields, their types, defaults, and descriptions. Useful for:
- Discovering available configuration options without reading code
- Validating client-provided overrides for `run_pipeline`'s `config_overrides` parameter
- Building UI for configuration

## Example workflows

### Quick pipeline run from Claude Code

```
"Run TreeShaker on my_data.csv with target 'churn' and task 'binary'"
```

Claude Code will call `run_pipeline` with those arguments and return a summary of recommendations.

### Inspect a dataset first

```
"What columns are in data.csv and should I pre-process any of them?"
```

Claude Code will call `inspect_dataset` to report dtypes and missing-value rates.

### Read back results

```
"Show me the recommendations from my previous TreeShaker run that's in results.csv"
```

Claude Code will call `get_recommendations` to parse and return the structured results.

### Discover configuration

```
"What configuration options does TreeShaker have?"
```

Claude Code will call `list_config_schema` to get the full schema and summarize available options.

## Advanced: config_overrides

`run_pipeline` exposes common parameters as top-level kwargs (e.g., `device`, `n_bootstrap`), but not every Config field. Use `config_overrides` to set less-common fields:

```python
run_pipeline(
    input_path="data.csv",
    target_column="price",
    task="regression",
    output_path="results",
    config_overrides={
        "shap_importance_threshold": 0.02,
        "permutation_importance_threshold": 0.08,
        "imbalance_warning_threshold": 0.15,
    }
)
```

This is equivalent to:

```bash
treeshaker --input data.csv --target price --task regression \
  --shap-threshold 0.02 --perm-threshold 0.08 \
  --imbalance-threshold 0.15
```

Valid keys are any field name in the `Config` dataclass. Invalid keys raise a `ValueError`.

## Error handling

All tools return structured errors on failure. Common error cases:

- **File not found:** `inspect_dataset` or `get_recommendations` with a missing file path
- **Invalid CSV:** Empty, malformed, or unreadable file
- **Invalid arguments:** Missing required fields, invalid task type, invalid device
- **Config validation:** Invalid split sizes, conflicting parameters

Each error includes a clear message describing the problem and how to fix it.

## Performance notes

- **`run_pipeline`** — completes in seconds to minutes depending on dataset size, feature count, and tuning settings. Large datasets with `--n-bootstrap` or `--shap-interactions` can be slow.
- **`inspect_dataset`** — very fast; just reads and profiles the CSV once.
- **`get_recommendations`** — fast; just parses the results file.
- **`list_config_schema`** — instant; returns pre-computed schema.

## Limitations

- The server is **single-threaded**. Multiple concurrent requests queue and run sequentially.
- Results files are written to disk. If you run multiple pipelines in parallel (e.g., from different tools calling the same MCP server), use different `output_path` values to avoid conflicts.
- The `inspect_dataset` tool does not support non-CSV formats (e.g., Parquet, JSON).
