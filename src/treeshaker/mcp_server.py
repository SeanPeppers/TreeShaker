"""Local MCP (Model Context Protocol) server wrapping the TreeShaker Python API.

Exposes TreeShaker's pipeline as structured MCP tools so an AI assistant (or
any MCP client) can run feature selection, inspect datasets, read back
results, and discover configuration options without shelling out to the CLI
and parsing CSV/stdout.

This module requires the optional `mcp` dependency:

    pip install -e ".[mcp]"

Run the server (stdio transport, the standard local MCP pattern) with:

    python -m treeshaker.mcp_server

See README.md ("MCP server (experimental)") for how to wire this into an MCP
client such as Claude Code.
"""

import dataclasses
import logging
from dataclasses import fields, is_dataclass
from typing import Any, Literal, get_type_hints

import pandas as pd

from .config import BootstrapConfig, Config, XGBoostConfig
from .dtype_handling import _looks_like_date_strings, _looks_like_free_text
from .exceptions import TreeShakerError
from .pipeline import FeatureSelectionPipeline
from .synthetic_features import SyntheticConfig
from .utils import derive_output_stem

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without extra installed
    raise ImportError(
        "The 'mcp' package is required to run the TreeShaker MCP server."
        ' Install it with: pip install -e ".[mcp]"'
    ) from exc

logger = logging.getLogger(__name__)

mcp = FastMCP("treeshaker")


def _build_config(
    input_path: str,
    target_column: str,
    task: str,
    output_path: str,
    check_residuals: bool = False,
    compute_shap_interactions: bool = False,
    synthesize_features: bool = False,
    synthesis_top_n: int = 5,
    synthesis_mi_threshold: float = 0.01,
    synthesis_operators: list[str] | None = None,
    dfe_loop_feedback: bool = False,
    dfe_max_inject: int = 5,
    n_bootstrap: int = 0,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    skip_tuning: bool = False,
    skip_leakage_check: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 42,
    n_permutations: int = 5,
    permutation_metric: str | None = None,
    shap_sample_size: int | None = None,
    group_by_prefix: bool = False,
    use_boruta: bool = False,
    use_rfecv: bool = False,
    use_lightgbm: bool = False,
    use_catboost: bool = False,
    use_rf_baseline: bool = False,
    use_sklearn_permutation_importance: bool = False,
    use_gpu: bool = False,
    use_optuna_pruning: bool = False,
    large_dataset_sample_threshold: int | None = None,
    large_dataset_sample_frac: float = 0.5,
    use_mlflow: bool = False,
    mlflow_experiment_name: str = "treeshaker",
    parallel_feature_evaluation: bool = False,
    n_jobs: int = -1,
    streaming: bool = False,
    streaming_chunk_size: int = 100_000,
    n_stability_runs: int = 0,
    compute_confidence_scores: bool = False,
    streaming_eval_sample_size: int = 50_000,
    n_shadow_probes: int = 0,
    shadow_probe_percentile: float = 100.0,
    collinearity_threshold: float = 0.0,
    collinearity_method: str = "spearman",
    config_overrides: dict | None = None,
) -> Config:
    """Build a Config from run_pipeline's explicit kwargs plus generic overrides.

    Explicit kwargs map onto the relevant nested sub-config (e.g.
    `synthesize_features` -> `synthetic_config.synthesize_from_interactions`,
    `n_bootstrap` -> `bootstrap_config.n_bootstrap`). Anything in
    `config_overrides` is applied last, directly as top-level Config fields,
    so callers can reach less common fields (e.g. `shap_importance_threshold`)
    without TreeShaker having to hand-enumerate every one.
    """
    output_file = output_path
    if not output_file.lower().endswith((".csv", ".html")):
        output_file = f"{output_path}.csv"
    output_format = "html" if output_file.lower().endswith(".html") else "csv"

    synthetic_config = SyntheticConfig(
        synthesize_from_interactions=synthesize_features,
        interaction_top_n=synthesis_top_n,
        mi_threshold=synthesis_mi_threshold,
        operators=(
            synthesis_operators
            if synthesis_operators is not None
            else ["product", "ratio"]
        ),
        dfe_loop_feedback=dfe_loop_feedback,
        dfe_max_inject=dfe_max_inject,
    )
    if synthesize_features and not compute_shap_interactions:
        # Mirrors the CLI/pipeline contract documented in pipeline.run():
        # synthesize_from_interactions requires compute_shap_interactions.
        # Rather than silently no-op (pipeline.run already warns+skips in
        # that case), turn it on here so the tool call does what it implies.
        compute_shap_interactions = True

    config_kwargs: dict[str, Any] = {
        "input_file": input_path,
        "target_column": target_column,
        "task": task,
        "output_file": output_file,
        "output_format": output_format,
        "check_residuals": check_residuals,
        "compute_shap_interactions": compute_shap_interactions,
        "synthetic_config": synthetic_config,
        "bootstrap_config": BootstrapConfig(n_bootstrap=n_bootstrap),
        "xgboost_config": XGBoostConfig(device=device),
        "skip_tuning": skip_tuning,
        "skip_leakage_check": skip_leakage_check,
        "test_size": test_size,
        "val_size": val_size,
        "random_seed": random_seed,
        "n_permutations": n_permutations,
        "permutation_metric": permutation_metric,
        "shap_sample_size": shap_sample_size,
        "group_by_prefix": group_by_prefix,
        "use_boruta": use_boruta,
        "use_rfecv": use_rfecv,
        "use_lightgbm": use_lightgbm,
        "use_catboost": use_catboost,
        "use_rf_baseline": use_rf_baseline,
        "use_sklearn_permutation_importance": use_sklearn_permutation_importance,
        "use_gpu": use_gpu,
        "use_optuna_pruning": use_optuna_pruning,
        "large_dataset_sample_threshold": large_dataset_sample_threshold,
        "large_dataset_sample_frac": large_dataset_sample_frac,
        "use_mlflow": use_mlflow,
        "mlflow_experiment_name": mlflow_experiment_name,
        "parallel_feature_evaluation": parallel_feature_evaluation,
        "n_jobs": n_jobs,
        "streaming": streaming,
        "streaming_chunk_size": streaming_chunk_size,
        "n_stability_runs": n_stability_runs,
        "compute_confidence_scores": compute_confidence_scores,
        "streaming_eval_sample_size": streaming_eval_sample_size,
        "n_shadow_probes": n_shadow_probes,
        "shadow_probe_percentile": shadow_probe_percentile,
        "collinearity_threshold": collinearity_threshold,
        "collinearity_method": collinearity_method,
    }

    if config_overrides:
        for key, value in config_overrides.items():
            if key not in {f.name for f in fields(Config)}:
                raise ValueError(
                    f"Unknown config_overrides key: {key!r}. See"
                    " list_config_schema() for valid Config field names."
                )
            config_kwargs[key] = value

    return Config(**config_kwargs)


@mcp.tool()
def run_pipeline(
    input_path: str,
    target_column: str,
    task: str,
    output_path: str = "results",
    check_residuals: bool = False,
    compute_shap_interactions: bool = False,
    synthesize_features: bool = False,
    synthesis_top_n: int = 5,
    synthesis_mi_threshold: float = 0.01,
    synthesis_operators: list[str] | None = None,
    dfe_loop_feedback: bool = False,
    dfe_max_inject: int = 5,
    n_bootstrap: int = 0,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    skip_tuning: bool = False,
    skip_leakage_check: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 42,
    n_permutations: int = 5,
    permutation_metric: str | None = None,
    shap_sample_size: int | None = None,
    group_by_prefix: bool = False,
    use_boruta: bool = False,
    use_rfecv: bool = False,
    use_lightgbm: bool = False,
    use_catboost: bool = False,
    use_rf_baseline: bool = False,
    use_sklearn_permutation_importance: bool = False,
    use_gpu: bool = False,
    use_optuna_pruning: bool = False,
    large_dataset_sample_threshold: int | None = None,
    large_dataset_sample_frac: float = 0.5,
    use_mlflow: bool = False,
    mlflow_experiment_name: str = "treeshaker",
    parallel_feature_evaluation: bool = False,
    n_jobs: int = -1,
    streaming: bool = False,
    streaming_chunk_size: int = 100_000,
    n_stability_runs: int = 0,
    compute_confidence_scores: bool = False,
    streaming_eval_sample_size: int = 50_000,
    n_shadow_probes: int = 0,
    shadow_probe_percentile: float = 100.0,
    collinearity_threshold: float = 0.0,
    collinearity_method: str = "spearman",
    config_overrides: dict | None = None,
) -> dict:
    """Run the full TreeShaker feature selection pipeline end to end.

    Args:
        input_path: Path to the input CSV file.
        target_column: Name of the target column.
        task: One of "binary", "multiclass", "regression".
        output_path: Where to write results. A ".csv" extension is appended
            automatically if missing. Use a ".html" extension to instead get
            an HTML report with a color-coded Keep/Caution/Drop table and an
            embedded SHAP importance chart.
        check_residuals: Compute residual-correlation diagnostics (opt-in).
        compute_shap_interactions: Compute pairwise SHAP interaction values
            (opt-in, expensive).
        synthesize_features: Suggest synthetic derived features (pairwise
            products and ratios, plus target-mean-encoded categorical
            interactions) from SHAP interactions (opt-in, advisory by default;
            implies compute_shap_interactions).
        synthesis_top_n: Number of strongest SHAP-interacting pairs to consider
            as synthetic candidates. Only used when synthesize_features is True.
        synthesis_mi_threshold: Minimum mutual information a synthetic candidate
            must exceed to be reported/injected. Only used when
            synthesize_features is True.
        synthesis_operators: Operators to build candidates from — any of
            "product" and "ratio". Defaults to both. Only used when
            synthesize_features is True.
        dfe_loop_feedback: Inject the top synthetic candidates back into the
            pipeline so they compete against the raw features in a re-fit; a raw
            feature whose synthetic derivative achieves strictly higher SHAP
            importance is marked "ReplacedBySynthetic". Requires
            synthesize_features. Off by default (advisory-only).
        dfe_max_inject: Maximum synthetic features to inject when
            dfe_loop_feedback is True (top-scoring by MI). Default 5.
        n_bootstrap: Number of bootstrap resamples for importance stability
            (0 disables).
        device: XGBoost training device: "auto", "cuda", or "cpu".
        skip_tuning: Skip Optuna hyperparameter tuning.
        skip_leakage_check: Skip all leakage/logical-error checks (the seven
            checks in LeakageDetector, including look-ahead leakage). Harm
            detection (SHAP/permutation importance) still runs.
        test_size: Held-out final test fraction.
        val_size: Validation fraction of the remaining train+val pool.
        random_seed: Random seed for reproducibility.
        n_permutations: Number of shuffles per feature for permutation importance.
        permutation_metric: Optional override metric for permutation importance.
        shap_sample_size: Optional row cap for SHAP importance computation.
        group_by_prefix: Auto-derive feature groups from common name prefixes.
        use_boruta: Run Boruta feature selection (opt-in).
        use_rfecv: Run sklearn RFECV feature selection (opt-in).
        use_lightgbm: Compute LightGBM feature importances (opt-in).
        use_catboost: Compute CatBoost feature importances (opt-in).
        use_rf_baseline: Compute Random Forest baseline feature importances (opt-in).
        use_sklearn_permutation_importance: Compute sklearn permutation
            importance on the existing trained model (opt-in).
        use_gpu: Enable GPU acceleration for XGBoost (opt-in).
        use_optuna_pruning: Enable Optuna trial pruning and early stopping (opt-in).
        large_dataset_sample_threshold: Optional row count threshold for sampling.
        large_dataset_sample_frac: Fraction of rows to sample if threshold exceeded.
        use_mlflow: Log this run to MLflow for experiment tracking (opt-in).
            Requires the optional 'mlflow' package.
        mlflow_experiment_name: MLflow experiment name to log under. Only
            used when use_mlflow is True.
        parallel_feature_evaluation: Run the per-feature permutation-importance
            loop across worker processes instead of sequentially (opt-in).
        n_jobs: Worker process count when parallel_feature_evaluation is True
            (-1 = all cores). Ignored otherwise.
        streaming: Opt into the out-of-core / streaming data path for datasets
            too large for RAM. Reads the CSV in chunks and downcasts numeric
            dtypes per chunk to lower peak memory (opt-in, off by default).
        streaming_chunk_size: Rows per chunk when streaming is True. Ignored
            otherwise.
        n_stability_runs: Re-run the pipeline this many times with different
            random seeds and append stability_score / stability_votes columns.
            0 = disabled (default). Use with skip_tuning=True for reasonable
            runtime.
        compute_confidence_scores: Compute calibrated confidence scores (0-1)
            for each feature's recommendation (opt-in).
        streaming_eval_sample_size: Max rows used for the SHAP/permutation/MI/
            leakage/residual diagnostics when streaming is True; larger splits
            are sampled down to this. Ignored otherwise.
        n_shadow_probes: Number of shadow probe columns to inject (0 = disabled,
            suggested 5-20).
        shadow_probe_percentile: Percentile of probe importances to use as noise
            threshold (0-100, default 100.0/max).
        config_overrides: Optional dict of additional Config field overrides
            (e.g. {"shap_importance_threshold": 0.02}) for anything not
            covered by the explicit kwargs above.

    Returns:
        A dict summary: feature_count, recommendation counts (keep/drop/
        caution), top_features (by SHAP importance), leakage/harm warnings
        for dropped/cautioned features, and the output file path(s) written.
        On a known TreeShaker error (bad config, bad input data), returns
        {"error": "<message>"} instead of raising.
    """
    try:
        config = _build_config(
            input_path=input_path,
            target_column=target_column,
            task=task,
            output_path=output_path,
            check_residuals=check_residuals,
            compute_shap_interactions=compute_shap_interactions,
            synthesize_features=synthesize_features,
            synthesis_top_n=synthesis_top_n,
            synthesis_mi_threshold=synthesis_mi_threshold,
            synthesis_operators=synthesis_operators,
            dfe_loop_feedback=dfe_loop_feedback,
            dfe_max_inject=dfe_max_inject,
            n_bootstrap=n_bootstrap,
            device=device,
            skip_tuning=skip_tuning,
            skip_leakage_check=skip_leakage_check,
            test_size=test_size,
            val_size=val_size,
            random_seed=random_seed,
            n_permutations=n_permutations,
            permutation_metric=permutation_metric,
            shap_sample_size=shap_sample_size,
            group_by_prefix=group_by_prefix,
            use_boruta=use_boruta,
            use_rfecv=use_rfecv,
            use_lightgbm=use_lightgbm,
            use_catboost=use_catboost,
            use_rf_baseline=use_rf_baseline,
            use_sklearn_permutation_importance=use_sklearn_permutation_importance,
            use_gpu=use_gpu,
            use_optuna_pruning=use_optuna_pruning,
            large_dataset_sample_threshold=large_dataset_sample_threshold,
            large_dataset_sample_frac=large_dataset_sample_frac,
            use_mlflow=use_mlflow,
            mlflow_experiment_name=mlflow_experiment_name,
            parallel_feature_evaluation=parallel_feature_evaluation,
            n_jobs=n_jobs,
            streaming=streaming,
            streaming_chunk_size=streaming_chunk_size,
            n_stability_runs=n_stability_runs,
            compute_confidence_scores=compute_confidence_scores,
            streaming_eval_sample_size=streaming_eval_sample_size,
            n_shadow_probes=n_shadow_probes,
            shadow_probe_percentile=shadow_probe_percentile,
            config_overrides=config_overrides,
        )

    except (TreeShakerError, ValueError) as exc:
        return {"error": str(exc)}

    try:
        pipeline = FeatureSelectionPipeline(config)
        results = pipeline.run()
    except TreeShakerError as exc:
        return {"error": str(exc)}

    rec_counts = results["recommendation"].value_counts().to_dict()
    top_features = (
        results.sort_values("shap_importance", ascending=False)
        .head(10)[["feature_name", "shap_importance", "recommendation"]]
        .to_dict(orient="records")
    )
    warnings = [
        {
            "feature_name": row["feature_name"],
            "recommendation": row["recommendation"],
            "reason": row["reason"],
        }
        for _, row in results.iterrows()
        if row["recommendation"] in ("Drop", "Caution")
    ]

    output_files = {"results": config.output_file}
    stem = derive_output_stem(config.output_file)
    if pipeline.group_summary is not None:
        output_files["groups"] = f"{stem}_groups.csv"
    if pipeline.interactions is not None:
        output_files["interactions"] = f"{stem}_interactions.csv"
    if pipeline.suggested_features is not None:
        output_files["suggested_features"] = f"{stem}_suggested_features.csv"

    return {
        "feature_count": len(results),
        "keep_count": int(rec_counts.get("Keep", 0)),
        "drop_count": int(rec_counts.get("Drop", 0)),
        "caution_count": int(rec_counts.get("Caution", 0)),
        "top_features": top_features,
        "warnings": warnings,
        "output_files": output_files,
    }


@mcp.tool()
def get_recommendations(output_path: str) -> dict:
    """Read back a previously-written TreeShaker results file as structured JSON.

    Args:
        output_path: Path to a results CSV (or HTML) file previously written
            by run_pipeline / the TreeShaker CLI.

    Returns:
        {"features": [{"feature_name": ..., "recommendation": ..., "reason":
        ..., "shap_importance": ..., ...}, ...]} with one entry per row of
        the results file, preserving all columns present (e.g.
        permutation_importance, mutual_info_score, residual_correlation
        when present). On failure (missing/unreadable file), returns
        {"error": "<message>"}.
    """
    try:
        if output_path.lower().endswith(".html"):
            tables = pd.read_html(output_path)
            if not tables:
                return {"error": f"No tables found in HTML file: {output_path}"}
            df = tables[0]
        else:
            df = pd.read_csv(output_path)
    except FileNotFoundError:
        return {"error": f"Results file not found: {output_path}"}
    except Exception as exc:  # noqa: BLE001 - surface as a clean error, not a traceback
        return {"error": f"Failed to read results file {output_path!r}: {exc}"}

    return {"features": df.to_dict(orient="records")}


@mcp.tool()
def inspect_dataset(input_path: str) -> dict:
    """Load a CSV and report per-column dtype, missing-value rate, and warnings.

    Reuses TreeShaker's own dtype-handling heuristics (dtype_handling.py) so
    the same free-text/date-string/categorical detection used by the actual
    pipeline is reflected here, rather than a separate ad-hoc set of rules.

    Args:
        input_path: Path to the input CSV file.

    Returns:
        {"row_count": int, "column_count": int, "columns": [{"name": ...,
        "dtype": ..., "missing_rate": ..., "n_unique": ..., "suggestion":
        "" | "<preprocessing note>"}, ...]}.
        On failure (missing/unreadable file), returns {"error": "<message>"}.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        return {"error": f"Input file not found: {input_path}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to read CSV {input_path!r}: {exc}"}

    columns = []
    for name in df.columns:
        col = df[name]
        missing_rate = float(col.isna().mean())
        n_unique = int(col.nunique(dropna=True))
        suggestion = ""

        if col.dtype == object or isinstance(col.dtype, pd.StringDtype):
            if _looks_like_date_strings(col) is not None:
                suggestion = (
                    "Looks like a date string column; TreeShaker will"
                    " auto-derive year/month/day/dayofweek numeric columns"
                    " from it."
                )
            elif _looks_like_free_text(col):
                suggestion = (
                    "Looks like free text (mostly-unique, long string"
                    " values); TreeShaker will reject this column unless"
                    " it is pre-vectorized/embedded into numeric columns"
                    " or dropped before running the pipeline."
                )
            else:
                suggestion = (
                    "String column; TreeShaker will cast this to a"
                    " categorical dtype automatically."
                )
        elif pd.api.types.is_datetime64_any_dtype(col):
            suggestion = (
                "Datetime column; TreeShaker will auto-derive"
                " year/month/day/dayofweek numeric columns from it."
            )
        elif missing_rate > 0:
            suggestion = (
                "Numeric column with missing values; XGBoost handles NaN"
                " natively so no preprocessing is required."
            )

        columns.append(
            {
                "name": name,
                "dtype": str(col.dtype),
                "missing_rate": missing_rate,
                "n_unique": n_unique,
                "suggestion": suggestion,
            }
        )

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
    }


def _describe_dataclass_fields(cls: type) -> list[dict]:
    """Introspect a dataclass's fields into name/type/default/doc dicts.

    Nested dataclass fields are recursively expanded under a "fields" key
    so a client can discover the full shape of e.g. Config.optuna_config
    without a separate call.
    """
    try:
        hints = get_type_hints(cls)
    except Exception:  # noqa: BLE001 - fall back to raw annotations if hints fail
        hints = {}

    out = []
    for f in fields(cls):
        type_hint = hints.get(f.name, f.type)
        entry: dict[str, Any] = {
            "name": f.name,
            "type": str(type_hint),
        }
        if f.default is not dataclasses.MISSING:
            entry["default"] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            try:
                default_instance = f.default_factory()
            except Exception:  # noqa: BLE001
                default_instance = None
            is_nested = is_dataclass(default_instance) and not isinstance(
                default_instance, type
            )
            if is_nested:
                entry["default"] = "<see nested fields>"
                entry["fields"] = _describe_dataclass_fields(type(default_instance))
            else:
                entry["default"] = default_instance
        else:
            entry["default"] = None
        out.append(entry)
    return out


@mcp.tool()
def list_config_schema() -> dict:
    """Introspect Config (and nested sub-configs) into a discoverable schema.

    Returns:
        {"fields": [{"name": ..., "type": ..., "default": ..., "fields":
        [...] (only present for nested dataclass fields)}, ...]} describing
        every top-level Config field, with OptunaConfig, XGBoostConfig,
        BootstrapConfig, LeakageConfig, ResidualConfig, and SyntheticConfig
        expanded inline under their respective field's "fields" key.
    """
    return {"fields": _describe_dataclass_fields(Config)}


def main() -> None:
    """Entry point for `python -m treeshaker.mcp_server` / `treeshaker-mcp` script."""
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
