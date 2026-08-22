"""Utility functions for TreeShaker."""

import logging
import os
import re
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from .exceptions import ValidationError

#: Named metric overrides supported by `--permutation-metric` /
#: Config.permutation_metric.
VALID_PERMUTATION_METRICS = {"roc_auc", "accuracy", "neg_mse", "wape"}


@contextmanager
def ignore_constant_corr_warnings() -> Iterator[None]:
    """Silence numpy's divide-by-zero RuntimeWarning from correlating a
    zero-variance (constant) column.

    A constant column has zero standard deviation, so Pearson/Spearman
    correlation divides by zero and numpy emits an ``invalid value``/``divide``
    RuntimeWarning while returning NaN. Every ``.corr()`` caller in TreeShaker
    already treats that NaN as "no correlation" (``pd.isna`` guards, or
    ``NaN >= threshold`` evaluating False), so the warning is pure console
    noise — common on messy real datasets that carry constant columns. Wrap
    the correlation call in this context manager to suppress it without
    changing any result.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        yield


@lru_cache(maxsize=1)
def detect_cuda_available() -> bool:
    """Detect whether XGBoost can actually train on a CUDA GPU right now.

    This does more than check whether the installed xgboost build was
    compiled with CUDA support -- it attempts a trivial real fit with
    ``device="cuda"`` on a tiny synthetic dataset, since that's the only
    reliable way to know a usable GPU is actually present and the CUDA
    runtime is correctly set up. The result is cached for the lifetime of
    the process (via lru_cache) since each check involves XGBoost booster
    setup and would otherwise be re-run for every fold/fit, which is
    expensive for something that can't change mid-run.

    Returns:
        True if a CUDA device is available and a trivial XGBoost fit with
        device="cuda" succeeds; False otherwise (no GPU, no CUDA-enabled
        build, driver issue, etc).
    """
    try:
        import xgboost as xgb

        X = np.zeros((4, 2), dtype=np.float32)
        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        model = xgb.XGBRegressor(
            n_estimators=1, max_depth=1, device="cuda", verbosity=0
        )
        model.fit(X, y)
        return True
    except Exception:
        return False


def resolve_xgboost_device(device: str, use_gpu: bool = False) -> str:
    """Resolve a configured device setting to the literal value passed to XGBoost.

    Args:
        device: One of "auto", "cuda", "cpu" (see XGBoostConfig.device).
        use_gpu: If True, resolves to "cuda" regardless of device setting.

    Returns:
        The resolved device string ("cuda" or "cpu") to pass as XGBoost's
        `device` parameter.

    Raises:
        ValueError: If device is not one of "auto", "cuda", or "cpu".
    """
    if use_gpu:
        return "cuda"
    if device == "auto":
        return "cuda" if detect_cuda_available() else "cpu"
    if device in ("cuda", "cpu"):
        return device
    raise ValueError(f"device must be one of 'auto', 'cuda', 'cpu', got {device!r}")


def validate_target(df: pd.DataFrame, target_column: str, task: str) -> None:
    """Validate that the target column exists and is compatible with the task.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        task: One of "binary", "multiclass", or "regression".

    Raises:
        ValidationError: If the target column is missing or incompatible with
            the task.
    """
    if target_column not in df.columns:
        raise ValidationError(f"Target column {target_column!r} not found in data")
    n_unique = df[target_column].nunique()
    if task == "binary" and n_unique != 2:
        raise ValidationError(
            f"Binary task requires exactly 2 unique target values, got {n_unique}"
        )
    if task == "multiclass" and n_unique < 3:
        raise ValidationError(
            f"Multiclass task requires at least 3 unique target values, got {n_unique}"
        )
    if task == "regression":
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValidationError("Regression task requires a numeric target column")


def check_class_imbalance(
    df: pd.DataFrame,
    target_column: str,
    task: str,
    imbalance_warning_threshold: float = 0.1,
) -> str | None:
    """Check for severe class imbalance that could undermine permutation importance.

    Under extreme class imbalance, metrics like ROC-AUC and accuracy can stay
    nearly flat regardless of which feature is shuffled, which makes
    permutation importance unreliable — features can look unimportant simply
    because the metric barely moves. This check only applies to binary and
    multiclass tasks; regression has no notion of class balance.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        task: One of "binary", "multiclass", or "regression".
        imbalance_warning_threshold: Minority class fraction below which a
            warning is returned. Default 0.1 (10%).

    Returns:
        A plain-English warning string if the minority class fraction is
        below imbalance_warning_threshold, otherwise None. Always None for
        the "regression" task.
    """
    if task not in ("binary", "multiclass"):
        return None

    class_counts = df[target_column].value_counts(normalize=True)
    if class_counts.empty:
        return None

    minority_fraction = class_counts.min()
    if minority_fraction >= imbalance_warning_threshold:
        return None

    minority_class = class_counts.idxmin()
    pct = minority_fraction * 100
    return (
        f"Severe class imbalance detected: the minority class"
        f" ({minority_class!r}) makes up only {pct:.1f}% of the target,"
        f" below the {imbalance_warning_threshold * 100:.0f}% warning"
        f" threshold. Under this level of imbalance, accuracy and even"
        f" ROC-AUC can barely move regardless of which feature is shuffled,"
        f" which makes permutation importance unreliable — features may"
        f" look unimportant simply because the metric is insensitive."
        f" Consider using a metric more sensitive to the minority class"
        f" (e.g. PR-AUC or F1) when interpreting these results."
    )


def _split_chronological(
    df: pd.DataFrame,
    time_column: str,
    test_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically by a time column (ascending order).

    Sorts by time_column ascending, then splits using the same proportions
    as random splitting (test_size from the full set, val_size from the
    remaining train+val pool), but preserving temporal order.

    Args:
        df: DataFrame to split (must contain time_column).
        time_column: Column name to sort by (must contain sortable values).
        test_size: Fraction of the FULL dataset held out as the final test set.
        val_size: Fraction of the remaining train+val pool for validation.

    Returns:
        A tuple of (train_df, val_df, test_df) in chronological order.

    Raises:
        ValidationError: If time_column is missing or contains non-sortable values.
    """
    if time_column not in df.columns:
        raise ValidationError(
            f"time_column {time_column!r} not found in data."
            f" Available columns: {sorted(df.columns)}"
        )

    # Warn if the raw data was not already sorted — lag/rolling features
    # computed on unsorted data before calling TreeShaker are corrupted.
    if not df[time_column].is_monotonic_increasing:
        logging.warning(
            f"time_column '{time_column}' is not monotonically sorted in the"
            " input data. If temporal features (lags, rolling windows) were"
            " engineered on the unsorted data before this pipeline run, those"
            " features may contain future information and should be"
            " recomputed on sorted data."
        )

    # Sort by time ascending
    try:
        df_sorted = df.sort_values(by=time_column).reset_index(drop=True)
    except TypeError as exc:
        raise ValidationError(
            f"time_column {time_column!r} contains non-sortable values: {exc}"
        ) from exc

    n = len(df_sorted)
    test_count = int(np.round(n * test_size))
    # val_size is a fraction of the train+val pool (after removing test)
    train_val_count = n - test_count
    val_count = int(np.round(train_val_count * val_size))

    # Chronological order: earliest train, then middle val, then latest test
    test_df = df_sorted.iloc[n - test_count :].reset_index(drop=True)
    train_val = df_sorted.iloc[: n - test_count]
    val_df = train_val.iloc[len(train_val) - val_count :].reset_index(drop=True)
    train_df = train_val.iloc[: len(train_val) - val_count].reset_index(drop=True)

    return train_df, val_df, test_df


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to their smallest safe dtype.

    Integer columns are downcast to the smallest signed integer dtype that
    holds their range; float columns are narrowed (float64 -> float32 where
    pandas deems it safe). Non-numeric columns are left untouched. This is a
    pure memory-reduction step -- integer values are preserved exactly, and
    floats are narrowed only via `pd.to_numeric(downcast=...)`.

    Used by the streaming chunked-read path so each chunk is narrowed before
    concatenation, lowering peak memory versus loading at default
    int64/float64 width.
    """
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif pd.api.types.is_float_dtype(dtype):
            out[col] = pd.to_numeric(out[col], downcast="float")
    return out


def _read_csv_streaming(filepath: str, chunk_size: int) -> pd.DataFrame:
    """Read a CSV in row chunks, downcasting numeric dtypes per chunk.

    This is the Stage 1 memory-reduction path for `streaming=True`: rather
    than `pd.read_csv` materializing the whole file at default int64/float64
    width, the file is streamed in `chunk_size`-row pieces, each narrowed via
    `_downcast_numeric`, then concatenated. The returned DataFrame is
    identical in contents to `pd.read_csv(filepath)` modulo numeric dtype
    width.

    Note: this still produces a single in-memory DataFrame at the end. True
    out-of-core *training* (never materializing the full frame) is layered on
    in later stages via XGBoost's external-memory DataIter; this stage's value
    is the dtype downcast, which helps datasets that fit in RAM only after
    narrowing.
    """
    chunks = [
        _downcast_numeric(chunk)
        for chunk in pd.read_csv(filepath, chunksize=chunk_size)
    ]
    if not chunks:
        # Mirror pd.read_csv's EmptyDataError contract for an empty file.
        raise pd.errors.EmptyDataError("No columns to parse from file")
    return pd.concat(chunks, ignore_index=True)


def load_csv(
    filepath: str,
    target_column: str,
    task: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 42,
    time_column: str | None = None,
    large_dataset_sample_threshold: int | None = None,
    large_dataset_sample_frac: float = 0.5,
    streaming: bool = False,
    streaming_chunk_size: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a CSV file and split it into train, validation, and test sets.

    The split is done in two sequential steps (the standard sklearn pattern
    for a 3-way split): first test_size is held out from the full dataset as
    the final, never-fit-on test set; then val_size (a fraction of the
    *remaining* train+val pool) is held out from what's left as the
    validation set used for Optuna trial evaluation / early stopping.

    When time_column is set, the split is chronological (sorted ascending by
    time_column) rather than random, preserving the same proportions.

    Args:
        filepath: Path to the input CSV file.
        target_column: Name of the target column.
        task: One of "binary", "multiclass", or "regression".
        test_size: Fraction of the FULL dataset held out as the final test
            set. This set is never used for fitting any model.
        val_size: Fraction of the remaining train+val pool (i.e. after
            test_size is removed) used for validation during hyperparameter
            tuning.
        random_seed: Random seed for reproducible splits (ignored if
            time_column is set).
        time_column: Optional column name for chronological splitting. When
            set, data is sorted by this column and split chronologically
            rather than randomly. When None (default), splits are random.
        large_dataset_sample_threshold: Optional row count threshold for sampling.
        large_dataset_sample_frac: Fraction of rows to sample if threshold exceeded.
        streaming: If True, read the CSV in `streaming_chunk_size` row chunks
            and downcast numeric dtypes per chunk before concatenating,
            lowering peak memory versus a single full-width read. The split
            logic is unchanged. When False (default), the file is read in one
            `pd.read_csv` call exactly as before.
        streaming_chunk_size: Rows per chunk when streaming=True. Ignored
            otherwise.

    Returns:
        A tuple of (train_df, val_df, test_df), all including the target
        column.

    Raises:
        ValidationError: If the input file is missing, empty, unreadable as
            CSV, the target column is missing or incompatible with the task,
            the target column has no non-null values, or there are too few
            rows to perform a stratified split. If time_column is set and
            the column is missing or contains non-sortable values.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {filepath}")

    if not Path(filepath).exists():
        raise ValidationError(
            f"Input file not found: '{filepath}'. Check the path and try again."
        )

    try:
        if streaming:
            logger.info(
                f"Streaming read enabled: reading in chunks of"
                f" {streaming_chunk_size} rows with per-chunk numeric downcast"
            )
            df = _read_csv_streaming(filepath, streaming_chunk_size)
        else:
            df = pd.read_csv(filepath)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValidationError(
            f"Input file '{filepath}' is empty or not valid CSV."
        ) from exc

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    validate_target(df, target_column, task)
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValidationError(
            f"Target column {target_column!r} has no non-null values —"
            " nothing to train on."
        )

    if (
        large_dataset_sample_threshold is not None
        and len(df) > large_dataset_sample_threshold
    ):
        logger.info(
            f"Input row count ({len(df)}) exceeds threshold"
            f" ({large_dataset_sample_threshold}); applying auto-stratified sampling"
            f" with fraction {large_dataset_sample_frac}"
        )
        stratify_sample = (
            df[target_column] if task in ("binary", "multiclass") else None
        )
        try:
            df, _ = train_test_split(
                df,
                train_size=large_dataset_sample_frac,
                random_state=random_seed,
                stratify=stratify_sample,
            )
            df = df.reset_index(drop=True)
        except ValueError as exc:
            if stratify_sample is not None:
                raise ValidationError(
                    "Not enough data to perform stratified large-dataset sampling: "
                    f"{exc}"
                ) from exc
            raise

    if time_column is not None:
        # Chronological split
        logger.info(f"Using chronological split by {time_column!r}")
        train_df, val_df, test_df = _split_chronological(
            df, time_column, test_size, val_size
        )
    else:
        # Random split (existing behavior)
        stratify = df[target_column] if task in ("binary", "multiclass") else None
        try:
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_seed,
                stratify=stratify,
            )
        except ValueError as exc:
            if stratify is not None:
                raise ValidationError(
                    "Not enough data to perform a stratified train/test"
                    f" split with test_size={test_size}: {exc} Try providing"
                    " more data or adjusting --test-size."
                ) from exc
            raise

        stratify_train_val = (
            train_val_df[target_column] if task in ("binary", "multiclass") else None
        )
        try:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size,
                random_state=random_seed,
                stratify=stratify_train_val,
            )
        except ValueError as exc:
            if stratify_train_val is not None:
                raise ValidationError(
                    "Not enough data to perform a stratified train/validation"
                    f" split with val_size={val_size}: {exc} Try providing"
                    " more data or adjusting --val-size."
                ) from exc
            raise

    logger.info(
        f"Split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test"
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Weighted Absolute Percentage Error (WAPE).

    WAPE = sum(abs(y_true - y_pred)) / sum(abs(y_true))

    This is the standard business metric for volume/demand forecasting,
    as opposed to MSE which penalizes large errors quadratically and is
    less directly interpretable in business terms ("we were off by X% of
    total volume").

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        WAPE as a float. 0 means a perfect forecast; values are
        non-negative and unbounded above. Lower is better (unlike most
        metrics in this module, WAPE is NOT "higher is better" --
        get_metric_fn negates it for that reason).

    Raises:
        ValidationError: If sum(abs(y_true)) is 0, since WAPE is undefined
            (division by zero) when total actual volume is zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total_actual = np.abs(y_true).sum()
    if total_actual == 0:
        raise ValidationError(
            "WAPE is undefined when sum(abs(y_true)) == 0 (no actual volume"
            " to compare against). Use a different permutation metric for"
            " this target."
        )
    return float(np.abs(y_true - y_pred).sum() / total_actual)


def get_metric_fn(task: str, override: str | None = None):
    """Return the evaluation metric function for the given task.

    The returned function has the signature (y_true, y_pred) -> float,
    where higher values indicate better performance.

    Args:
        task: One of "binary", "multiclass", or "regression".
        override: Optional explicit metric name to use instead of the
            automatic per-task default. One of "roc_auc", "accuracy",
            "neg_mse", or "wape". If None (the default), the automatic
            per-task behavior is preserved exactly as before: ROC-AUC for
            binary, accuracy for multiclass, negative MSE for regression.

    Returns:
        A callable metric function.

    Raises:
        ValueError: If the task is not recognized, or override is not a
            recognized metric name.
    """
    if override is not None:
        if override == "roc_auc":
            return lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
        if override == "accuracy":
            return lambda y_true, y_pred: accuracy_score(y_true, y_pred)
        if override == "neg_mse":
            return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        if override == "wape":
            return lambda y_true, y_pred: -compute_wape(y_true, y_pred)
        raise ValueError(
            f"Unknown permutation metric override: {override!r}. Valid"
            f" options are: {sorted(VALID_PERMUTATION_METRICS)}"
        )

    if task == "binary":
        return lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
    if task == "multiclass":
        return lambda y_true, y_pred: accuracy_score(y_true, y_pred)
    if task == "regression":
        return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    raise ValueError(f"Unknown task: {task!r}")


def derive_output_stem(output_file: str) -> str:
    """Return output_file with its final extension removed.

    Shared by every "<output_stem>_<suffix>.csv" derived-output-path site
    (group summary, SHAP interactions, synthetic feature suggestions, and
    the MCP server's mirrored output_files dict), which otherwise differ
    only in trivial extension-stripping mechanics (`os.path.splitext` vs.
    `str.rsplit(".", 1)`) repeated at each call site.

    Args:
        output_file: Path to the main results output file, e.g.
            "results.csv" or "out/report.html".

    Returns:
        output_file with its final extension removed, e.g. "results" or
        "out/report".
    """
    return os.path.splitext(output_file)[0]


#: CSS background color per recommendation tier, shared by the row
#: highlighting and the importance chart's bar colors so the two stay
#: visually consistent.
_RECOMMENDATION_COLORS = {
    "Keep": "#d4edda",
    "Caution": "#fff3cd",
    "Drop": "#f8d7da",
}

_RECOMMENDATION_CHART_COLORS = {
    "Keep": "#28a745",
    "Caution": "#ffc107",
    "Drop": "#dc3545",
}


def _render_importance_chart(results: pd.DataFrame) -> str | None:
    """Render a horizontal bar chart of SHAP importance, color-coded by
    recommendation tier, as a base64-encoded PNG `<img>` tag.

    Returns None (rather than raising) if matplotlib is unavailable or the
    results have no shap_importance column, since the chart is a visual
    nice-to-have and a missing/broken chart shouldn't block report
    generation.
    """
    if "shap_importance" not in results.columns or results.empty:
        return None
    try:
        import base64
        from io import BytesIO

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    ordered = results.sort_values("shap_importance", ascending=True)
    colors = [
        _RECOMMENDATION_CHART_COLORS.get(rec, "#6c757d")
        for rec in ordered.get("recommendation", pd.Series(["Keep"] * len(ordered)))
    ]
    fig_height = max(2.0, 0.35 * len(ordered))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(ordered["feature_name"], ordered["shap_importance"], color=colors)
    ax.set_xlabel("SHAP importance")
    ax.set_title("Feature importance by recommendation")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img alt="Feature importance chart" src="data:image/png;base64,{encoded}">'


def _highlight_recommendation_rows(table_html: str) -> str:
    """Color each `<tr>` in a pandas-rendered table by its Keep/Caution/Drop
    recommendation cell, without altering or re-escaping any cell content.

    Operates on already-escaped HTML (`DataFrame.to_html(escape=True)`), so
    this only ever matches the literal, known-safe recommendation strings —
    it never interprets feature names, reasons, or other user-derived data
    as HTML. The recommendation column's position varies (it's not always
    first), so each `<tr>...</tr>` block is searched as a whole rather than
    assuming the marker immediately follows the opening tag.
    """

    def _color_row(match: re.Match[str]) -> str:
        row = match.group(0)
        for recommendation, color in _RECOMMENDATION_COLORS.items():
            if f"<td>{recommendation}</td>" in row:
                return row.replace("<tr>", f'<tr style="background-color: {color}">', 1)
        return row

    return re.sub(r"<tr>.*?</tr>", _color_row, table_html, flags=re.DOTALL)


def save_results(
    results: pd.DataFrame, output_file: str, output_format: str = "csv"
) -> None:
    """Save feature selection results to a file.

    Args:
        results: DataFrame with columns: feature_name, shap_importance,
            permutation_importance, recommendation, reason.
        output_file: Path to write the output file.
        output_format: Either "csv" or "html".

    Raises:
        ValueError: If output_format is not "csv" or "html".
    """
    logger = logging.getLogger(__name__)
    if output_format == "csv":
        results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    elif output_format == "html":
        n_total = len(results)
        counts = results["recommendation"].value_counts()
        summary = (
            f"<p><strong>Total features:</strong> {n_total} &nbsp;|&nbsp; "
            f"<strong style='color:#28a745'>Keep:</strong> {counts.get('Keep', 0)} "
            f"&nbsp;|&nbsp; <strong style='color:#cc9a06'>Caution:</strong>"
            f" {counts.get('Caution', 0)} &nbsp;|&nbsp;"
            f" <strong style='color:#dc3545'>Drop:</strong> {counts.get('Drop', 0)}</p>"
        )
        chart_img = _render_importance_chart(results)
        chart_html = f"<h2>Importance chart</h2>{chart_img}" if chart_img else ""
        table = _highlight_recommendation_rows(results.to_html(index=False))
        html = f"""<!DOCTYPE html>
<html>
<head><title>TreeShaker Feature Selection Report</title></head>
<body>
<h1>TreeShaker Feature Selection Report</h1>
{summary}
{chart_html}
<h2>Feature details</h2>
{table}
</body>
</html>"""
        with open(output_file, "w") as f:
            f.write(html)
        logger.info(f"Report saved to {output_file}")
    else:
        raise ValueError(
            f"output_format must be 'csv' or 'html', got {output_format!r}"
        )
