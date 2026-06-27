"""Utility functions for TreeShaker."""

import logging

import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from .exceptions import ValidationError


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


def load_csv(
    filepath: str,
    target_column: str,
    task: str,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a CSV file and split it into train and validation sets.

    Args:
        filepath: Path to the input CSV file.
        target_column: Name of the target column.
        task: One of "binary", "multiclass", or "regression".
        test_size: Fraction of data to use for validation.
        random_seed: Random seed for reproducible splits.

    Returns:
        A tuple of (train_df, val_df), both including the target column.

    Raises:
        ValidationError: If the target column is missing or incompatible with
            the task.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    validate_target(df, target_column, task)
    df = df.dropna(subset=[target_column])
    stratify = df[target_column] if task in ("binary", "multiclass") else None
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )
    logger.info(f"Split: {len(train_df)} train, {len(val_df)} validation")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_metric_fn(task: str):
    """Return the evaluation metric function for the given task.

    The returned function has the signature (y_true, y_pred) -> float,
    where higher values indicate better performance.

    Args:
        task: One of "binary", "multiclass", or "regression".

    Returns:
        A callable metric function.

    Raises:
        ValueError: If the task is not recognized.
    """
    if task == "binary":
        return lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
    if task == "multiclass":
        return lambda y_true, y_pred: accuracy_score(y_true, y_pred)
    if task == "regression":
        return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    raise ValueError(f"Unknown task: {task!r}")


def save_results(
    results: pd.DataFrame, output_file: str, output_format: str = "csv"
) -> None:
    """Save feature selection results to a file.

    Args:
        results: DataFrame with columns: feature_name, shap_importance,
            permutation_importance, is_harmful, harm_reason.
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
        n_harmful = results["is_harmful"].sum()
        n_total = len(results)
        summary = (
            f"<p><strong>Total features:</strong> {n_total} | "
            f"<strong>Harmful features:</strong> {n_harmful}</p>"
        )
        table = results.to_html(index=False, escape=False)
        html = f"""<!DOCTYPE html>
<html>
<head><title>TreeShaker Feature Selection Report</title></head>
<body>
<h1>TreeShaker Feature Selection Report</h1>
{summary}
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
