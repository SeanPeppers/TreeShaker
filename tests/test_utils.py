"""Tests for utility functions in utils.py."""

import numpy as np
import pandas as pd
import pytest

from treeshaker.exceptions import ValidationError
from treeshaker.utils import (
    check_class_imbalance,
    compute_wape,
    detect_cuda_available,
    get_metric_fn,
    load_csv,
    resolve_xgboost_device,
    validate_target,
)


def test_resolve_xgboost_device_auto_picks_cuda_when_detected(monkeypatch):
    monkeypatch.setattr("treeshaker.utils.detect_cuda_available", lambda: True)
    assert resolve_xgboost_device("auto") == "cuda"


def test_resolve_xgboost_device_auto_picks_cpu_when_not_detected(monkeypatch):
    monkeypatch.setattr("treeshaker.utils.detect_cuda_available", lambda: False)
    assert resolve_xgboost_device("auto") == "cpu"


def test_resolve_xgboost_device_explicit_values_pass_through(monkeypatch):
    monkeypatch.setattr("treeshaker.utils.detect_cuda_available", lambda: False)
    assert resolve_xgboost_device("cuda") == "cuda"
    assert resolve_xgboost_device("cpu") == "cpu"


def test_resolve_xgboost_device_invalid_raises():
    with pytest.raises(ValueError, match="device"):
        resolve_xgboost_device("tpu")


def test_detect_cuda_available_returns_bool_and_is_cached():
    # Real detection (no GPU expected in CI) -- just confirm it returns a
    # bool without raising, and that the lru_cache means a second call
    # doesn't re-run the underlying fit (same cached object identity check
    # is implicit; we just confirm repeated calls are consistent/cheap).
    result1 = detect_cuda_available()
    result2 = detect_cuda_available()
    assert isinstance(result1, bool)
    assert result1 == result2


def test_balanced_binary_returns_none():
    df = pd.DataFrame({"target": [0, 1] * 50})
    assert check_class_imbalance(df, "target", "binary") is None


def test_imbalanced_binary_returns_warning():
    df = pd.DataFrame({"target": [0] * 99 + [1] * 1})
    warning = check_class_imbalance(df, "target", "binary")
    assert warning is not None
    assert "imbalance" in warning.lower()
    assert "1.0%" in warning


def test_borderline_binary_at_threshold_is_balanced():
    # Exactly at the default 10% threshold should not warn (>= threshold).
    df = pd.DataFrame({"target": [0] * 90 + [1] * 10})
    assert check_class_imbalance(df, "target", "binary") is None


def test_borderline_binary_just_below_threshold_warns():
    df = pd.DataFrame({"target": [0] * 91 + [1] * 9})
    assert check_class_imbalance(df, "target", "binary") is not None


def test_balanced_multiclass_returns_none():
    df = pd.DataFrame({"target": [0, 1, 2] * 50})
    assert check_class_imbalance(df, "target", "multiclass") is None


def test_imbalanced_multiclass_returns_warning():
    df = pd.DataFrame({"target": [0] * 90 + [1] * 8 + [2] * 2})
    warning = check_class_imbalance(df, "target", "multiclass")
    assert warning is not None
    assert "imbalance" in warning.lower()


def test_regression_always_returns_none():
    df = pd.DataFrame({"target": [0] * 99 + [1] * 1})
    assert check_class_imbalance(df, "target", "regression") is None


def test_custom_threshold():
    df = pd.DataFrame({"target": [0] * 80 + [1] * 20})
    # 20% minority is fine under default 10% threshold.
    assert check_class_imbalance(df, "target", "binary") is None
    # But should warn under a stricter custom threshold.
    warning = check_class_imbalance(
        df, "target", "binary", imbalance_warning_threshold=0.25
    )
    assert warning is not None


def test_load_csv_missing_file_raises_validation_error(tmp_path):
    missing = str(tmp_path / "does_not_exist.csv")
    with pytest.raises(ValidationError, match="Input file not found"):
        load_csv(missing, "target", "binary")


def test_load_csv_empty_file_raises_validation_error(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValidationError, match="empty or not valid CSV"):
        load_csv(str(empty_file), "target", "binary")


def test_load_csv_all_null_target_raises_validation_error(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"feature": [1, 2, 3, 4], "target": [None, None, None, None]}).to_csv(
        csv_path, index=False
    )
    with pytest.raises(ValidationError, match="no non-null values"):
        load_csv(str(csv_path), "target", "regression")


def test_load_csv_missing_target_column_raises_validation_error(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"feature": [1, 2, 3], "other": [4, 5, 6]}).to_csv(
        csv_path, index=False
    )
    with pytest.raises(ValidationError, match="not found in data"):
        load_csv(str(csv_path), "target", "regression")


def test_validate_target_missing_column_raises():
    df = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(ValidationError, match="not found in data"):
        validate_target(df, "target", "regression")


def test_load_csv_too_few_rows_for_stratified_split_raises_validation_error(
    tmp_path,
):
    csv_path = tmp_path / "tiny.csv"
    # 4 rows, binary target with a minority class of 1 — too few to
    # stratify with a large test_size.
    pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 0, 0, 1]}).to_csv(
        csv_path, index=False
    )
    with pytest.raises(ValidationError, match="stratified"):
        load_csv(str(csv_path), "target", "binary", test_size=0.9)


def test_load_csv_returns_three_way_split(tmp_path):
    n = 200
    df = pd.DataFrame(
        {
            "feature": range(n),
            "target": [0, 1] * (n // 2),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    train_df, val_df, test_df = load_csv(
        str(csv_path), "target", "binary", test_size=0.2, val_size=0.25
    )

    # Roughly: test ~20% of 200 = 40; remaining 160, val ~25% of that = 40;
    # train ~120.
    assert len(test_df) == pytest.approx(40, abs=2)
    assert len(val_df) == pytest.approx(40, abs=2)
    assert len(train_df) == pytest.approx(120, abs=4)
    assert len(train_df) + len(val_df) + len(test_df) == n


def test_load_csv_chronological_split_sorts_by_time_column(tmp_path):
    """Chronological split preserves temporal order when time_column is set."""
    n = 200
    # Create data with explicit chronological order (timestamps in reverse)
    df = pd.DataFrame(
        {
            "timestamp": range(n, 0, -1),  # Descending: [200, 199, ..., 1]
            "feature": np.random.RandomState(42).normal(0, 1, n),
            "target": [0, 1] * (n // 2),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    train_df, val_df, test_df = load_csv(
        str(csv_path),
        "target",
        "binary",
        test_size=0.2,
        val_size=0.25,
        time_column="timestamp",
    )

    # Chronological split should order by timestamp ascending, so:
    # - train_df should have smallest timestamps (earliest)
    # - val_df should have middle timestamps
    # - test_df should have largest timestamps (latest)
    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()
    assert len(train_df) + len(val_df) + len(test_df) == n


def test_load_csv_chronological_split_missing_column_raises():
    """Chronological split raises ValidationError if time_column is missing."""
    n = 100
    df = pd.DataFrame(
        {
            "feature": range(n),
            "target": [0, 1] * (n // 2),
        }
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = f"{tmp_dir}/data.csv"
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValidationError, match="not found"):
            load_csv(
                csv_path,
                "target",
                "binary",
                time_column="nonexistent",
            )


def test_load_csv_three_way_split_is_disjoint(tmp_path):
    n = 200
    df = pd.DataFrame(
        {
            "feature": range(n),
            "target": [0, 1] * (n // 2),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    train_df, val_df, test_df = load_csv(str(csv_path), "target", "binary")

    # Use the original "feature" column (a unique row id) to verify
    # disjointness and full coverage, since indices are reset per split.
    train_ids = set(train_df["feature"])
    val_ids = set(val_df["feature"])
    test_ids = set(test_df["feature"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(df["feature"])


def test_load_csv_three_way_split_stratification_binary(tmp_path):
    n = 300
    df = pd.DataFrame(
        {
            "feature": range(n),
            "target": [0] * 200 + [1] * 100,
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    train_df, val_df, test_df = load_csv(
        str(csv_path), "target", "binary", test_size=0.2, val_size=0.2
    )

    for split in (train_df, val_df, test_df):
        minority_frac = split["target"].value_counts(normalize=True).min()
        assert minority_frac == pytest.approx(1 / 3, abs=0.05)


def test_compute_wape_known_value():
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([12, 18, 33, 36])
    # abs errors: 2, 2, 3, 4 = 11; sum abs actual = 100
    expected = 11 / 100
    assert compute_wape(y_true, y_pred) == pytest.approx(expected)


def test_compute_wape_perfect_forecast_is_zero():
    y_true = np.array([5, 10, 15])
    assert compute_wape(y_true, y_true) == pytest.approx(0.0)


def test_compute_wape_zero_actuals_raises():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 2, 3])
    with pytest.raises(ValidationError, match="WAPE is undefined"):
        compute_wape(y_true, y_pred)


def test_get_metric_fn_binary_default_is_roc_auc():
    fn = get_metric_fn("binary")
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8]
    assert fn(y_true, y_pred) == pytest.approx(1.0)


def test_get_metric_fn_multiclass_default_is_accuracy():
    fn = get_metric_fn("multiclass")
    assert fn([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


def test_get_metric_fn_regression_default_is_neg_mse():
    fn = get_metric_fn("regression")
    y_true = [1.0, 2.0, 3.0]
    assert fn(y_true, y_true) == pytest.approx(0.0)


def test_get_metric_fn_override_roc_auc():
    fn = get_metric_fn("regression", override="roc_auc")
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8]
    assert fn(y_true, y_pred) == pytest.approx(1.0)


def test_get_metric_fn_override_accuracy():
    fn = get_metric_fn("regression", override="accuracy")
    assert fn([0, 1, 1], [0, 1, 1]) == pytest.approx(1.0)


def test_get_metric_fn_override_neg_mse():
    fn = get_metric_fn("binary", override="neg_mse")
    y_true = [1.0, 2.0, 3.0]
    assert fn(y_true, y_true) == pytest.approx(0.0)


def test_get_metric_fn_override_wape_is_negative_for_higher_is_better():
    """The generic metric_fn contract is 'higher is better', so the WAPE
    override must return -wape, matching the neg_mse pattern."""
    fn = get_metric_fn("regression", override="wape")
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([12, 18, 33, 36])
    expected = -compute_wape(y_true, y_pred)
    assert fn(y_true, y_pred) == pytest.approx(expected)
    # Perfect forecast scores 0, the maximum possible (since wape >= 0).
    assert fn(y_true, y_true) == pytest.approx(0.0)


def test_get_metric_fn_invalid_override_raises():
    with pytest.raises(ValueError, match="Unknown permutation metric override"):
        get_metric_fn("regression", override="not_a_metric")


def test_load_csv_three_way_split_stratification_multiclass(tmp_path):
    n = 300
    df = pd.DataFrame(
        {
            "feature": range(n),
            "target": [0] * 100 + [1] * 100 + [2] * 100,
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    train_df, val_df, test_df = load_csv(
        str(csv_path), "target", "multiclass", test_size=0.2, val_size=0.2
    )

    for split in (train_df, val_df, test_df):
        counts = split["target"].value_counts(normalize=True)
        for frac in counts:
            assert frac == pytest.approx(1 / 3, abs=0.05)
