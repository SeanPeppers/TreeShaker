"""Tests for utility functions in utils.py."""

import pandas as pd
import pytest

from treeshaker.exceptions import ValidationError
from treeshaker.utils import check_class_imbalance, load_csv, validate_target


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
