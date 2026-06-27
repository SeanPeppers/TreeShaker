"""Dtype coverage audit: "any feature of any data type" claim.

TreeShaker's pitch is that it can be pointed at an arbitrary CSV. This file
is the single place every dtype category that claim implies is exercised
end-to-end, so the coverage stays visible in one file rather than scattered
across the other test modules (which test individual functions/modules in
isolation).

Categories covered, each in its own synthetic pandas fixture for CI safety
(no network calls -- see module docstrings in dtype_handling.py for the
per-category rationale):

1. Numeric (baseline) -- confirm no regression.
2. Low-cardinality string categorical.
3. High-cardinality string categorical (e.g. user IDs).
4. Date-string / datetime64 columns.
5. Free-text columns (should raise, not silently mis-encode).
6. Missing values (NaN).
7. Boolean columns.
8. A combined fixture mixing all of the above in one dataset.

Plus one real-public-dataset smoke test using sklearn's bundled
`load_diabetes` (no network call, ships with scikit-learn) for the "should
just work" numeric baseline end-to-end.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_config

from treeshaker.config import Config
from treeshaker.dtype_handling import (
    prepare_feature_dtypes,
    prepare_feature_dtypes_splits,
)
from treeshaker.exceptions import ValidationError
from treeshaker.importance import (
    compute_mutual_information,
    compute_permutation_importance,
)
from treeshaker.pipeline import FeatureSelectionPipeline


def _fast_config(input_file: str, task: str, tmp_path) -> Config:
    """A Config with tuning skipped, for fast deterministic test runs."""
    return make_config(input_file, task, tmp_path, skip_tuning=True)


# ---------------------------------------------------------------------------
# Fixtures: one per dtype category, plus a combined one.
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """Baseline: purely numeric features, no edge cases."""
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    df = pd.DataFrame({"num1": x1, "num2": x2})
    df["target"] = (x1 + x2 + rng.standard_normal(n) * 0.1 > 0).astype(int)
    return df


@pytest.fixture
def low_cardinality_categorical_df() -> pd.DataFrame:
    """A string column with only 3 distinct values."""
    rng = np.random.default_rng(1)
    n = 200
    cat = rng.choice(["red", "green", "blue"], size=n)
    num = rng.standard_normal(n)
    df = pd.DataFrame({"color": cat, "num": num})
    weight = np.where(cat == "red", 1.0, np.where(cat == "green", -1.0, 0.0))
    df["target"] = (weight + num + rng.standard_normal(n) * 0.1 > 0).astype(int)
    return df


@pytest.fixture
def high_cardinality_categorical_df() -> pd.DataFrame:
    """A string column with hundreds of unique short values (e.g. zip codes)."""
    rng = np.random.default_rng(2)
    n = 300
    zip_code = [f"{rng.integers(10000, 99999)}" for _ in range(n)]
    num = rng.standard_normal(n)
    df = pd.DataFrame({"zip_code": zip_code, "num": num})
    df["target"] = (num + rng.standard_normal(n) * 0.1 > 0).astype(int)
    return df


@pytest.fixture
def datetime_df() -> pd.DataFrame:
    """A native datetime64 column plus a numeric feature."""
    rng = np.random.default_rng(3)
    n = 200
    dates = pd.date_range("2021-01-01", periods=n, freq="3D")
    num = rng.standard_normal(n)
    df = pd.DataFrame({"signup_date": dates, "num": num})
    df["target"] = (num + rng.standard_normal(n) * 0.1 > 0).astype(int)
    return df


@pytest.fixture
def date_string_csv(tmp_path, datetime_df) -> str:
    """The datetime_df written to CSV -- pd.read_csv reads dates as strings,
    which is the realistic path users hit (no native datetime64 dtype
    survives a CSV round-trip without parse_dates)."""
    path = tmp_path / "date_string.csv"
    datetime_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def free_text_df() -> pd.DataFrame:
    """A column of long, mostly-unique strings (not categorical-like)."""
    rng = np.random.default_rng(4)
    n = 200
    words = ["excellent", "terrible", "average", "outstanding", "mediocre", "great"]
    text = [
        f"This product was {rng.choice(words)} and the delivery took"
        f" {rng.integers(1, 30)} days, review id {i}"
        for i in range(n)
    ]
    num = rng.standard_normal(n)
    df = pd.DataFrame({"review_text": text, "num": num})
    df["target"] = (num + rng.standard_normal(n) * 0.1 > 0).astype(int)
    return df


@pytest.fixture
def nan_df() -> pd.DataFrame:
    """A numeric column with ~20% missing values."""
    rng = np.random.default_rng(5)
    n = 200
    num1 = rng.standard_normal(n)
    num2 = rng.standard_normal(n)
    nan_mask = rng.random(n) < 0.2
    num2_with_nan = np.where(nan_mask, np.nan, num2)
    df = pd.DataFrame({"num1": num1, "num2_sparse": num2_with_nan})
    df["target"] = (num1 + num2 + rng.standard_normal(n) * 0.1 > 0).astype(int)
    return df


@pytest.fixture
def boolean_df() -> pd.DataFrame:
    """A boolean column plus a numeric feature."""
    rng = np.random.default_rng(6)
    n = 200
    flag = rng.choice([True, False], size=n)
    num = rng.standard_normal(n)
    df = pd.DataFrame({"is_active": flag, "num": num})
    weight = np.where(flag, 1.0, -1.0)
    df["target"] = (weight + num + rng.standard_normal(n) * 0.1 > 0).astype(int)
    return df


@pytest.fixture
def combined_df() -> pd.DataFrame:
    """All dtype categories (except free text) mixed into one dataset."""
    rng = np.random.default_rng(7)
    n = 250
    num1 = rng.standard_normal(n)
    num2 = rng.standard_normal(n)
    cat_low = rng.choice(["a", "b", "c"], size=n)
    cat_high = [f"user_{rng.integers(0, 100000)}" for _ in range(n)]
    dates = pd.date_range("2019-06-01", periods=n, freq="2D")
    nan_mask = rng.random(n) < 0.2
    num_with_nan = np.where(nan_mask, np.nan, rng.standard_normal(n))
    flag = rng.choice([True, False], size=n)

    df = pd.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "cat_low": cat_low,
            "cat_high": cat_high,
            "signup_date": dates,
            "nan_num": num_with_nan,
            "is_active": flag,
        }
    )
    df["target"] = (num1 + num2 + rng.standard_normal(n) * 0.1 > 0).astype(int)
    return df


def _to_csv(df: pd.DataFrame, tmp_path, name: str) -> str:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# 1. Numeric baseline -- confirm no regression.
# ---------------------------------------------------------------------------


def test_numeric_baseline_runs_unchanged(numeric_df, tmp_path):
    csv_path = _to_csv(numeric_df, tmp_path, "numeric.csv")
    config = _fast_config(csv_path, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert set(results["feature_name"]) == {"num1", "num2"}
    assert results["shap_importance"].notna().all()


# ---------------------------------------------------------------------------
# 2. Low-cardinality categorical.
# ---------------------------------------------------------------------------


def test_low_cardinality_categorical_cast_to_category_dtype(
    low_cardinality_categorical_df,
):
    prepared, feature_names, report = prepare_feature_dtypes(
        low_cardinality_categorical_df, ["color", "num"]
    )
    assert "color" in report.categorical_columns
    assert isinstance(prepared["color"].dtype, pd.CategoricalDtype)
    assert feature_names == ["color", "num"]


def test_low_cardinality_categorical_pipeline_runs(
    low_cardinality_categorical_df, tmp_path
):
    csv_path = _to_csv(low_cardinality_categorical_df, tmp_path, "low_card.csv")
    config = _fast_config(csv_path, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert set(results["feature_name"]) == {"color", "num"}
    # The categorical feature should get a real (non-NaN) SHAP importance --
    # it has genuine signal baked into the fixture's target.
    color_row = results[results["feature_name"] == "color"].iloc[0]
    assert pd.notna(color_row["shap_importance"])


# ---------------------------------------------------------------------------
# 3. High-cardinality categorical (e.g. zip codes).
# ---------------------------------------------------------------------------


def test_high_cardinality_categorical_cast_to_category_not_rejected(
    high_cardinality_categorical_df,
):
    prepared, feature_names, report = prepare_feature_dtypes(
        high_cardinality_categorical_df, ["zip_code", "num"]
    )
    assert "zip_code" in report.categorical_columns
    assert isinstance(prepared["zip_code"].dtype, pd.CategoricalDtype)
    # High cardinality alone (hundreds of unique short values) must not
    # trip the free-text rejection -- only long+mostly-unique values should.
    assert high_cardinality_categorical_df["zip_code"].nunique() > 100


def test_high_cardinality_categorical_pipeline_runs(
    high_cardinality_categorical_df, tmp_path
):
    csv_path = _to_csv(high_cardinality_categorical_df, tmp_path, "high_card.csv")
    config = _fast_config(csv_path, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert set(results["feature_name"]) == {"zip_code", "num"}


def test_high_cardinality_categorical_survives_train_val_test_concat():
    """Regression test: train/val/test splits of a high-cardinality column
    have non-overlapping categories. Concatenating two `category`-dtype
    frames with different category sets silently demotes the column back
    to plain object/str dtype in pandas, which previously surfaced as a
    confusing XGBoost dtype error inside train_model() (which concats train
    + val for the final fit). prepare_feature_dtypes_splits must prepare
    all three splits jointly so they share one category set."""
    rng = np.random.default_rng(8)
    n = 300
    df = pd.DataFrame(
        {
            "id_col": [f"id_{i}" for i in range(n)],  # every value unique
            "num": rng.standard_normal(n),
            "target": rng.choice([0, 1], size=n),
        }
    )
    train, val, test = df.iloc[:180], df.iloc[180:240], df.iloc[240:]
    train_p, val_p, test_p, feature_names, _ = prepare_feature_dtypes_splits(
        train, val, test, ["id_col", "num"]
    )
    combined = pd.concat([train_p, val_p], ignore_index=True)
    assert isinstance(combined["id_col"].dtype, pd.CategoricalDtype)


# ---------------------------------------------------------------------------
# 4. Datetime / date-string columns.
# ---------------------------------------------------------------------------


def test_native_datetime64_column_expanded_to_numeric_columns(datetime_df):
    prepared, feature_names, report = prepare_feature_dtypes(
        datetime_df, ["signup_date", "num"]
    )
    assert "signup_date" in report.datetime_columns_expanded
    expected_derived = [
        "signup_date_year",
        "signup_date_month",
        "signup_date_day",
        "signup_date_dayofweek",
    ]
    assert report.datetime_columns_expanded["signup_date"] == expected_derived
    assert "signup_date" not in prepared.columns
    for derived in expected_derived:
        assert derived in prepared.columns
        assert pd.api.types.is_numeric_dtype(prepared[derived])
    assert set(feature_names) == {"num", *expected_derived}


def test_date_string_column_from_csv_also_expanded(date_string_csv):
    """pd.read_csv has no native datetime dtype -- a date column always
    arrives as plain strings. This must be detected and expanded the same
    way as a native datetime64 column, not silently cast to (meaningless,
    near-unique) categorical or rejected as free text."""
    df = pd.read_csv(date_string_csv)
    assert df["signup_date"].dtype == object or df["signup_date"].dtype.name == "str"
    prepared, feature_names, report = prepare_feature_dtypes(df, ["signup_date", "num"])
    assert "signup_date" in report.datetime_columns_expanded
    assert "signup_date_year" in prepared.columns


def test_datetime_pipeline_runs_end_to_end(date_string_csv, tmp_path):
    config = _fast_config(date_string_csv, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert "signup_date_year" in set(results["feature_name"])
    assert "signup_date" not in set(results["feature_name"])


# ---------------------------------------------------------------------------
# 5. Free text -- must raise a clear, actionable error, not silently
#    mis-encode as categorical.
# ---------------------------------------------------------------------------


def test_free_text_column_raises_validation_error(free_text_df):
    with pytest.raises(ValidationError, match="free text"):
        prepare_feature_dtypes(free_text_df, ["review_text", "num"])


def test_free_text_pipeline_raises_clear_error(free_text_df, tmp_path):
    csv_path = _to_csv(free_text_df, tmp_path, "free_text.csv")
    config = _fast_config(csv_path, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    with pytest.raises(ValidationError, match="free text"):
        pipeline.run()


# ---------------------------------------------------------------------------
# 6. Missing values (NaN).
# ---------------------------------------------------------------------------


def test_nan_column_left_untouched_by_dtype_preparation(nan_df):
    prepared, feature_names, _ = prepare_feature_dtypes(nan_df, ["num1", "num2_sparse"])
    # XGBoost natively handles NaN in numeric features -- prepare_feature_dtypes
    # should not impute or drop it.
    assert prepared["num2_sparse"].isna().sum() == nan_df["num2_sparse"].isna().sum()


def test_nan_pipeline_runs_without_crashing(nan_df, tmp_path):
    csv_path = _to_csv(nan_df, tmp_path, "nan.csv")
    config = _fast_config(csv_path, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert set(results["feature_name"]) == {"num1", "num2_sparse"}
    # Mutual information must not crash or return NaN due to the missing
    # values reaching sklearn's mutual_info_* estimators (which reject NaN
    # outright) -- compute_mutual_information median-imputes internally.
    sparse_row = results[results["feature_name"] == "num2_sparse"].iloc[0]
    assert pd.notna(sparse_row["mutual_info_score"])


def test_mutual_information_handles_nan_without_crashing():
    rng = np.random.default_rng(9)
    n = 100
    x = rng.standard_normal(n)
    x_with_nan = np.where(rng.random(n) < 0.3, np.nan, x)
    y = (x > 0).astype(int)
    df = pd.DataFrame({"x": x_with_nan, "target": y})
    scores = compute_mutual_information(
        data=df,
        target_column="target",
        feature_names=["x"],
        task="binary",
        random_seed=0,
    )
    assert np.isfinite(scores).all()


def test_residual_correlation_skips_nan_safely(nan_df, tmp_path):
    """check_residuals=True must not crash or silently report a misleading
    'no correlation' due to NaN propagating into the Pearson correlation --
    pandas' .corr() already drops NaN pairwise, confirmed here end-to-end."""
    csv_path = _to_csv(nan_df, tmp_path, "nan_residuals.csv")
    config = _fast_config(csv_path, "binary", tmp_path)
    config.check_residuals = True
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert "residual_correlation" in results.columns
    assert results["residual_correlation"].notna().all()


# ---------------------------------------------------------------------------
# 7. Boolean columns.
# ---------------------------------------------------------------------------


def test_boolean_column_left_untouched_by_dtype_preparation(boolean_df):
    prepared, feature_names, report = prepare_feature_dtypes(
        boolean_df, ["is_active", "num"]
    )
    assert prepared["is_active"].dtype == bool
    assert "is_active" not in report.categorical_columns
    assert "is_active" not in report.datetime_columns_expanded


def test_boolean_pipeline_runs_end_to_end(boolean_df, tmp_path):
    csv_path = _to_csv(boolean_df, tmp_path, "boolean.csv")
    config = _fast_config(csv_path, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    flag_row = results[results["feature_name"] == "is_active"].iloc[0]
    assert pd.notna(flag_row["shap_importance"])
    assert pd.notna(flag_row["mutual_info_score"])


def test_permutation_importance_preserves_boolean_dtype_after_shuffle():
    """Shuffling a column via np.random.permutation degrades pandas
    `category` dtype to plain object -- this confirms boolean columns
    (which were already fine before this audit) and category columns
    (fixed by this audit) both round-trip a shuffle without dtype loss."""
    import xgboost as xgb

    rng = np.random.default_rng(10)
    n = 100
    df = pd.DataFrame(
        {
            "flag": rng.choice([True, False], size=n),
            "cat": pd.Series(rng.choice(["x", "y", "z"], size=n), dtype="category"),
        }
    )
    y = rng.choice([0, 1], size=n)
    model = xgb.XGBClassifier(enable_categorical=True)
    model.fit(df, y)

    importances, stds = compute_permutation_importance(
        model=model,
        data=pd.concat([df, pd.Series(y, name="target")], axis=1),
        target_column="target",
        feature_names=["flag", "cat"],
        metric_fn=lambda y_true, y_pred: float((y_true == y_pred).mean()),
        n_repeats=2,
    )
    assert np.isfinite(importances).all()
    assert np.isfinite(stds).all()


# ---------------------------------------------------------------------------
# 8. Combined fixture: everything mixed together in one dataset.
# ---------------------------------------------------------------------------


def test_combined_mixed_dtypes_pipeline_runs_end_to_end(combined_df, tmp_path):
    csv_path = _to_csv(combined_df, tmp_path, "combined.csv")
    config = _fast_config(csv_path, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    expected_features = {
        "num1",
        "num2",
        "cat_low",
        "cat_high",
        "nan_num",
        "is_active",
        "signup_date_year",
        "signup_date_month",
        "signup_date_day",
        "signup_date_dayofweek",
    }
    assert set(results["feature_name"]) == expected_features
    assert results["shap_importance"].notna().all()
    assert results["mutual_info_score"].notna().all()


def test_combined_mixed_dtypes_with_residuals_and_interactions(combined_df, tmp_path):
    """A stress test enabling the optional check_residuals and
    compute_shap_interactions paths together, since those touch feature
    columns directly (residual correlation, SHAP interaction matrix,
    synthetic-feature product candidates) and were a separate risk area
    from the core dtype preparation."""
    csv_path = _to_csv(combined_df, tmp_path, "combined_full.csv")
    config = _fast_config(csv_path, "binary", tmp_path)
    config.check_residuals = True
    config.compute_shap_interactions = True
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert "residual_correlation" in results.columns
    assert pipeline.interactions is not None
    assert len(pipeline.interactions) > 0


# ---------------------------------------------------------------------------
# Real public dataset smoke test -- no network call (sklearn-bundled).
# ---------------------------------------------------------------------------


def test_real_dataset_smoke_test_load_diabetes(tmp_path):
    """End-to-end smoke test on a real, bundled (no network) sklearn
    dataset, confirming the numeric "should just work" baseline holds on
    real-world (not synthetic) data."""
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    df = data.frame.rename(columns={"target": "target"})
    csv_path = _to_csv(df, tmp_path, "diabetes.csv")
    config = _fast_config(csv_path, "regression", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert len(results) == len(data.feature_names)
    assert results["shap_importance"].notna().all()
