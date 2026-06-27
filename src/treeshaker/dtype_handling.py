"""Dtype validation and preparation for arbitrary input feature columns.

TreeShaker's value proposition is "feed it any CSV and get back feature
recommendations" -- but XGBoost, SHAP, mutual information, and the
correlation/VIF-based leakage checks each make their own (different, and
sometimes incompatible) assumptions about column dtype. Left unhandled, the
mismatches show up in one of two bad ways: a confusing low-level traceback
from deep inside xgboost/sklearn, or a silently wrong result (e.g. a string
ID column accidentally included in a VIF regression).

This module is the single place those dtype assumptions are reconciled
*before* anything is fit, so every downstream module can keep its existing,
simpler dtype assumptions:

- **Numeric / bool**: left untouched. `pd.api.types.is_numeric_dtype` is
  already True for bool columns, so every existing numeric-only code path
  (VIF, correlation, mutual information) already handles them correctly
  with no change needed here.
- **Datetime**: XGBoost cannot consume a datetime64 column directly, and
  the failure mode is a confusing low-level error. Rather than silently
  dropping the column or raising and forcing a manual workaround, this
  auto-derives clearly-named numeric columns (`{col}_year`, `{col}_month`,
  `{col}_day`, `{col}_dayofweek`) and removes the raw datetime column from
  the feature set -- consistent with the project's existing convention of
  preferring an automatic, clearly-named derivation over a hard failure
  (see `derive_prefix_groups` in pipeline.py). No lag/rolling/cyclical
  feature engineering is attempted; that is out of scope and left to the
  user.
- **String columns**: split into categorical vs. free text by cardinality
  and average value length (see `_looks_like_free_text`). Categorical
  columns (low *or* high cardinality -- e.g. a zip code or user ID column
  with hundreds of uniques is still "categorical", just high-cardinality)
  are cast to pandas `category` dtype, which XGBoost 2.x consumes natively
  with `enable_categorical=True`. Free text columns are not safe to treat
  as categorical (every value is ~unique, so a category encoding carries
  no signal and SHAP/leakage stats on it are meaningless) -- these raise a
  `ValidationError` with actionable guidance to pre-vectorize/embed the
  column before passing it to TreeShaker.
- **NaN**: not handled here. XGBoost natively tolerates NaN in numeric
  features (it learns a default split direction for missing values), and
  the leakage/residual correlation code paths already skip non-numeric
  columns and propagate NaN-safe correlations (pandas' `.corr()` already
  drops NaN pairwise). No change needed for this category; see
  tests/test_dtype_coverage.py for the confirming coverage.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from .exceptions import ValidationError

#: A string column is treated as free text (not categorical) when both
#: of these hold: the fraction of unique values is at or above this ratio...
_FREE_TEXT_UNIQUE_RATIO_THRESHOLD = 0.5
#: ...AND the average string length is at or above this many characters.
#: Long, mostly-unique strings (e.g. a free-text product review) carry no
#: usable signal as a category encoding -- every value is close to its own
#: singleton category. A high-cardinality but short-valued column (e.g. a
#: 5-character zip code) does NOT trip this check and is correctly treated
#: as ordinary (if high-cardinality) categorical data instead.
_FREE_TEXT_AVG_LENGTH_THRESHOLD = 20.0

#: A string column is treated as a date-string column (and expanded the
#: same way as a native datetime64 column) when at least this fraction of
#: its non-null values successfully parse as a date via pd.to_datetime.
#: CSV input has no native datetime dtype -- pd.read_csv always reads a
#: date column as plain strings -- so this string-based heuristic is the
#: only way to detect "this is actually a date" for CSV-sourced data.
#: Less than 100% tolerates a few stray malformed/missing-looking values
#: without losing the whole column to the free-text/categorical path.
_DATE_STRING_PARSE_RATIO_THRESHOLD = 0.95


@dataclass
class DtypePreparationReport:
    """Summary of dtype handling decisions made for one dataset.

    Attributes:
        datetime_columns_expanded: Mapping of original datetime column name
            to the list of derived numeric column names that replaced it.
        categorical_columns: Names of string columns cast to pandas
            `category` dtype (covers both low- and high-cardinality cases).
    """

    datetime_columns_expanded: dict[str, list[str]]
    categorical_columns: list[str]


def _looks_like_free_text(col: pd.Series) -> bool:
    """Heuristically detect a free-text column (not safe to treat as categorical).

    Args:
        col: A string-dtype (object or pandas StringDtype) Series.

    Returns:
        True if the column's values are both mostly-unique and long on
        average, suggesting free text (e.g. product reviews, comments)
        rather than a categorical label (e.g. a zip code or user ID, which
        can be high-cardinality but is still short and meaningfully
        repeated/structured).
    """
    non_null = col.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique() / len(non_null)
    avg_length = non_null.astype(str).str.len().mean()
    return (
        unique_ratio >= _FREE_TEXT_UNIQUE_RATIO_THRESHOLD
        and avg_length >= _FREE_TEXT_AVG_LENGTH_THRESHOLD
    )


def _looks_like_date_strings(col: pd.Series) -> pd.Series | None:
    """Detect a string column that actually holds date/datetime values.

    CSV input has no native datetime dtype: `pd.read_csv` always reads a
    date column (e.g. "2020-01-01") as a plain string column. Without this
    check, such a column would otherwise fall through to the categorical
    path -- treating each calendar date as its own category is rarely
    useful and, for a column spanning many distinct dates, is functionally
    indistinguishable from the free-text/high-cardinality problem this
    module exists to avoid.

    Args:
        col: A string-dtype (object or pandas StringDtype) Series.

    Returns:
        The parsed datetime64 Series (same index as col, unparseable
        values as NaT) if at least
        `_DATE_STRING_PARSE_RATIO_THRESHOLD` of its non-null values parse
        as a date. Otherwise None.
    """
    non_null = col.dropna()
    if non_null.empty:
        return None
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    parse_ratio = parsed.notna().mean()
    if parse_ratio < _DATE_STRING_PARSE_RATIO_THRESHOLD:
        return None
    return pd.to_datetime(col, errors="coerce", format="mixed")


def _expand_datetime_column(col: pd.Series, name: str) -> pd.DataFrame:
    """Derive clearly-named numeric columns from a datetime column.

    Args:
        col: A datetime64-dtype Series.
        name: The original column name, used as a prefix for derived names.

    Returns:
        DataFrame with columns `{name}_year`, `{name}_month`, `{name}_day`,
        `{name}_dayofweek`, indexed the same as col. `dayofweek` is 0=Monday
        .. 6=Sunday, matching pandas' own convention. Values are nullable
        Int64 so that NaT rows propagate as NaN rather than crashing.
    """
    dt = pd.DatetimeIndex(col)
    return pd.DataFrame(
        {
            f"{name}_year": dt.year.to_numpy(),
            f"{name}_month": dt.month.to_numpy(),
            f"{name}_day": dt.day.to_numpy(),
            f"{name}_dayofweek": dt.dayofweek.to_numpy(),
        },
        index=col.index,
    ).astype("float64")


def prepare_feature_dtypes(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, list[str], DtypePreparationReport]:
    """Validate and normalize feature column dtypes before model fitting.

    Must be called once, immediately after the train/val/test split (so the
    same column set and derived names are used consistently across all
    three splits) and before any model fitting, SHAP, mutual information,
    or leakage detection.

    Args:
        df: A DataFrame containing at least the columns in feature_names
            (the target column, if present, is left untouched).
        feature_names: Ordered list of feature column names to validate and
            prepare. Columns not in this list are passed through unchanged.

    Returns:
        A tuple (prepared_df, prepared_feature_names, report):
        - prepared_df: A copy of df with datetime columns replaced by their
          derived numeric columns and string categorical columns cast to
          `category` dtype. Row order and non-feature columns (e.g. the
          target) are unchanged.
        - prepared_feature_names: feature_names with datetime column names
          replaced by their derived column names (in place, preserving
          relative order of the other features) and bool/numeric columns
          unchanged.
        - report: A DtypePreparationReport describing what was changed, for
          logging.

    Raises:
        ValidationError: If a feature column looks like free text (mostly
            unique, long string values) -- treating it as categorical would
            be meaningless (every value its own near-singleton category)
            and silently produce uninformative SHAP/leakage results. The
            error message recommends pre-vectorizing or embedding the
            column (e.g. TF-IDF, sentence embeddings) before passing it to
            TreeShaker, and explains how to opt out for a column that is
            not actually free text (rename or pre-encode it).
    """
    logger = logging.getLogger(__name__)
    prepared = df.copy()
    new_feature_names: list[str] = []
    datetime_expanded: dict[str, list[str]] = {}
    categorical_columns: list[str] = []

    for name in feature_names:
        col = prepared[name]

        if pd.api.types.is_datetime64_any_dtype(col):
            derived = _expand_datetime_column(col, name)
            prepared = prepared.drop(columns=[name])
            prepared = pd.concat([prepared, derived], axis=1)
            derived_names = list(derived.columns)
            datetime_expanded[name] = derived_names
            new_feature_names.extend(derived_names)
            logger.info(
                f"Feature '{name}' is a datetime column; auto-derived"
                f" numeric columns {derived_names} and removed the raw"
                " datetime column from the feature set."
            )
            continue

        if col.dtype == object or isinstance(col.dtype, pd.StringDtype):
            parsed_dates = _looks_like_date_strings(col)
            if parsed_dates is not None:
                derived = _expand_datetime_column(parsed_dates, name)
                prepared = prepared.drop(columns=[name])
                prepared = pd.concat([prepared, derived], axis=1)
                derived_names = list(derived.columns)
                datetime_expanded[name] = derived_names
                new_feature_names.extend(derived_names)
                logger.info(
                    f"Feature '{name}' looks like a date string column;"
                    f" auto-derived numeric columns {derived_names} and"
                    " removed the raw string column from the feature set."
                )
                continue

            if _looks_like_free_text(col):
                raise ValidationError(
                    f"Feature '{name}' looks like free text, not a"
                    " categorical column: its values are mostly unique"
                    " and long on average. Treating it as a categorical"
                    " feature would be meaningless (every value would be"
                    " close to its own singleton category) and would"
                    " produce uninformative SHAP and leakage statistics."
                    " Pre-vectorize or embed this column (e.g. TF-IDF or"
                    " a sentence embedding) into numeric columns before"
                    " passing it to TreeShaker, or drop it from the input"
                    " if it isn't meant to be a model feature."
                )
            prepared[name] = col.astype("category")
            categorical_columns.append(name)
            new_feature_names.append(name)
            continue

        # Numeric (including bool, which pandas already reports as
        # numeric) and pandas `category` dtype columns are left untouched
        # -- existing code paths already handle them correctly.
        new_feature_names.append(name)

    if categorical_columns:
        logger.info(
            f"{len(categorical_columns)} string column(s) cast to"
            f" categorical dtype for native XGBoost support: {categorical_columns}"
        )

    report = DtypePreparationReport(
        datetime_columns_expanded=datetime_expanded,
        categorical_columns=categorical_columns,
    )
    return prepared, new_feature_names, report


def prepare_feature_dtypes_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], DtypePreparationReport]:
    """Apply prepare_feature_dtypes consistently across a train/val/test split.

    Calling `prepare_feature_dtypes` independently on each split is *not*
    equivalent to calling it once before splitting: pandas `category`
    dtype only records the categories present in *that* frame, so train,
    val, and test end up with different category sets for any
    high-cardinality categorical column (e.g. a user ID column where
    almost every value is split-specific). That divergence is latent until
    something concatenates two splits with different category sets --
    which `pd.concat` silently resolves by **demoting the column back to
    plain object/str dtype** -- and `train_model` does exactly that
    (`pd.concat([train_data, val_data])` for the final combined fit),
    surfacing as the same confusing low-level XGBoost dtype error this
    module exists to prevent, just one step later.

    This function avoids that by preparing dtypes on the three splits
    *concatenated*, then slicing the result back apart by row count -- so
    every derived `category` column shares one category set (the union
    across all three splits) no matter how the splits get recombined
    downstream.

    Args:
        train_df: Training split, including the target column.
        val_df: Validation split, including the target column.
        test_df: Test split, including the target column.
        feature_names: Ordered list of feature column names to validate
            and prepare. Columns not in this list are passed through
            unchanged.

    Returns:
        A tuple (train_prepared, val_prepared, test_prepared,
        prepared_feature_names, report), analogous to
        `prepare_feature_dtypes`'s return value but for all three splits
        at once, each retaining its original row count and index.

    Raises:
        ValidationError: Propagated from `prepare_feature_dtypes` if a
            feature column looks like free text.
    """
    n_train, n_val = len(train_df), len(val_df)
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    prepared, prepared_feature_names, report = prepare_feature_dtypes(
        combined, feature_names
    )
    train_prepared = prepared.iloc[:n_train].reset_index(drop=True)
    val_prepared = prepared.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_prepared = prepared.iloc[n_train + n_val :].reset_index(drop=True)
    return train_prepared, val_prepared, test_prepared, prepared_feature_names, report
