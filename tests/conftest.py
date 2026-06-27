"""Shared pytest fixtures for TreeShaker tests."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def binary_data() -> pd.DataFrame:
    """200-row binary classification dataset with 10 features (6 informative)."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    return df


@pytest.fixture
def multiclass_data() -> pd.DataFrame:
    """250-row multiclass dataset with 15 features and 3 classes."""
    X, y = make_classification(
        n_samples=250,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(15)])
    df["target"] = y
    return df


@pytest.fixture
def regression_data() -> pd.DataFrame:
    """300-row regression dataset with 20 features (15 informative)."""
    X, y = make_regression(
        n_samples=300,
        n_features=20,
        n_informative=15,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    df["target"] = y
    return df


@pytest.fixture
def binary_csv(tmp_path, binary_data) -> str:
    """Binary classification data saved as CSV. Returns file path."""
    path = tmp_path / "binary.csv"
    binary_data.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def multiclass_csv(tmp_path, multiclass_data) -> str:
    """Multiclass data saved as CSV. Returns file path."""
    path = tmp_path / "multiclass.csv"
    multiclass_data.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def regression_csv(tmp_path, regression_data) -> str:
    """Regression data saved as CSV. Returns file path."""
    path = tmp_path / "regression.csv"
    regression_data.to_csv(path, index=False)
    return str(path)
