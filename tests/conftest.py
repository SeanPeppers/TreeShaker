"""Shared pytest fixtures for TreeShaker tests."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from treeshaker.config import Config

# The per-feature results schema produced by a pipeline run with no optional
# columns enabled (no bootstrap, no residual diagnostics, no feature groups).
# Tests that enable one of those features assert against this set plus the
# extra columns that feature adds.
BASE_RESULT_COLUMNS = {
    "feature_name",
    "shap_importance",
    "permutation_importance",
    "permutation_importance_std",
    "mutual_info_score",
    "recommendation",
    "reason",
}


def make_config(input_file: str, task: str, tmp_path, **overrides) -> Config:
    """Build a Config for test runs, defaulting to fast/deterministic settings.

    Centralizes the input/target/output boilerplate shared by nearly every
    pipeline-level test. Pass any Config field as a keyword override, e.g.
    ``make_config(csv, "binary", tmp_path, skip_tuning=True)`` or
    ``make_config(csv, "binary", tmp_path, bootstrap_config=BootstrapConfig(...))``.

    Defaults to 5 Optuna trials (fast tuning) unless ``skip_tuning=True`` is
    passed, or an explicit ``optuna_config`` override is given.
    """
    kwargs: dict = {
        "input_file": input_file,
        "target_column": "target",
        "task": task,
        "output_file": str(tmp_path / "results.csv"),
        "output_format": "csv",
    }
    if "optuna_config" not in overrides and not overrides.get("skip_tuning"):
        from treeshaker.config import OptunaConfig

        kwargs["optuna_config"] = OptunaConfig(n_trials=5, timeout=60)
    kwargs.update(overrides)
    return Config(**kwargs)


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


def cli_argv(
    *extra: str,
    input: str = "f.csv",
    target: str = "y",
    task: str = "binary",
) -> list:
    """Build a ``sys.argv``-shaped list for CLI tests.

    Every CLI test patches ``sys.argv`` with ``["treeshaker", "--input", ...,
    "--target", ..., "--task", ...]`` followed by whatever extra flags that
    test cares about. This centralizes the common prefix; pass additional
    flag/value strings positionally, e.g. ``cli_argv("--device", "cpu")``.
    """
    return [
        "treeshaker",
        "--input",
        input,
        "--target",
        target,
        "--task",
        task,
        *extra,
    ]


def _patch_shap_tree_explainer_spy(monkeypatch, on_call):
    """Patch ``treeshaker.importance.shap.TreeExplainer`` with a spy that
    calls ``on_call(X, kwargs)`` on every ``shap_values`` invocation, while
    still delegating to the real TreeExplainer for the computation.

    Shared by ``shap_row_count_spy`` and ``shap_additivity_kwarg_spy``,
    which otherwise each wrap the same ``TreeExplainer.__init__``/
    ``shap_values`` delegation in their own near-identical class -- this
    factors out that wrapper, leaving each fixture to decide only what to
    record from a call.
    """
    import shap

    original_tree_explainer = shap.TreeExplainer

    class _SpyExplainer:
        def __init__(self, m):
            self._inner = original_tree_explainer(m)

        def shap_values(self, X, **kwargs):
            on_call(X, kwargs)
            return self._inner.shap_values(X, **kwargs)

    monkeypatch.setattr("treeshaker.importance.shap.TreeExplainer", _SpyExplainer)


@pytest.fixture
def shap_row_count_spy(monkeypatch):
    """Patch ``treeshaker.importance.shap.TreeExplainer`` with a spy that
    records the number of rows passed to ``shap_values`` on each call,
    while still delegating to the real TreeExplainer for the computation.

    Returns the ``seen_rows`` list, appended to on every ``shap_values`` call.
    """
    seen_rows: list = []
    _patch_shap_tree_explainer_spy(
        monkeypatch, on_call=lambda X, kwargs: seen_rows.append(len(X))
    )
    return seen_rows


@pytest.fixture
def shap_kwargs_spy(monkeypatch):
    """Patch ``treeshaker.importance.shap.TreeExplainer`` with a spy that
    records the kwargs passed to ``shap_values`` on each call (e.g.
    ``check_additivity``), while still delegating to the real
    TreeExplainer for the computation.

    Returns the ``seen_kwargs`` dict, updated with the kwargs from the
    most recent ``shap_values`` call.
    """
    seen_kwargs: dict = {}
    _patch_shap_tree_explainer_spy(
        monkeypatch, on_call=lambda X, kwargs: seen_kwargs.update(kwargs)
    )
    return seen_kwargs
