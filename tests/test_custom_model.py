"""Tests for the custom_model passthrough (Python-API-only Config option)."""

import pandas as pd
import pytest
from conftest import BASE_RESULT_COLUMNS
from conftest import make_config as make_fast_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from treeshaker.config import Config
from treeshaker.pipeline import FeatureSelectionPipeline


def test_custom_model_default_is_none():
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.custom_model is None


def test_default_behavior_unaffected_without_custom_model(binary_csv, tmp_path):
    """No custom_model set -> existing XGBoost path, unchanged results schema."""
    config = make_fast_config(binary_csv, "binary", tmp_path, skip_tuning=True)
    results = FeatureSelectionPipeline(config).run()
    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == BASE_RESULT_COLUMNS
    assert len(results) == 10


def test_custom_model_runs_end_to_end(binary_csv, tmp_path):
    """A plain sklearn estimator runs the full pipeline and produces results."""
    config = make_fast_config(
        binary_csv,
        "binary",
        tmp_path,
        skip_tuning=True,
        custom_model=RandomForestClassifier(n_estimators=10, random_state=42),
    )
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == BASE_RESULT_COLUMNS
    assert len(results) == 10
    assert isinstance(pipeline.model, RandomForestClassifier)
    # The clone must be fitted but distinct from the original, unfitted object.
    assert pipeline.model is not config.custom_model
    assert not hasattr(config.custom_model, "estimators_")


def test_custom_model_missing_methods_raises_value_error(binary_csv, tmp_path):
    class NotAnEstimator:
        pass

    config = make_fast_config(
        binary_csv,
        "binary",
        tmp_path,
        skip_tuning=True,
        custom_model=NotAnEstimator(),
    )
    pipeline = FeatureSelectionPipeline(config)
    with pytest.raises(ValueError, match="missing required method"):
        pipeline.run()


def test_custom_model_missing_predict_only(binary_csv, tmp_path):
    class FitOnly:
        def fit(self, X, y):
            return self

    config = make_fast_config(
        binary_csv,
        "binary",
        tmp_path,
        skip_tuning=True,
        custom_model=FitOnly(),
    )
    pipeline = FeatureSelectionPipeline(config)
    with pytest.raises(ValueError, match="predict"):
        pipeline.run()


def test_custom_model_shap_degrades_gracefully_for_non_tree_model(binary_csv, tmp_path):
    """Non-tree custom_model (LogisticRegression) must not crash the pipeline."""
    config = make_fast_config(
        binary_csv,
        "binary",
        tmp_path,
        skip_tuning=True,
        custom_model=LogisticRegression(max_iter=200),
    )
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == BASE_RESULT_COLUMNS
    assert len(results) == 10
    # shap_importance is either computed via the model-agnostic fallback, or
    # all zeros if SHAP failed entirely -- either way, no NaNs/crash.
    assert results["shap_importance"].notna().all()
