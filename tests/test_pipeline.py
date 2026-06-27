"""End-to-end pipeline tests."""

import pandas as pd

from treeshaker.config import Config, OptunaConfig
from treeshaker.pipeline import FeatureSelectionPipeline


def make_fast_config(input_file, task, tmp_path):
    """Return a Config set up for fast test runs (5 Optuna trials)."""
    return Config(
        input_file=input_file,
        target_column="target",
        task=task,
        output_file=str(tmp_path / "results.csv"),
        output_format="csv",
        optuna_config=OptunaConfig(n_trials=5, timeout=60),
    )


def test_pipeline_binary_run(binary_csv, tmp_path):
    config = make_fast_config(binary_csv, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == {
        "feature_name",
        "shap_importance",
        "permutation_importance",
        "recommendation",
        "reason",
    }
    assert len(results) == 10


def test_pipeline_regression_run(regression_csv, tmp_path):
    config = make_fast_config(regression_csv, "regression", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert len(results) == 20
    assert results["shap_importance"].is_monotonic_decreasing


def test_pipeline_recommendation_values(binary_csv, tmp_path):
    config = make_fast_config(binary_csv, "binary", tmp_path)
    results = FeatureSelectionPipeline(config).run()
    valid = {"Drop", "Caution", "Keep"}
    assert results["recommendation"].isin(valid).all()


def test_pipeline_saves_csv(binary_csv, tmp_path):
    output = tmp_path / "out.csv"
    config = Config(
        input_file=binary_csv,
        target_column="target",
        task="binary",
        output_file=str(output),
        optuna_config=OptunaConfig(n_trials=3, timeout=30),
    )
    FeatureSelectionPipeline(config).run()
    assert output.exists()
    df = pd.read_csv(output)
    assert "feature_name" in df.columns
    assert "recommendation" in df.columns


def test_pipeline_saves_html(binary_csv, tmp_path):
    output = tmp_path / "report.html"
    config = Config(
        input_file=binary_csv,
        target_column="target",
        task="binary",
        output_file=str(output),
        output_format="html",
        optuna_config=OptunaConfig(n_trials=3, timeout=30),
    )
    FeatureSelectionPipeline(config).run()
    assert output.exists()
    content = output.read_text()
    assert "<table" in content


def test_pipeline_multiclass_run(multiclass_csv, tmp_path):
    config = make_fast_config(multiclass_csv, "multiclass", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert len(results) == 15


def test_keep_reason_is_empty(binary_csv, tmp_path):
    config = make_fast_config(binary_csv, "binary", tmp_path)
    results = FeatureSelectionPipeline(config).run()
    keep_rows = results[results["recommendation"] == "Keep"]
    assert (keep_rows["reason"] == "").all()
