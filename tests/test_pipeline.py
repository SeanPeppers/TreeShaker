"""End-to-end pipeline tests."""

from unittest.mock import patch

import pandas as pd
import pytest

from treeshaker.config import BootstrapConfig, Config, OptunaConfig, XGBoostConfig
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
        "mutual_info_score",
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


def test_no_groups_by_default(binary_csv, tmp_path):
    """With no group config, results schema is unchanged and no group file."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert set(results.columns) == {
        "feature_name",
        "shap_importance",
        "permutation_importance",
        "mutual_info_score",
        "recommendation",
        "reason",
    }
    assert pipeline.group_summary is None
    groups_file = tmp_path / "results_groups.csv"
    assert not groups_file.exists()


def test_explicit_feature_groups(binary_csv, tmp_path):
    config = make_fast_config(binary_csv, "binary", tmp_path)
    config.feature_groups = {
        "group_a": ["feature_0", "feature_1"],
        "group_b": ["feature_2", "feature_3"],
    }
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    # Per-feature schema is unchanged.
    assert set(results.columns) == {
        "feature_name",
        "shap_importance",
        "permutation_importance",
        "mutual_info_score",
        "recommendation",
        "reason",
    }

    assert pipeline.group_summary is not None
    summary = pipeline.group_summary
    assert set(summary.columns) == {
        "group_name",
        "n_features",
        "total_shap_importance",
        "features",
    }
    assert set(summary["group_name"]) == {"group_a", "group_b"}
    assert (summary["n_features"] == 2).all()

    groups_file = tmp_path / "results_groups.csv"
    assert groups_file.exists()
    saved = pd.read_csv(groups_file)
    assert set(saved["group_name"]) == {"group_a", "group_b"}

    # Sanity check: total_shap_importance for a group equals the sum of its
    # members' shap_importance in the per-feature results.
    shap_by_feature = dict(
        zip(results["feature_name"], results["shap_importance"], strict=True)
    )
    group_a_row = summary[summary["group_name"] == "group_a"].iloc[0]
    expected_total = shap_by_feature["feature_0"] + shap_by_feature["feature_1"]
    assert group_a_row["total_shap_importance"] == pytest.approx(expected_total)


def test_group_by_prefix_auto_derivation(regression_csv, tmp_path):
    """Prefix-derived groups require renaming columns to share a prefix."""
    df = pd.read_csv(regression_csv)
    rename_map = {
        "feature_0": "price_min",
        "feature_1": "price_max",
        "feature_2": "price_avg",
    }
    df = df.rename(columns=rename_map)
    renamed_path = tmp_path / "renamed.csv"
    df.to_csv(renamed_path, index=False)

    config = make_fast_config(str(renamed_path), "regression", tmp_path)
    config.group_by_prefix = True
    pipeline = FeatureSelectionPipeline(config)
    pipeline.run()

    assert pipeline.group_summary is not None
    summary = pipeline.group_summary
    assert "price" in set(summary["group_name"])
    price_row = summary[summary["group_name"] == "price"].iloc[0]
    assert price_row["n_features"] == 3
    assert set(price_row["features"].split(", ")) == {
        "price_min",
        "price_max",
        "price_avg",
    }

    groups_file = tmp_path / "results_groups.csv"
    assert groups_file.exists()


def test_group_by_prefix_ignored_when_no_underscore(tmp_path, binary_data):
    """Columns with no underscore can't form a prefix group."""
    df = binary_data.rename(columns=lambda c: c.replace("feature_", "f"))
    path = tmp_path / "nounderscore.csv"
    df.to_csv(path, index=False)
    config = make_fast_config(str(path), "binary", tmp_path)
    config.group_by_prefix = True
    pipeline = FeatureSelectionPipeline(config)
    pipeline.run()
    assert pipeline.group_summary is None


def test_pipeline_no_bootstrap_by_default(binary_csv, tmp_path):
    """n_bootstrap=0 (the default) must not change the output schema."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    assert config.bootstrap_config.n_bootstrap == 0
    results = FeatureSelectionPipeline(config).run()
    assert set(results.columns) == {
        "feature_name",
        "shap_importance",
        "permutation_importance",
        "mutual_info_score",
        "recommendation",
        "reason",
    }


def make_fast_bootstrap_config(input_file, task, tmp_path, n_bootstrap=3):
    """Config with bootstrap importance enabled, kept small/fast for tests."""
    return Config(
        input_file=input_file,
        target_column="target",
        task=task,
        output_file=str(tmp_path / "results.csv"),
        output_format="csv",
        optuna_config=OptunaConfig(n_trials=3, timeout=60),
        xgboost_config=XGBoostConfig(max_depth=2, extra_params={"n_estimators": 5}),
        bootstrap_config=BootstrapConfig(n_bootstrap=n_bootstrap),
    )


def test_pipeline_bootstrap_adds_columns(binary_csv, tmp_path):
    config = make_fast_bootstrap_config(binary_csv, "binary", tmp_path)
    results = FeatureSelectionPipeline(config).run()
    assert set(results.columns) == {
        "feature_name",
        "shap_importance",
        "permutation_importance",
        "mutual_info_score",
        "recommendation",
        "reason",
        "shap_importance_mean",
        "shap_importance_std",
    }
    assert len(results) == 10
    assert (results["shap_importance_mean"] >= 0).all()
    assert (results["shap_importance_std"] >= 0).all()


def test_pipeline_bootstrap_shapes(binary_csv, tmp_path):
    config = make_fast_bootstrap_config(binary_csv, "binary", tmp_path, n_bootstrap=3)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert results["shap_importance_mean"].shape == (10,)
    assert results["shap_importance_std"].shape == (10,)


def test_pipeline_bootstrap_regression(regression_csv, tmp_path):
    config = make_fast_bootstrap_config(regression_csv, "regression", tmp_path)
    results = FeatureSelectionPipeline(config).run()
    assert "shap_importance_mean" in results.columns
    assert "shap_importance_std" in results.columns
    assert len(results) == 20


def test_skip_tuning_off_by_default(binary_csv, tmp_path):
    """skip_tuning defaults to False, matching existing behavior."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    assert config.skip_tuning is False


def test_pipeline_skip_tuning_runs_without_optuna(binary_csv, tmp_path):
    """With skip_tuning=True, the pipeline completes and never calls Optuna."""
    config = Config(
        input_file=binary_csv,
        target_column="target",
        task="binary",
        output_file=str(tmp_path / "results.csv"),
        output_format="csv",
        xgboost_config=XGBoostConfig(max_depth=2, extra_params={"n_estimators": 5}),
        skip_tuning=True,
    )
    pipeline = FeatureSelectionPipeline(config)

    with patch("optuna.create_study") as mock_create_study:
        results = pipeline.run()

    mock_create_study.assert_not_called()
    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == {
        "feature_name",
        "shap_importance",
        "permutation_importance",
        "mutual_info_score",
        "recommendation",
        "reason",
    }
    assert len(results) == 10


def test_pipeline_skip_tuning_uses_configured_xgboost_params(binary_csv, tmp_path):
    """skip_tuning trains directly from XGBoostConfig, not Optuna's best_params."""
    xgb_cfg = XGBoostConfig(max_depth=3, extra_params={"n_estimators": 5})
    config = Config(
        input_file=binary_csv,
        target_column="target",
        task="binary",
        output_file=str(tmp_path / "results.csv"),
        output_format="csv",
        xgboost_config=xgb_cfg,
        skip_tuning=True,
    )
    pipeline = FeatureSelectionPipeline(config)
    pipeline.run()

    assert pipeline.model.get_params()["max_depth"] == 3
    assert pipeline.model.get_params()["n_estimators"] == 5
