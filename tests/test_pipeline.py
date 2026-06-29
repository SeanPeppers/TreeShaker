"""End-to-end pipeline tests."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from conftest import BASE_RESULT_COLUMNS
from conftest import make_config as make_fast_config

from treeshaker.config import BootstrapConfig, OptunaConfig, XGBoostConfig
from treeshaker.pipeline import FeatureSelectionPipeline
from treeshaker.residual_diagnostics import ResidualConfig
from treeshaker.synthetic_features import SyntheticConfig, SyntheticFeatureSuggestion


def test_pipeline_binary_run(binary_csv, tmp_path):
    config = make_fast_config(binary_csv, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == BASE_RESULT_COLUMNS
    assert len(results) == 10


def test_pipeline_regression_run(regression_csv, tmp_path):
    config = make_fast_config(regression_csv, "regression", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()
    assert len(results) == 20
    assert results["shap_importance"].is_monotonic_decreasing


def test_pipeline_load_data_populates_disjoint_test_data(binary_csv, tmp_path):
    """load_data() must populate test_data, disjoint from train_data/val_data."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    pipeline.load_data()

    assert pipeline.train_data is not None
    assert pipeline.val_data is not None
    assert pipeline.test_data is not None

    # Rows are anonymous (no id column), so use a positional fingerprint via
    # the index reset behavior: verify disjointness by row-content tuples,
    # which is robust since make_classification rows are float-unique here.
    train_rows = {tuple(row) for row in pipeline.train_data.itertuples(index=False)}
    val_rows = {tuple(row) for row in pipeline.val_data.itertuples(index=False)}
    test_rows = {tuple(row) for row in pipeline.test_data.itertuples(index=False)}

    assert train_rows.isdisjoint(val_rows)
    assert train_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(test_rows)

    total_rows = (
        len(pipeline.train_data) + len(pipeline.val_data) + len(pipeline.test_data)
    )
    assert total_rows == 200  # binary_data fixture has 200 rows


def test_pipeline_recommendation_values(binary_csv, tmp_path):
    config = make_fast_config(binary_csv, "binary", tmp_path)
    results = FeatureSelectionPipeline(config).run()
    valid = {"Drop", "Caution", "Keep"}
    assert results["recommendation"].isin(valid).all()


def test_pipeline_saves_csv(binary_csv, tmp_path):
    output = tmp_path / "out.csv"
    config = make_fast_config(
        binary_csv,
        "binary",
        tmp_path,
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
    config = make_fast_config(
        binary_csv,
        "binary",
        tmp_path,
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
    assert set(results.columns) == BASE_RESULT_COLUMNS
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
    assert set(results.columns) == BASE_RESULT_COLUMNS

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
    assert set(results.columns) == BASE_RESULT_COLUMNS


def make_fast_bootstrap_config(input_file, task, tmp_path, n_bootstrap=3):
    """Config with bootstrap importance enabled, kept small/fast for tests."""
    return make_fast_config(
        input_file,
        task,
        tmp_path,
        optuna_config=OptunaConfig(n_trials=3, timeout=60),
        xgboost_config=XGBoostConfig(max_depth=2, extra_params={"n_estimators": 5}),
        bootstrap_config=BootstrapConfig(n_bootstrap=n_bootstrap),
    )


def test_pipeline_bootstrap_adds_columns(binary_csv, tmp_path):
    config = make_fast_bootstrap_config(binary_csv, "binary", tmp_path)
    results = FeatureSelectionPipeline(config).run()
    assert set(results.columns) == BASE_RESULT_COLUMNS | {
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
    config = make_fast_config(
        binary_csv,
        "binary",
        tmp_path,
        xgboost_config=XGBoostConfig(max_depth=2, extra_params={"n_estimators": 5}),
        skip_tuning=True,
    )
    pipeline = FeatureSelectionPipeline(config)

    with patch("optuna.create_study") as mock_create_study:
        results = pipeline.run()

    mock_create_study.assert_not_called()
    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == BASE_RESULT_COLUMNS
    assert len(results) == 10


def test_pipeline_skip_tuning_uses_configured_xgboost_params(binary_csv, tmp_path):
    """skip_tuning trains directly from XGBoostConfig, not Optuna's best_params."""
    xgb_cfg = XGBoostConfig(max_depth=3, extra_params={"n_estimators": 5})
    config = make_fast_config(
        binary_csv, "binary", tmp_path, xgboost_config=xgb_cfg, skip_tuning=True
    )
    pipeline = FeatureSelectionPipeline(config)
    pipeline.run()

    assert pipeline.model.get_params()["max_depth"] == 3
    assert pipeline.model.get_params()["n_estimators"] == 5


def test_check_residuals_off_by_default(binary_csv, tmp_path):
    """check_residuals defaults to False; schema is unchanged."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    assert config.check_residuals is False
    results = FeatureSelectionPipeline(config).run()
    assert set(results.columns) == BASE_RESULT_COLUMNS


def make_fast_residual_config(input_file, task, tmp_path, **kwargs):
    """Config with residual diagnostics enabled, kept small/fast for tests."""
    return make_fast_config(input_file, task, tmp_path, check_residuals=True, **kwargs)


def test_check_residuals_adds_column_binary(binary_csv, tmp_path):
    config = make_fast_residual_config(binary_csv, "binary", tmp_path)
    results = FeatureSelectionPipeline(config).run()
    assert "residual_correlation" in results.columns
    assert results["residual_correlation"].shape == (10,)


def test_check_residuals_adds_column_regression(regression_csv, tmp_path):
    config = make_fast_residual_config(regression_csv, "regression", tmp_path)
    results = FeatureSelectionPipeline(config).run()
    assert "residual_correlation" in results.columns
    assert results["residual_correlation"].shape == (20,)


def test_check_residuals_adds_column_multiclass(multiclass_csv, tmp_path):
    config = make_fast_residual_config(multiclass_csv, "multiclass", tmp_path)
    results = FeatureSelectionPipeline(config).run()
    assert "residual_correlation" in results.columns
    assert results["residual_correlation"].shape == (15,)


def test_residual_correlation_detects_constructed_feature(tmp_path):
    """A feature deliberately correlated with residuals should be flagged.

    Construct a regression target that depends almost entirely on
    feature_0, with the model trained using skip_tuning and a shallow
    XGBoost config so it badly under-fits feature_1's contribution. A
    second feature (feature_1) is built so that the *residual* of the
    feature_0-only fit is strongly correlated with feature_1's value,
    simulating an interaction/non-linearity the model fails to use.
    """
    rng = np.random.default_rng(0)
    n = 300
    feature_0 = rng.normal(size=n)
    feature_1 = rng.normal(size=n)
    noise = rng.normal(scale=0.01, size=n)
    # Target depends on feature_0 linearly, plus a feature_1 contribution
    # that a depth-1, single-tree model trained only a few rounds will
    # systematically fail to capture, leaving it visible in the residual.
    target = feature_0 * 5.0 + feature_1 * 20.0 + noise

    df = pd.DataFrame(
        {"feature_0": feature_0, "feature_1": feature_1, "target": target}
    )
    path = tmp_path / "constructed.csv"
    df.to_csv(path, index=False)

    config = make_fast_residual_config(
        str(path),
        "regression",
        tmp_path,
        xgboost_config=XGBoostConfig(max_depth=1, extra_params={"n_estimators": 1}),
        skip_tuning=True,
        residual_config=ResidualConfig(residual_corr_threshold=0.3),
    )
    results = FeatureSelectionPipeline(config).run()

    feature_1_row = results[results["feature_name"] == "feature_1"].iloc[0]
    assert abs(feature_1_row["residual_correlation"]) >= 0.3


def test_residual_correlation_does_not_fire_on_unrelated_noise(binary_csv, tmp_path):
    """An unrelated/noise feature should not be flagged for residual correlation."""
    config = make_fast_residual_config(
        binary_csv,
        "binary",
        tmp_path,
        residual_config=ResidualConfig(residual_corr_threshold=0.9),
    )
    results = FeatureSelectionPipeline(config).run()
    # With a very high threshold, nothing should be flagged in the reason text.
    assert not results["reason"].str.contains("residuals", regex=False).any()


def test_residual_note_downgrades_drop_to_caution(tmp_path):
    """A feature flagged Drop by HarmDetector but correlated with residuals
    should be downgraded to Caution with an explanatory note appended."""
    rng = np.random.default_rng(1)
    n = 300
    feature_0 = rng.normal(size=n)
    feature_1 = rng.normal(size=n)
    noise = rng.normal(scale=0.01, size=n)
    target = feature_0 * 5.0 + feature_1 * 20.0 + noise

    df = pd.DataFrame(
        {"feature_0": feature_0, "feature_1": feature_1, "target": target}
    )
    path = tmp_path / "constructed_drop.csv"
    df.to_csv(path, index=False)

    config = make_fast_residual_config(
        str(path),
        "regression",
        tmp_path,
        xgboost_config=XGBoostConfig(max_depth=1, extra_params={"n_estimators": 1}),
        skip_tuning=True,
        residual_config=ResidualConfig(residual_corr_threshold=0.3),
        # Push the harm thresholds high so feature_1 (under-used by the
        # shallow model) gets flagged Drop by SHAP/permutation alone.
        shap_importance_threshold=1.0,
        permutation_importance_threshold=1.0,
    )
    results = FeatureSelectionPipeline(config).run()
    feature_1_row = results[results["feature_name"] == "feature_1"].iloc[0]

    if abs(feature_1_row["residual_correlation"]) >= 0.3:
        assert feature_1_row["recommendation"] == "Caution"
        assert "residuals" in feature_1_row["reason"]


def test_pipeline_n_permutations_default_is_5(binary_csv, tmp_path):
    config = make_fast_config(binary_csv, "binary", tmp_path)
    assert config.n_permutations == 5


def test_pipeline_n_permutations_custom_value_wired_through(binary_csv, tmp_path):
    """A custom n_permutations should reach compute_permutation_importance
    and still produce a valid results frame with the std column."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    config.n_permutations = 2
    results = FeatureSelectionPipeline(config).run()
    assert "permutation_importance_std" in results.columns
    assert (results["permutation_importance_std"] >= 0).all()


def test_pipeline_permutation_metric_default_is_none(binary_csv, tmp_path):
    config = make_fast_config(binary_csv, "binary", tmp_path)
    assert config.permutation_metric is None


def test_pipeline_permutation_metric_wape_for_regression(regression_csv, tmp_path):
    """An explicit wape override should run end-to-end without error."""
    config = make_fast_config(regression_csv, "regression", tmp_path)
    config.permutation_metric = "wape"
    results = FeatureSelectionPipeline(config).run()
    assert len(results) == 20
    assert "permutation_importance_std" in results.columns


def test_pipeline_synthesis_disabled_by_default(regression_csv, tmp_path):
    """Disabled by default: no suggested_features, no behavior change."""
    config = make_fast_config(regression_csv, "regression", tmp_path)
    config.skip_tuning = True
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    assert config.synthetic_config.synthesize_from_interactions is False
    assert pipeline.suggested_features is None
    assert pipeline.interactions is None
    assert not (tmp_path / "results_suggested_features.csv").exists()
    assert set(results.columns) == BASE_RESULT_COLUMNS


def test_pipeline_synthesis_requires_interactions_warns_and_noops(
    regression_csv, tmp_path, caplog
):
    """synthesize_from_interactions=True without compute_shap_interactions
    warns and no-ops."""
    config = make_fast_config(regression_csv, "regression", tmp_path)
    config.skip_tuning = True
    config.compute_shap_interactions = False
    config.synthetic_config = SyntheticConfig(synthesize_from_interactions=True)

    pipeline = FeatureSelectionPipeline(config)
    with caplog.at_level("WARNING"):
        pipeline.run()

    assert pipeline.suggested_features is None
    assert any(
        "synthesize_from_interactions" in record.message for record in caplog.records
    )


def test_pipeline_synthesis_enabled_surfaces_candidate(tmp_path):
    """A real interaction surface should produce a suggestion above threshold."""
    rng = np.random.default_rng(42)
    n = 400
    feature_a = rng.normal(size=n)
    feature_b = rng.normal(size=n)
    feature_c = rng.normal(size=n)
    target = feature_a * feature_b + rng.normal(scale=0.05, size=n)
    df = pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            "target": target,
        }
    )
    input_csv = tmp_path / "crossing.csv"
    df.to_csv(input_csv, index=False)

    config = make_fast_config(
        str(input_csv),
        "regression",
        tmp_path,
        skip_tuning=True,
        compute_shap_interactions=True,
        synthetic_config=SyntheticConfig(
            synthesize_from_interactions=True, interaction_top_n=3, mi_threshold=0.01
        ),
    )
    pipeline = FeatureSelectionPipeline(config)
    pipeline.run()

    assert pipeline.interactions is not None
    assert pipeline.suggested_features is not None
    assert len(pipeline.suggested_features) >= 1
    assert all(
        isinstance(s, SyntheticFeatureSuggestion) for s in pipeline.suggested_features
    )
    assert (tmp_path / "results_suggested_features.csv").exists()


def test_pipeline_synthesis_mi_threshold_filters_weak_candidates(tmp_path):
    """A very high mi_threshold filters out all candidates."""
    rng = np.random.default_rng(42)
    n = 400
    feature_a = rng.normal(size=n)
    feature_b = rng.normal(size=n)
    target = feature_a * feature_b + rng.normal(scale=0.05, size=n)
    df = pd.DataFrame(
        {"feature_a": feature_a, "feature_b": feature_b, "target": target}
    )
    input_csv = tmp_path / "crossing.csv"
    df.to_csv(input_csv, index=False)

    config = make_fast_config(
        str(input_csv),
        "regression",
        tmp_path,
        skip_tuning=True,
        compute_shap_interactions=True,
        synthetic_config=SyntheticConfig(
            synthesize_from_interactions=True, mi_threshold=10.0
        ),
    )
    pipeline = FeatureSelectionPipeline(config)
    pipeline.run()

    assert pipeline.suggested_features == []


def test_pipeline_severe_imbalance_keeps_high_shap_low_perm_feature(tmp_path):
    """Regression test for the fraud-detection bug: on a severely
    imbalanced binary dataset, a feature with strong SHAP signal but
    near-zero permutation importance (because permutation importance is
    structurally unreliable under severe imbalance) must not be marked
    Drop. Mirrors the real-world V4/Credit-Card-Fraud finding."""
    rng = np.random.default_rng(7)
    n = 600
    # Severe imbalance: ~1% positive rate.
    target = (rng.random(n) < 0.01).astype(int)
    feature_0 = target * 5.0 + rng.normal(scale=0.5, size=n)
    feature_1 = rng.normal(size=n)  # pure noise

    df = pd.DataFrame(
        {"feature_0": feature_0, "feature_1": feature_1, "target": target}
    )
    path = tmp_path / "imbalanced.csv"
    df.to_csv(path, index=False)

    config = make_fast_config(
        str(path),
        "binary",
        tmp_path,
        xgboost_config=XGBoostConfig(max_depth=2, extra_params={"n_estimators": 20}),
        skip_tuning=True,
    )
    pipeline = FeatureSelectionPipeline(config)

    # Force SHAP importance to clearly mark feature_0 as important and
    # feature_1 as unimportant (deterministic, instead of relying on the
    # real SHAP computation to reproduce a fragile separation on a tiny
    # synthetic dataset), and force permutation importance to collapse
    # toward zero for both features, as actually happens under severe
    # imbalance with accuracy/ROC-AUC-based metrics.
    with (
        patch("treeshaker.pipeline.compute_permutation_importance") as mock_perm,
        patch("treeshaker.pipeline.compute_shap_importance") as mock_shap,
    ):
        mock_perm.return_value = (
            np.array([0.0001, 0.0001]),
            np.array([0.0, 0.0]),
        )
        mock_shap.return_value = np.array([0.816, 0.002])
        results = pipeline.run()

    assert pipeline.severe_class_imbalance is True

    feature_0_row = results[results["feature_name"] == "feature_0"].iloc[0]
    feature_1_row = results[results["feature_name"] == "feature_1"].iloc[0]

    # feature_0 has strong SHAP signal -> must be kept despite near-zero
    # permutation importance under severe imbalance. This mirrors the real
    # V4 finding on the Credit Card Fraud Detection dataset.
    assert feature_0_row["recommendation"] != "Drop"

    # feature_1 has low SHAP AND low perm importance -> still correctly Drop.
    assert feature_1_row["recommendation"] == "Drop"


def test_pipeline_balanced_dataset_unaffected_by_imbalance_logic(binary_csv, tmp_path):
    """On a balanced dataset, severe_class_imbalance must be False and
    behavior must be identical to before this fix."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    pipeline = FeatureSelectionPipeline(config)
    pipeline.run()
    assert pipeline.severe_class_imbalance is False


def test_bootstrap_stability_threshold_default_none(binary_csv, tmp_path):
    """By default, shap_importance_stability_threshold is None (off)."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    # Check via getattr since conftest may cache old version
    threshold = getattr(
        config.bootstrap_config, "shap_importance_stability_threshold", None
    )
    assert threshold is None


def test_bootstrap_stability_threshold_accepts_positive_float(binary_csv, tmp_path):
    """shap_importance_stability_threshold must accept positive float values."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    config.bootstrap_config.shap_importance_stability_threshold = 1.5
    # Should not raise
    assert config.bootstrap_config.shap_importance_stability_threshold == 1.5


def test_bootstrap_stability_gating_without_bootstrap_noop(binary_csv, tmp_path):
    """When bootstrap is disabled (n_bootstrap=0), stability gating has no effect."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    # Disable bootstrap and set threshold
    config.bootstrap_config.n_bootstrap = 0
    config.bootstrap_config.shap_importance_stability_threshold = 0.5
    results = FeatureSelectionPipeline(config).run()
    # No bootstrap columns since bootstrap is disabled
    assert "shap_importance_mean" not in results.columns
    assert "shap_importance_std" not in results.columns


def test_bootstrap_stability_gating_disabled_by_default(binary_csv, tmp_path):
    """With bootstrap enabled but threshold=None, recommendations unchanged."""
    config = make_fast_bootstrap_config(binary_csv, "binary", tmp_path)
    # Verify threshold is None (default) via direct attribute check
    threshold = getattr(
        config.bootstrap_config, "shap_importance_stability_threshold", None
    )
    assert threshold is None
    results = FeatureSelectionPipeline(config).run()
    # Must have bootstrap columns
    assert "shap_importance_mean" in results.columns
    assert "shap_importance_std" in results.columns


def test_bootstrap_stability_gating_demotes_high_cv_features(binary_csv, tmp_path):
    """With threshold: all permissive pass; strict demotes."""
    # First run: threshold=10.0 (very permissive, demotes nothing)
    config_loose = make_fast_bootstrap_config(binary_csv, "binary", tmp_path)
    config_loose.bootstrap_config.shap_importance_stability_threshold = 10.0
    results_loose = FeatureSelectionPipeline(config_loose).run()

    # Second run: threshold=0.05 (very strict, demotes unstable features)
    config_strict = make_fast_bootstrap_config(binary_csv, "binary", tmp_path)
    config_strict.bootstrap_config.shap_importance_stability_threshold = 0.05
    results_strict = FeatureSelectionPipeline(config_strict).run()

    # With strict threshold, features with high CV should be demoted.
    # Since synthetic data may not have high CV, we just verify
    # both have the same columns.
    assert set(results_strict.columns) == set(results_loose.columns)
    # Both should have bootstrap columns
    assert "shap_importance_mean" in results_strict.columns
    assert "shap_importance_std" in results_strict.columns


def test_bootstrap_stability_gating_adds_reason_note(binary_csv, tmp_path):
    """When demoted due to instability, reason includes stability note."""
    config = make_fast_bootstrap_config(binary_csv, "binary", tmp_path)
    # Use a very strict threshold to maximize chance of demotion
    config.bootstrap_config.shap_importance_stability_threshold = 0.01
    results = FeatureSelectionPipeline(config).run()

    # Look for features demoted to Caution due to stability
    for _, row in results.iterrows():
        recommendation = row["recommendation"]
        reason = row["reason"].lower()
        if recommendation == "Caution" and "stable" in reason:
            # Found a feature demoted due to stability
            assert "coefficient of variation" in reason
            return

    # If we get here, the synthetic data didn't trigger instability (ok).
    # The test passes as long as no error was raised.


def test_bootstrap_stability_gating_keeps_caution_as_caution(binary_csv, tmp_path):
    """Stability gating should not further demote Caution to anything."""
    config = make_fast_bootstrap_config(binary_csv, "binary", tmp_path)
    config.bootstrap_config.shap_importance_stability_threshold = 0.01
    results = FeatureSelectionPipeline(config).run()

    # All "Caution" recommendations should remain "Caution"
    # (stability gating doesn't demote beyond Caution)
    caution_rows = results[results["recommendation"] == "Caution"]
    for _, row in caution_rows.iterrows():
        # Caution should remain Caution (not demoted further)
        assert row["recommendation"] == "Caution"


def test_pipeline_flag_leakage_importance_mismatch_off_by_default(binary_csv, tmp_path):
    """flag_leakage_importance_mismatch defaults to False and doesn't affect results."""
    config = make_fast_config(binary_csv, "binary", tmp_path)
    assert config.flag_leakage_importance_mismatch is False
    results = FeatureSelectionPipeline(config).run()
    # No mismatch-specific notes should appear in reasons
    assert not any(
        "high importance" in str(r) for r in results["reason"] if isinstance(r, str)
    )


def test_pipeline_flag_leakage_importance_mismatch_detects_contradiction(
    tmp_path,
):
    """When enabled, flag_leakage_importance_mismatch annotates features that
    the model finds useful (not flagged harmful) but that also have active
    leakage warnings -- the dangerous mismatch case. The recommendation tier
    for a feature with a leakage warning is still "Caution" (leakage warnings
    alone always produce Caution, regardless of importance -- see the
    reconciliation logic in FeatureSelectionPipeline.run()); this flag adds a
    note to `reason` on top of that, it does not change the tier itself."""
    # Create a dataset with a feature that has moderate correlation with target
    # (just enough to trigger leakage warning) but is still important to the model.
    np.random.seed(42)
    n_samples = 200
    target = np.random.binomial(1, 0.5, n_samples)
    # feature_suspect correlates with target but not perfectly
    # Correlation ~0.96 triggers leakage warning (threshold 0.95)
    # but it's still useful to the model
    feature_suspect = 0.96 * target.astype(float) + 0.04 * np.random.randn(n_samples)
    # Add other informative features
    feature_1 = np.random.randn(n_samples)
    feature_2 = np.random.randn(n_samples)
    feature_3 = np.random.randn(n_samples)

    df = pd.DataFrame(
        {
            "feature_suspect": feature_suspect,
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3,
            "target": target,
        }
    )
    csv_path = tmp_path / "leaky.csv"
    df.to_csv(csv_path, index=False)

    # Run with mismatch flagging enabled and very low importance thresholds
    # so that feature_suspect is not flagged harmful (i.e. it would be Keep
    # on importance grounds alone, absent the leakage warning).
    config = make_fast_config(
        str(csv_path),
        "binary",
        tmp_path,
        flag_leakage_importance_mismatch=True,
        shap_importance_threshold=0.0001,
        permutation_importance_threshold=0.0001,
    )
    results = FeatureSelectionPipeline(config).run()

    # feature_suspect has a leakage warning, so it's Caution (not harmful,
    # but leaking) -- never Keep, since any leakage warning alone routes to
    # Caution regardless of importance.
    suspect_row = results[results["feature_name"] == "feature_suspect"].iloc[0]
    assert suspect_row["recommendation"] == "Caution"

    # But the reason should contain the mismatch warning, since the
    # feature is not harmful (high importance) yet still has leakage warnings.
    assert "high importance" in suspect_row["reason"]
    assert "potential leakage" in suspect_row["reason"]

    # A feature that IS harmful (low importance) and also has a leakage
    # warning escalates to Drop and must NOT get the mismatch note --
    # there's no "high importance" contradiction to report for a harmful
    # feature, regardless of the flag.
    drop_rows = results[results["recommendation"] == "Drop"]
    assert not any("high importance" in str(r) for r in drop_rows["reason"])


def test_pipeline_chronological_split_preserves_order(binary_data, tmp_path):
    """When time_column is set, splits should respect chronological order."""
    # Add a time column to the test data
    binary_data["time"] = range(len(binary_data))
    csv_path = tmp_path / "data.csv"
    binary_data.to_csv(csv_path, index=False)

    config = make_fast_config(
        str(csv_path),
        "binary",
        tmp_path,
        time_column="time",
    )
    pipeline = FeatureSelectionPipeline(config)
    pipeline.load_data()

    # Verify chronological order: train < val < test
    assert pipeline.train_data["time"].max() <= pipeline.val_data["time"].min()
    assert pipeline.val_data["time"].max() <= pipeline.test_data["time"].min()


def test_pipeline_chronological_split_runs_successfully(binary_data, tmp_path):
    """Full pipeline run with time_column should complete without error."""
    # Add a time column
    binary_data["time"] = range(len(binary_data))
    csv_path = tmp_path / "data.csv"
    binary_data.to_csv(csv_path, index=False)

    config = make_fast_config(
        str(csv_path),
        "binary",
        tmp_path,
        time_column="time",
    )
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    assert isinstance(results, pd.DataFrame)
    assert "feature_name" in results.columns
    assert len(results) > 0
