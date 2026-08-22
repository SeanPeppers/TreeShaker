"""Tests for the confidence heuristic score."""

import numpy as np
from conftest import make_config as make_fast_config

from treeshaker.confidence import compute_confidence_scores
from treeshaker.config import Config
from treeshaker.pipeline import FeatureSelectionPipeline


def test_confidence_scores_function_range_and_logic() -> None:
    # Direct test for compute_confidence_scores logic
    shap_importance = np.array([0.05, 0.01, 0.005, 0.1, 0.011])
    perm_importance = np.array([0.1, 0.05, 0.01, 0.5, 0.049])
    recommendations = ["Keep", "Caution", "Drop", "Keep", "Drop"]
    shap_threshold = 0.01
    perm_threshold = 0.05

    scores = compute_confidence_scores(
        shap_importance=shap_importance,
        perm_importance=perm_importance,
        recommendations=recommendations,
        shap_threshold=shap_threshold,
        perm_threshold=perm_threshold,
    )

    # Check shape and type
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 5
    assert scores.dtype == float

    # Verify score range is [0.5, 1.0]
    assert np.all(scores >= 0.5)
    assert np.all(scores <= 1.0)

    # Caution recommendation must result in exactly 0.5
    assert scores[1] == 0.5

    # Features far from thresholds should score near 1.0
    # shap_importance[3]=0.1 is 10x shap_threshold,
    # perm_importance[3]=0.5 is 10x perm_threshold
    assert scores[3] > 0.95

    # Borderline features (distance near 0) should score near 0.5
    # shap_importance[4]=0.011 is very close to shap_threshold=0.01,
    # so distance is small
    assert 0.5 < scores[4] < 0.6


def test_disabled_by_default() -> None:
    # Check default config value
    cfg = Config(input_file="dummy.csv", target_column="y", task="regression")
    assert cfg.compute_confidence_scores is False


def test_confidence_score_column_appears_when_enabled(binary_csv, tmp_path) -> None:
    # Test pipeline with confidence score disabled
    config_disabled = make_fast_config(binary_csv, "binary", tmp_path, skip_tuning=True)
    pipeline_disabled = FeatureSelectionPipeline(config_disabled)
    results_disabled = pipeline_disabled.run()
    assert "confidence_score" not in results_disabled.columns

    # Test pipeline with confidence score enabled
    config_enabled = make_fast_config(
        binary_csv,
        "binary",
        tmp_path,
        skip_tuning=True,
        compute_confidence_scores=True,
    )
    pipeline_enabled = FeatureSelectionPipeline(config_enabled)
    results_enabled = pipeline_enabled.run()

    # Verify column presence and range of values
    assert "confidence_score" in results_enabled.columns
    scores = results_enabled["confidence_score"]
    assert scores.dtype in (np.float64, np.float32, float)
    assert (scores >= 0.5).all()
    assert (scores <= 1.0).all()
