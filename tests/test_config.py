"""Tests for config.py dataclasses and validation."""

import pytest

from treeshaker.config import Config, OptunaConfig, XGBoostConfig
from treeshaker.exceptions import ConfigError
from treeshaker.synthetic_features import SyntheticConfig


def test_xgboost_config_to_dict_includes_extra_params():
    cfg = XGBoostConfig(max_depth=8, extra_params={"tree_method": "hist"})
    d = cfg.to_dict()
    assert d["max_depth"] == 8
    assert d["tree_method"] == "hist"
    assert "extra_params" not in d


def test_xgboost_config_defaults():
    cfg = XGBoostConfig()
    d = cfg.to_dict()
    assert d["max_depth"] == 6
    assert d["learning_rate"] == 0.1


def test_xgboost_config_device_defaults_to_auto():
    cfg = XGBoostConfig()
    assert cfg.device == "auto"


def test_xgboost_config_auto_resolves_to_cuda_when_available(monkeypatch):
    monkeypatch.setattr("treeshaker.utils.detect_cuda_available", lambda: True)
    cfg = XGBoostConfig(device="auto")
    d = cfg.to_dict()
    assert d["device"] == "cuda"


def test_xgboost_config_auto_resolves_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr("treeshaker.utils.detect_cuda_available", lambda: False)
    cfg = XGBoostConfig(device="auto")
    d = cfg.to_dict()
    assert d["device"] == "cpu"


def test_xgboost_config_explicit_cpu_passes_through_regardless_of_detection(
    monkeypatch,
):
    monkeypatch.setattr("treeshaker.utils.detect_cuda_available", lambda: True)
    cfg = XGBoostConfig(device="cpu")
    d = cfg.to_dict()
    assert d["device"] == "cpu"

    monkeypatch.setattr("treeshaker.utils.detect_cuda_available", lambda: False)
    cfg2 = XGBoostConfig(device="cpu")
    d2 = cfg2.to_dict()
    assert d2["device"] == "cpu"


def test_xgboost_config_explicit_cuda_succeeds_when_detected(monkeypatch):
    monkeypatch.setattr("treeshaker.config.detect_cuda_available", lambda: True)
    cfg = XGBoostConfig(device="cuda")
    d = cfg.to_dict()
    assert d["device"] == "cuda"


def test_xgboost_config_explicit_cuda_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr("treeshaker.config.detect_cuda_available", lambda: False)
    with pytest.raises(ConfigError, match="cuda"):
        XGBoostConfig(device="cuda")


def test_xgboost_config_invalid_device_raises():
    with pytest.raises(ConfigError, match="device"):
        XGBoostConfig(device="tpu")


def test_optuna_config_defaults():
    cfg = OptunaConfig()
    assert cfg.n_trials == 50
    assert cfg.timeout == 300
    assert cfg.seed == 42


def test_config_valid_tasks():
    for task in ("binary", "multiclass", "regression"):
        cfg = Config(input_file="f.csv", target_column="y", task=task)
        assert cfg.task == task


def test_config_invalid_task_raises():
    with pytest.raises(ConfigError, match="task must be one of"):
        Config(input_file="f.csv", target_column="y", task="unsupported")


def test_config_invalid_test_size_raises():
    with pytest.raises(ConfigError, match="test_size"):
        Config(input_file="f.csv", target_column="y", task="binary", test_size=1.5)


def test_config_invalid_val_size_raises():
    with pytest.raises(ConfigError, match="val_size"):
        Config(input_file="f.csv", target_column="y", task="binary", val_size=1.5)


def test_config_test_plus_val_size_too_large_raises():
    with pytest.raises(ConfigError, match="test_size \\+ val_size"):
        Config(
            input_file="f.csv",
            target_column="y",
            task="binary",
            test_size=0.6,
            val_size=0.5,
        )


def test_config_test_plus_val_size_exactly_one_raises():
    with pytest.raises(ConfigError, match="test_size \\+ val_size"):
        Config(
            input_file="f.csv",
            target_column="y",
            task="binary",
            test_size=0.5,
            val_size=0.5,
        )


def test_config_invalid_format_raises():
    with pytest.raises(ConfigError, match="output_format"):
        Config(
            input_file="f.csv",
            target_column="y",
            task="binary",
            output_format="json",
        )


def test_config_defaults():
    cfg = Config(input_file="f.csv", target_column="y", task="regression")
    assert cfg.output_file == "treeshaker_results.csv"
    assert cfg.output_format == "csv"
    assert cfg.test_size == 0.2
    assert cfg.val_size == 0.2
    assert cfg.random_seed == 42
    assert cfg.shap_importance_threshold == 0.01
    assert cfg.permutation_importance_threshold == 0.05
    assert cfg.n_permutations == 5
    assert cfg.permutation_metric is None
    assert cfg.shap_sample_size is None


def test_config_invalid_n_permutations_raises():
    with pytest.raises(ConfigError, match="n_permutations"):
        Config(input_file="f.csv", target_column="y", task="binary", n_permutations=0)


def test_config_shap_sample_size_default_none():
    cfg = Config(input_file="f.csv", target_column="y", task="binary")
    assert cfg.shap_sample_size is None


def test_config_shap_sample_size_accepts_positive_int():
    cfg = Config(
        input_file="f.csv", target_column="y", task="binary", shap_sample_size=500
    )
    assert cfg.shap_sample_size == 500


def test_config_invalid_shap_sample_size_raises():
    with pytest.raises(ConfigError, match="shap_sample_size"):
        Config(input_file="f.csv", target_column="y", task="binary", shap_sample_size=0)


def test_config_valid_permutation_metric_accepted():
    for metric in ("roc_auc", "accuracy", "neg_mse", "wape"):
        cfg = Config(
            input_file="f.csv",
            target_column="y",
            task="binary",
            permutation_metric=metric,
        )
        assert cfg.permutation_metric == metric


def test_config_invalid_permutation_metric_raises():
    with pytest.raises(ConfigError, match="permutation_metric"):
        Config(
            input_file="f.csv",
            target_column="y",
            task="binary",
            permutation_metric="bogus",
        )


def test_config_synthetic_config_defaults():
    cfg = Config(input_file="f.csv", target_column="y", task="regression")
    assert cfg.synthetic_config.synthesize_from_interactions is False
    assert cfg.synthetic_config.interaction_top_n == 5
    assert cfg.synthetic_config.mi_threshold == 0.01


def test_config_invalid_synthetic_mi_threshold_raises():
    with pytest.raises(ConfigError, match="mi_threshold"):
        Config(
            input_file="f.csv",
            target_column="y",
            task="binary",
            synthetic_config=SyntheticConfig(mi_threshold=-0.1),
        )


def test_config_invalid_synthetic_interaction_top_n_raises():
    with pytest.raises(ConfigError, match="interaction_top_n"):
        Config(
            input_file="f.csv",
            target_column="y",
            task="binary",
            synthetic_config=SyntheticConfig(interaction_top_n=0),
        )
