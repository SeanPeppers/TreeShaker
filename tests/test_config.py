"""Tests for config.py dataclasses and validation."""

import pytest

from treeshaker.config import Config, OptunaConfig, XGBoostConfig
from treeshaker.exceptions import ConfigError


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
