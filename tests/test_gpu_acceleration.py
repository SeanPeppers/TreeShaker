"""Tests for GPU acceleration configuration, CLI, and pipeline wiring."""

from unittest.mock import patch

import pytest

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.exceptions import ConfigError
from treeshaker.mcp_server import _build_config, list_config_schema
from treeshaker.pipeline import _build_xgb_fit_params
from treeshaker.utils import resolve_xgboost_device


def test_gpu_config_default():
    """Verify use_gpu defaults to False in Config."""
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.use_gpu is False


def test_gpu_config_raise_when_cuda_unavailable():
    """Verify Config raises ConfigError if use_gpu=True but CUDA is unavailable."""
    with patch("treeshaker.config.detect_cuda_available", return_value=False):
        with pytest.raises(ConfigError, match="use_gpu=True was explicitly requested"):
            Config(
                input_file="test.csv",
                target_column="target",
                task="binary",
                use_gpu=True,
            )


def test_gpu_config_valid_when_cuda_available():
    """Verify Config succeeds if use_gpu=True and CUDA is available."""
    with patch("treeshaker.config.detect_cuda_available", return_value=True):
        config = Config(
            input_file="test.csv",
            target_column="target",
            task="binary",
            use_gpu=True,
        )
        assert config.use_gpu is True


def test_resolve_xgboost_device_with_use_gpu():
    """Verify resolve_xgboost_device returns 'cuda' when use_gpu=True."""
    assert resolve_xgboost_device("cpu", use_gpu=True) == "cuda"
    assert resolve_xgboost_device("auto", use_gpu=True) == "cuda"


def test_build_xgb_fit_params_gpu():
    """Verify _build_xgb_fit_params sets GPU parameters when use_gpu=True."""
    with patch("treeshaker.config.detect_cuda_available", return_value=True):
        config = Config(
            input_file="test.csv",
            target_column="target",
            task="binary",
            use_gpu=True,
        )
        params = _build_xgb_fit_params({}, config)
        assert params["device"] == "cuda"

    with (
        patch("treeshaker.config.detect_cuda_available", return_value=False),
        patch("treeshaker.utils.detect_cuda_available", return_value=False),
    ):
        config_no_gpu = Config(
            input_file="test.csv",
            target_column="target",
            task="binary",
            use_gpu=False,
        )
        params_no_gpu = _build_xgb_fit_params({}, config_no_gpu)
        assert params_no_gpu["device"] != "cuda"


def test_cli_use_gpu_flag():
    """Verify --use-gpu CLI flag sets use_gpu=True in parsed args."""
    with patch(
        "sys.argv",
        [
            "treeshaker",
            "--input",
            "data.csv",
            "--target",
            "y",
            "--task",
            "binary",
            "--use-gpu",
        ],
    ):
        args = parse_arguments()
        assert args.use_gpu is True


def test_mcp_server_use_gpu():
    """Verify MCP server schema and _build_config support use_gpu."""
    schema = list_config_schema()
    field_names = [f["name"] for f in schema["fields"]]
    assert "use_gpu" in field_names

    with patch("treeshaker.config.detect_cuda_available", return_value=True):
        cfg = _build_config(
            input_path="data.csv",
            target_column="y",
            task="binary",
            output_path="out.csv",
            use_gpu=True,
        )
        assert cfg.use_gpu is True
