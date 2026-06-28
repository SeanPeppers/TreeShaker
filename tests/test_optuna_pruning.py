"""Tests for Optuna pruning and early stopping configuration and integration."""

from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.mcp_server import _build_config, list_config_schema
from treeshaker.pipeline import FeatureSelectionPipeline


def test_optuna_pruning_config_default():
    """Verify use_optuna_pruning defaults to False in Config."""
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.use_optuna_pruning is False


def test_optuna_pruning_config_set():
    """Verify use_optuna_pruning can be enabled in Config."""
    config = Config(
        input_file="test.csv",
        target_column="target",
        task="binary",
        use_optuna_pruning=True,
    )
    assert config.use_optuna_pruning is True


def test_cli_optuna_pruning_flag():
    """Verify --use-optuna-pruning CLI flag sets use_optuna_pruning=True."""
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
            "--use-optuna-pruning",
        ],
    ):
        args = parse_arguments()
        assert args.use_optuna_pruning is True


def test_mcp_server_optuna_pruning():
    """Verify MCP server schema and _build_config support use_optuna_pruning."""
    schema = list_config_schema()
    field_names = [f["name"] for f in schema["fields"]]
    assert "use_optuna_pruning" in field_names

    cfg = _build_config(
        input_path="data.csv",
        target_column="y",
        task="binary",
        output_path="out.csv",
        use_optuna_pruning=True,
    )
    assert cfg.use_optuna_pruning is True


def test_pipeline_tune_hyperparams_with_pruning(tmp_path):
    """Verify pipeline tune_hyperparams executes with use_optuna_pruning enabled."""
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 2,
            "feat2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5] * 2,
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
        }
    )
    df.to_csv(csv_file, index=False)

    config = Config(
        input_file=str(csv_file),
        target_column="target",
        task="binary",
        use_optuna_pruning=True,
    )
    config.optuna_config.n_trials = 2
    pipeline = FeatureSelectionPipeline(config)
    pipeline.load_data()
    with patch("treeshaker.utils.detect_cuda_available", return_value=False):
        best_params = pipeline.tune_hyperparams()
    assert isinstance(best_params, dict)


def test_pipeline_tune_hyperparams_pruning_occurs(tmp_path):
    """Verify that TrialPruned occurs when pruning is enabled with a bad trial."""
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 2,
            "feat2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5] * 2,
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
        }
    )
    df.to_csv(csv_file, index=False)

    config = Config(
        input_file=str(csv_file),
        target_column="target",
        task="binary",
        use_optuna_pruning=True,
    )
    config.optuna_config.n_trials = 6
    pipeline = FeatureSelectionPipeline(config)
    pipeline.load_data()

    captured_studies = []
    orig_create_study = optuna.create_study

    def mock_create_study(*args, **kwargs):
        study = orig_create_study(*args, **kwargs)
        captured_studies.append(study)
        return study

    trial_counter = [0]

    def mock_make_xgb_model(task, params):
        model = MagicMock()
        t_idx = trial_counter[0]
        trial_counter[0] += 1

        def mock_fit(X, y, **fit_kwargs):
            callbacks = params.get("callbacks", [])
            val_loss = 0.1 if t_idx < 5 else 10.0
            if callbacks:
                cb = callbacks[0]
                for epoch in range(3):
                    cb.after_iteration(
                        model, epoch, {"validation_0": {"logloss": [val_loss]}}
                    )
            model.predict_proba.side_effect = lambda X_eval: np.array(
                [[0.9, 0.1]] * len(X_eval)
            )

        model.fit.side_effect = mock_fit
        return model

    with (
        patch("optuna.create_study", side_effect=mock_create_study),
        patch("treeshaker.pipeline._make_xgb_model", side_effect=mock_make_xgb_model),
    ):
        pipeline.tune_hyperparams()

    study = captured_studies[-1]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    assert len(pruned) > 0


def test_pipeline_tune_hyperparams_without_pruning_regression(tmp_path):
    """Verify that with use_optuna_pruning=False, no pruner/callback is attached."""
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 2,
            "feat2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5] * 2,
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
        }
    )
    df.to_csv(csv_file, index=False)

    config = Config(
        input_file=str(csv_file),
        target_column="target",
        task="binary",
        use_optuna_pruning=False,
    )
    config.optuna_config.n_trials = 2
    pipeline = FeatureSelectionPipeline(config)
    pipeline.load_data()

    captured_studies = []
    orig_create_study = optuna.create_study

    def mock_create_study(*args, **kwargs):
        study = orig_create_study(*args, **kwargs)
        captured_studies.append(study)
        return study

    make_xgb_params = []

    def mock_make_xgb_model(task, params):
        make_xgb_params.append(params)
        model = MagicMock()

        def mock_fit(X, y, **fit_kwargs):
            model.predict_proba.side_effect = lambda X_eval: np.array(
                [[0.9, 0.1]] * len(X_eval)
            )

        model.fit.side_effect = mock_fit
        return model

    with (
        patch("optuna.create_study", side_effect=mock_create_study),
        patch("treeshaker.pipeline._make_xgb_model", side_effect=mock_make_xgb_model),
    ):
        pipeline.tune_hyperparams()

    study = captured_studies[-1]
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    for params in make_xgb_params:
        assert "callbacks" not in params
        assert "early_stopping_rounds" not in params
