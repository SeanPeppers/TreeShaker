import numpy as np
import pandas as pd
import xgboost as xgb

from treeshaker.cli import parse_arguments
from treeshaker.config import Config
from treeshaker.pipeline import (
    FeatureSelectionPipeline,
    _make_xgb_model,
    run_boruta_shap,
)


def test_use_boruta_config_default():
    config = Config(input_file="test.csv", target_column="target", task="binary")
    assert config.use_boruta is False


def test_use_boruta_cli_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "treeshaker",
            "--input",
            "test.csv",
            "--target",
            "target",
            "--task",
            "binary",
            "--use-boruta",
        ],
    )
    args = parse_arguments()
    assert args.use_boruta is True


def test_run_boruta_shap_execution():
    np.random.seed(42)
    n = 100
    signal = np.random.randn(n)
    noise = np.random.randn(n)
    target = (signal > 0).astype(int)

    df = pd.DataFrame({"signal": signal, "noise": noise, "target": target})
    model = _make_xgb_model(
        "binary",
        {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
            "verbosity": 0,
            "enable_categorical": True,
        },
    )
    model.fit(df[["signal", "noise"]], target)

    decisions = run_boruta_shap(
        model=model,
        data=df,
        target_column="target",
        feature_names=["signal", "noise"],
        task="binary",
        random_seed=42,
        n_iterations=10,
    )

    assert len(decisions) == 2
    for dec in decisions:
        assert dec in {"Confirmed", "Tentative", "Rejected"}


def test_run_boruta_shap_regression():
    np.random.seed(42)
    n = 100
    signal = np.random.randn(n)
    target = signal * 2.0 + np.random.randn(n) * 0.1

    df = pd.DataFrame({"signal": signal, "target": target})
    model = _make_xgb_model(
        "regression",
        {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
            "verbosity": 0,
        },
    )
    model.fit(df[["signal"]], target)

    decisions = run_boruta_shap(
        model=model,
        data=df,
        target_column="target",
        feature_names=["signal"],
        task="regression",
        random_seed=42,
        n_iterations=5,
    )

    assert len(decisions) == 1
    assert decisions[0] in {"Confirmed", "Tentative", "Rejected"}


def test_run_boruta_shap_empty_features():
    df = pd.DataFrame({"target": [0, 1]})
    model = xgb.XGBClassifier()
    decisions = run_boruta_shap(
        model=model,
        data=df,
        target_column="target",
        feature_names=[],
        task="binary",
    )
    assert decisions == []


def test_pipeline_with_boruta(tmp_path):
    csv_path = tmp_path / "data.csv"
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
            "target": np.random.choice([0, 1], size=50),
        }
    )
    df.to_csv(csv_path, index=False)

    config = Config(
        input_file=str(csv_path),
        target_column="target",
        task="binary",
        use_boruta=True,
        skip_tuning=True,
    )
    pipeline = FeatureSelectionPipeline(config)
    results = pipeline.run()

    assert "boruta_decision" in results.columns
    assert len(results["boruta_decision"]) == 2
