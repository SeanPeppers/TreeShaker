"""Tests for the TreeShaker MCP server tool functions.

These call the underlying tool functions directly as plain Python functions
(no MCP client/transport involved), since that's what's actually testable
without a live MCP session.
"""

import os

import pandas as pd
import pytest

from treeshaker.mcp_server import (
    _build_config,
    get_recommendations,
    inspect_dataset,
    list_config_schema,
    run_pipeline,
)


def test_run_pipeline_binary(binary_csv, tmp_path):
    output_path = str(tmp_path / "results")
    result = run_pipeline(
        input_path=binary_csv,
        target_column="target",
        task="binary",
        output_path=output_path,
        skip_tuning=True,
    )

    assert "error" not in result
    assert result["feature_count"] == 10
    assert result["keep_count"] + result["drop_count"] + result["caution_count"] == 10
    assert isinstance(result["top_features"], list)
    assert len(result["top_features"]) <= 10
    for row in result["top_features"]:
        assert {"feature_name", "shap_importance", "recommendation"} <= row.keys()
    assert isinstance(result["warnings"], list)
    assert "results" in result["output_files"]
    assert os.path.exists(result["output_files"]["results"])


def test_mcp_server_skip_leakage_check():
    """Verify MCP server schema and _build_config support skip_leakage_check.

    Regression test: the flag was previously documented but never wired
    into mcp_server.py's run_pipeline/_build_config at all.
    """
    schema = list_config_schema()
    field_names = [f["name"] for f in schema["fields"]]
    assert "skip_leakage_check" in field_names

    cfg = _build_config(
        input_path="data.csv",
        target_column="y",
        task="binary",
        output_path="out.csv",
        skip_leakage_check=True,
    )
    assert cfg.skip_leakage_check is True


def test_build_config_wires_dfe_loop_feedback():
    """DFE loop-feedback / operator knobs must reach synthetic_config."""
    cfg = _build_config(
        input_path="data.csv",
        target_column="y",
        task="binary",
        output_path="out.csv",
        synthesize_features=True,
        synthesis_top_n=3,
        synthesis_mi_threshold=0.05,
        synthesis_operators=["product"],
        dfe_loop_feedback=True,
        dfe_max_inject=2,
    )
    assert cfg.synthetic_config.synthesize_from_interactions is True
    assert cfg.synthetic_config.interaction_top_n == 3
    assert cfg.synthetic_config.mi_threshold == 0.05
    assert cfg.synthetic_config.operators == ["product"]
    assert cfg.synthetic_config.dfe_loop_feedback is True
    assert cfg.synthetic_config.dfe_max_inject == 2


def test_run_pipeline_regression(regression_csv, tmp_path):
    output_path = str(tmp_path / "reg_results")
    result = run_pipeline(
        input_path=regression_csv,
        target_column="target",
        task="regression",
        output_path=output_path,
        skip_tuning=True,
    )

    assert "error" not in result
    assert result["feature_count"] == 20


def test_run_pipeline_with_config_overrides(binary_csv, tmp_path):
    output_path = str(tmp_path / "results_override")
    result = run_pipeline(
        input_path=binary_csv,
        target_column="target",
        task="binary",
        output_path=output_path,
        skip_tuning=True,
        config_overrides={"shap_importance_threshold": 0.5},
    )

    assert "error" not in result
    # A much higher threshold should flag more features as harmful/dropped.
    assert result["drop_count"] >= 1


def test_run_pipeline_bad_input_path(tmp_path):
    result = run_pipeline(
        input_path=str(tmp_path / "does_not_exist.csv"),
        target_column="target",
        task="binary",
        output_path=str(tmp_path / "out"),
    )

    assert "error" in result
    assert isinstance(result["error"], str)
    assert "does_not_exist.csv" in result["error"]


def test_run_pipeline_bad_task(binary_csv, tmp_path):
    result = run_pipeline(
        input_path=binary_csv,
        target_column="target",
        task="not_a_real_task",
        output_path=str(tmp_path / "out"),
    )

    assert "error" in result


def test_run_pipeline_unknown_config_override_key(binary_csv, tmp_path):
    result = run_pipeline(
        input_path=binary_csv,
        target_column="target",
        task="binary",
        output_path=str(tmp_path / "out"),
        config_overrides={"this_field_does_not_exist": 123},
    )

    assert "error" in result
    assert "this_field_does_not_exist" in result["error"]


def test_get_recommendations(binary_csv, tmp_path):
    output_path = str(tmp_path / "rec_results")
    run_result = run_pipeline(
        input_path=binary_csv,
        target_column="target",
        task="binary",
        output_path=output_path,
        skip_tuning=True,
    )
    assert "error" not in run_result

    rec = get_recommendations(run_result["output_files"]["results"])
    assert "error" not in rec
    assert "features" in rec
    assert len(rec["features"]) == 10
    for row in rec["features"]:
        assert "feature_name" in row
        assert "recommendation" in row


def test_get_recommendations_missing_file(tmp_path):
    rec = get_recommendations(str(tmp_path / "missing_results.csv"))
    assert "error" in rec


def test_get_recommendations_html(binary_csv, tmp_path):
    output_path = str(tmp_path / "rec_results.html")
    run_result = run_pipeline(
        input_path=binary_csv,
        target_column="target",
        task="binary",
        output_path=output_path,
        skip_tuning=True,
    )
    assert "error" not in run_result

    rec = get_recommendations(run_result["output_files"]["results"])
    assert "error" not in rec
    assert "features" in rec
    assert len(rec["features"]) == 10


def test_inspect_dataset(binary_csv):
    result = inspect_dataset(binary_csv)

    assert "error" not in result
    assert result["row_count"] == 200
    assert result["column_count"] == 11
    names = {c["name"] for c in result["columns"]}
    assert "target" in names
    for col in result["columns"]:
        assert "dtype" in col
        assert "missing_rate" in col
        assert "n_unique" in col
        assert "suggestion" in col


def test_inspect_dataset_free_text_column(tmp_path):
    df = pd.DataFrame(
        {
            "id": range(50),
            "review": [
                f"this is a fairly long unique review number {i} here"
                for i in range(50)
            ],
            "target": [i % 2 for i in range(50)],
        }
    )
    path = tmp_path / "freetext.csv"
    df.to_csv(path, index=False)

    result = inspect_dataset(str(path))
    assert "error" not in result
    review_col = next(c for c in result["columns"] if c["name"] == "review")
    assert "free text" in review_col["suggestion"]


def test_inspect_dataset_missing_file(tmp_path):
    result = inspect_dataset(str(tmp_path / "nope.csv"))
    assert "error" in result


def test_list_config_schema_has_known_fields():
    schema = list_config_schema()
    assert "fields" in schema
    names = {f["name"] for f in schema["fields"]}

    # Known top-level Config fields (config.py).
    assert "check_residuals" in names
    assert "compute_shap_interactions" in names
    assert "task" in names
    assert "input_file" in names
    assert "target_column" in names
    assert "bootstrap_config" in names
    assert "synthetic_config" in names


def test_list_config_schema_expands_nested_dataclasses():
    schema = list_config_schema()
    bootstrap_field = next(
        f for f in schema["fields"] if f["name"] == "bootstrap_config"
    )
    assert "fields" in bootstrap_field
    nested_names = {f["name"] for f in bootstrap_field["fields"]}
    assert "n_bootstrap" in nested_names
    assert "sample_frac" in nested_names


@pytest.mark.parametrize("field_name", ["xgboost_config", "synthetic_config"])
def test_list_config_schema_expands_other_nested_dataclasses(field_name):
    schema = list_config_schema()
    field = next(f for f in schema["fields"] if f["name"] == field_name)
    assert "fields" in field
    assert len(field["fields"]) > 0
