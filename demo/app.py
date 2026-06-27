"""Minimal interactive Gradio demo for TreeShaker.

Run locally with:

    python demo/app.py

Requires the ``demo`` optional dependency group:

    pip install -e ".[demo]"
"""

import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from treeshaker import Config, FeatureSelectionPipeline, TreeShakerError

TASK_CHOICES = ["regression", "binary", "multiclass"]


def load_columns(file_path: str | None) -> gr.Dropdown:
    """Populate the target-column dropdown from the uploaded CSV's headers."""
    if not file_path:
        return gr.Dropdown(choices=[], value=None)
    try:
        columns = list(pd.read_csv(file_path, nrows=0).columns)
    except Exception:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=columns, value=columns[0] if columns else None)


def run_pipeline(
    file_path: str | None,
    target_column: str | None,
    task: str,
) -> tuple[pd.DataFrame, str]:
    """Run FeatureSelectionPipeline on the uploaded CSV and return results."""
    if not file_path:
        return pd.DataFrame(), "Please upload a CSV file."
    if not target_column:
        return pd.DataFrame(), "Please select a target column."

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = str(Path(tmp_dir) / "treeshaker_results.csv")
        try:
            config = Config(
                input_file=file_path,
                target_column=target_column,
                task=task,
                output_file=output_file,
                # Keep the demo fast: skip Optuna tuning and use a small
                # number of permutations so results come back quickly.
                skip_tuning=True,
            )
            pipeline = FeatureSelectionPipeline(config)
            results = pipeline.run()
        except TreeShakerError as exc:
            return pd.DataFrame(), f"TreeShaker error: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface any error to the UI
            return pd.DataFrame(), f"Unexpected error: {exc}"

    return results, f"Done. {len(results)} feature(s) evaluated."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="TreeShaker Demo") as demo:
        gr.Markdown(
            "# TreeShaker\n"
            "Upload a CSV, pick a target column and task type, then run "
            "automated feature selection. This is a minimal sandbox — "
            "see the [README](https://github.com/SeanPeppers/TreeShaker) "
            "for the full Python API and CLI."
        )

        with gr.Row():
            file_input = gr.File(label="CSV file", file_types=[".csv"], type="filepath")
            target_dropdown = gr.Dropdown(label="Target column", choices=[])
            task_dropdown = gr.Dropdown(
                label="Task type", choices=TASK_CHOICES, value="regression"
            )

        run_button = gr.Button("Run", variant="primary")
        status = gr.Markdown()
        results_table = gr.Dataframe(label="Feature recommendations", wrap=True)

        file_input.change(fn=load_columns, inputs=file_input, outputs=target_dropdown)
        run_button.click(
            fn=run_pipeline,
            inputs=[file_input, target_dropdown, task_dropdown],
            outputs=[results_table, status],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
