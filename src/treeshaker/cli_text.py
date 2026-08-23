"""Command-line interface for token-level text/transformer importance.

Separate entry point from the main tabular `treeshaker` CLI (see
text_importance.py's module docstring for why: text has no CSV/feature-name
shape to reuse). A CLI can't accept a live Python callable directly, so
predict_fn (and optionally tokenizer) are loaded from a "module:attribute"
dotted path -- the same "bring your own model, no NN framework dependency"
contract as Config.custom_model, just resolved from a string instead of
passed in-process.
"""

import argparse
import importlib
import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from .exceptions import TreeShakerError
from .text_importance import TextConfig, TextImportancePipeline


def _load_dotted_path(spec: str) -> Any:
    """Import "module.submodule:attribute" and return the attribute.

    Same convention as gunicorn's `app:app` / setuptools console_scripts
    entry points -- familiar to anyone who's shipped a Python CLI tool.
    """
    if ":" not in spec:
        raise ValueError(
            f"{spec!r} is not a valid 'module:attribute' path (missing ':')."
        )
    module_name, attr_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"Module {module_name!r} has no attribute {attr_name!r}."
        ) from exc


def _load_texts(path: str, text_column: str | None) -> list[str]:
    """Load texts from a plain .txt file (one per line) or a CSV column."""
    if path.lower().endswith(".csv"):
        if not text_column:
            raise ValueError("--text-column is required when --texts-file is a CSV.")
        df = pd.read_csv(path)
        if text_column not in df.columns:
            raise ValueError(f"Column {text_column!r} not found in {path!r}.")
        return [str(t) for t in df[text_column]]
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="treeshaker-text",
        description=(
            "Token-level SHAP attribution for text/transformer models"
            " (which tokens drove each prediction)."
        ),
    )
    parser.add_argument(
        "--predict-fn",
        required=True,
        help=(
            "Dotted path 'module:function' to a callable taking"
            " list[str] -> np.ndarray of shape (n_texts,) or"
            " (n_texts, n_classes). E.g. 'my_model:predict'."
        ),
    )
    parser.add_argument(
        "--texts-file",
        required=True,
        help="Path to a .txt file (one text per line) or a .csv file.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Column name holding the text, required when --texts-file is a CSV.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Optional dotted path 'module:tokenizer' passed to"
            " shap.maskers.Text (e.g. a Hugging Face tokenizer). Omit to"
            " use SHAP's own whitespace/punctuation tokenizer."
        ),
    )
    parser.add_argument(
        "--output",
        default="text_results.csv",
        help="Output file path. (default: text_results.csv)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top tokens (by mean_abs_shap) to keep. (default: 20)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug-level logging."
    )
    return parser.parse_args()


def main() -> int:
    """
    Entry point for the treeshaker-text CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    args = parse_arguments()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        predict_fn: Callable[[list[str]], np.ndarray] = _load_dotted_path(
            args.predict_fn
        )
        tokenizer = _load_dotted_path(args.tokenizer) if args.tokenizer else None
        texts = _load_texts(args.texts_file, args.text_column)
    except (ValueError, ImportError, OSError) as exc:
        logger.error(f"Configuration error: {exc}")
        return 1

    try:
        pipeline = TextImportancePipeline(
            predict_fn=predict_fn,
            texts=texts,
            config=TextConfig(
                tokenizer=tokenizer, top_n=args.top_n, output_file=args.output
            ),
        )
        pipeline.run()
        print(f"Results saved to: {args.output}")
        return 0
    except TreeShakerError as exc:
        logger.error(f"Pipeline error: {exc}")
        return 1
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
