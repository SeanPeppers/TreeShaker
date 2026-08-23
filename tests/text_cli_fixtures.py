"""Importable predict_fn/tokenizer fixtures for test_cli_text.py's dotted-path
loading (--predict-fn/--tokenizer resolve via 'module:attribute', which needs
a real importable module on disk -- can't be an inline test function)."""

import numpy as np


def predict(texts: list[str]) -> np.ndarray:
    return np.array([1.0 if "good" in t else 0.0 for t in texts])


not_callable = "this is not a function"
